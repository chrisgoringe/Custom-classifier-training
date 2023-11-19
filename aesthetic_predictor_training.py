import torch
import torch.nn as nn
import clip, datasets
from PIL import Image
from safetensors.torch import save_file, load_file
import os, math, random
from src.time_context import Timer
from src.data_holder import DataHolder
from transformers import Trainer, TrainerControl, TrainerState, TrainingArguments, TrainerCallback

from arguments import training_args, args

class CLIP:
    def __init__(self, pretrained="ViT-L/14", device="cuda"):
        self.model, self.preprocess = clip.load(pretrained, device=device)
        self.device = device
        self.cached = {}

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.to(torch.float)
   
    def prepare_from_file(self, filename, device="cuda"):
        if filename not in self.cached:
            self.cached[filename] = self.get_image_features_tensor(Image.open(filename)).to(device)
        return self.cached[filename]

class AestheticPredictor(nn.Module):
    def __init__(self, input_size=768, pretrained="", device="cuda", dropouts=[0.2,0.2,0.1]):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024), 
            nn.Dropout(dropouts[0]),          
            nn.Linear(1024, 128),
            nn.Dropout(dropouts[1]),
            nn.Linear(128, 64),
            nn.Dropout(dropouts[2]),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
        if pretrained: self.load_state_dict(load_file(pretrained))
        self.add_relu()
        self.layers.to(device)
        self.device = device

    def add_relu(self):
        if args['relu']:
            self.layers = nn.Sequential(
                self.layers[0], nn.ReLU(),
                self.layers[1], self.layers[2], nn.ReLU(),
                self.layers[3], self.layers[4], nn.ReLU(),
                self.layers[5], self.layers[6], nn.ReLU(),
                self.layers[7]
            )

    def remove_relu(self):
        if args['relu']:
            self.layers = nn.Sequential(
                self.layers[0], self.layers[2], self.layers[3],
                self.layers[5], self.layers[6],
                self.layers[8], self.layers[9], self.layers[11]
            )

    def forward(self, x, **kwargs):
        return self.layers(x)
    
def evaluate_files(files, predictor:AestheticPredictor, clipper:CLIP):
    def score_file(f):
        return predictor(clipper.prepare_from_file(f)).item()
    scores = [(score_file(f), f) for f in files]
    scores.sort()
    return scores
        
def evaluate_directory(image_directory, predictor=None, clipper=None):
    if predictor is None:
        with Timer('load models'):
            predictor = AestheticPredictor(dropouts=args['aesthetic_model_dropouts'])
            clipper = CLIP()

    return evaluate_files([os.path.join(image_directory,f) for f in os.listdir(image_directory)], predictor, clipper)

    
class CustomTrainer(Trainer):
    def __init__(self, model, train_dataset, eval_dataset, weights=None, **kwargs):
        super().__init__(model, **kwargs)
        self.weights = weights
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def compute_loss(self, model, inputs, return_outputs=False):
        scores = inputs['y'].to(torch.float)
        outputs = torch.flatten( model(inputs['x'].to(torch.float)) )
        if self.weights:
            delta = torch.subtract(outputs,scores)
            dsquared = torch.multiply(torch.square(delta),self.apply_weight(scores))
            loss = torch.sum(dsquared)
        else:
            loss_fct = nn.MSELoss()
            loss = loss_fct(scores.reshape(outputs.shape), outputs)
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=4)
    
    def get_eval_dataloader(self, eds):
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size=4)
    
    def apply_weight(self, scores:torch.Tensor):
        return torch.tensor( list(self.weights[str(int(x))] for x in scores), device=scores.device, dtype=scores.dtype  )
    
class QuickDataset(torch.utils.data.Dataset):
    def __init__(self, dh:DataHolder, clipper:CLIP, split):
        df = dh.get_dataframe()
        df = df[df["split"] == split]
        self.features = [clipper.get_image_features_tensor(Image.open(f)) for f in df['image']]
        self.scores = torch.tensor([float(l) for l in df['label_str']])
        self.allfiles = [f for f in df['image']]
        self.map = list(range(len(self)))
        self.shuffle()

    def __getitem__(self, i):
        x = self.features[self.map[i]]
        y = self.scores[self.map[i]]
        return {"x":x, "y":y}

    def __len__(self):
        return len(self.features)
    
    def files(self, filter):
        return [x for x in self.allfiles if os.path.basename(os.path.split(x)[0])==filter]
    
    def shuffle(self):
        random.shuffle(self.map)
    
def prepare(df):
    ds = datasets.Dataset.from_pandas(df)
    prepared_ds = ds.cast_column("image", datasets.Image())
    return prepared_ds

def evaluate(labels, eds:QuickDataset, predictor:AestheticPredictor, clipper:CLIP, weights=None, prnt:str=None, prntrmse:str=None):
    ev = {}
    sumsquare = 0
    for x in labels:
        scores = evaluate_files(eds.files(x), predictor, clipper)
        ev[x] = sum(score[0] for score in scores)/len(scores) if len(scores)>0 else 0
        ss = sum((float(x)-score[0])*(float(x)-score[0]) for score in scores)
        sumsquare += ss*weights[x] if weights else ss
    for x in ev:
        if prnt: print(prnt.format(x,ev[x]))
    rmse = math.sqrt(sumsquare/len(eds))
    if prntrmse: print(prntrmse.format(rmse))
    return ev, rmse


def train_predictor():
    pretrained = os.path.join(args['load_model'],"model.safetensors") if ('load_model' in args and args['load_model']) else args['base_model']
    top_level_images = args['top_level_image_directory']

    with Timer('load models'):
        predictor = AestheticPredictor(pretrained=pretrained)
        clipper = CLIP()
    with Timer('load images'):
        data = DataHolder(top_level=top_level_images)
    with Timer('CLIP images'):
        tds = QuickDataset(data, clipper, "train")
        eds = QuickDataset(data, clipper, "test")

    weights = data.weights() if args['weight_category_loss'] else None

    evaluate(data.labels, eds, predictor, clipper, weights, "Start: folder {:>1} average score {:>5.3f}", "weighted eval mse loss {:>6.3}")

    class EvaluationCallback(TrainerCallback):
        def __init__(self, every, labels):
            self.last = 0
            self.every = every
            self.labels = labels

        def do_eval(self, state):
            print("\n\n====\nEpoch "+str(state.epoch)+": ")
            predictor.eval()
            evaluate(self.labels, eds, predictor, clipper, weights, "folder {:>1} average score {:>5.3f}", "weighted eval mse loss {:>6.3}")
            predictor.train()
            print("====")

        def on_epoch_end(self, arguments, state, control, **kwargs):
            tds.shuffle()
            if not self.every: return
            if state.epoch - self.last < self.every: return
            self.last = state.epoch
            self.do_eval(state)

        def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            self.do_eval(state)

    with Timer('train model'):
        CustomTrainer(  model = predictor, 
                        train_dataset = tds,
                        eval_dataset = eds,
                        weights = weights if args['weight_category_loss'] else None,
                        args = TrainingArguments( remove_unused_columns=False, push_to_hub=False, output_dir=args['save_model'], **training_args ), 
                        callbacks = [EvaluationCallback(every=args['eval_every_n_epochs'], labels=data.labels)], 
                     ).train()

        predictor.remove_relu()
        save_file(predictor.state_dict(),os.path.join(args['save_model'],"model.safetensors"))

def print_args():
    print("args:")
    for a in args:
        print("{:>30} : {:<40}".format(a, str(args[a])))
    print("trainging_args")
    for a in training_args:
        print("{:>30} : {:<40}".format(a, str(training_args[a])))

if __name__=='__main__':
    print_args()
    train_predictor()