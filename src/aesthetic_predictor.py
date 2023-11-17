import torch
import torch.nn as nn
import clip, datasets
from PIL import Image
from safetensors.torch import save_file, load_file
import os
from time_context import Timer
from data_holder import DataHolder
from transformers import Trainer, TrainingArguments, TrainerCallback

class CLIP:
    def __init__(self, pretrained="ViT-L/14", device="cuda"):
        self.model, self.preprocess = clip.load(pretrained, device=device)
        self.device = device

    def get_image_features_tensor(self, image:Image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.to(torch.float)

    def get_image_features(self, image:Image):
        image_features = self.get_image_features_tensor(image)
        image_features = image_features.cpu().detach().numpy()
        return image_features
    
    def prepare_image(self, image:Image, device="cuda"):
        image_features = self.get_image_features(image)
        return torch.from_numpy(image_features).to(device).float()

class AestheticPredictor(nn.Module):
    def __init__(self, input_size=768, pretrained="models/sac+logos+ava1-l14-linearMSE.safetensors", device="cuda"):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
        self.load_state_dict(load_file(pretrained))
        self.layers.to(device)
        self.device = device

    def forward(self, x, **kwargs):
        return self.layers(x)
    
def evaluate_files(files, predictor, clipper):
    with Timer('evaluate'):
        def score_file(f):
            image = Image.open(f)
            return predictor(clipper.prepare_image(image)).item()
        scores = [(score_file(f), f) for f in files]
        scores.sort()
        return scores
        
def evaluate_directory(image_directory, predictor=None, clipper=None):
    if predictor is None:
        with Timer('load models'):
            predictor = AestheticPredictor()
            clipper = CLIP()

    return evaluate_files([os.path.join(image_directory,f) for f in os.listdir(image_directory)], predictor, clipper)

    
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        scores = inputs['y'].to(torch.float)
        outputs = torch.flatten( model(inputs['x'].to(torch.float)) )
        # compute custom loss (suppose one has 3 labels with different weights)
        if self.applyweight:
            delta = outputs - scores
            dsquared = torch.square(delta) * self.weight(scores)
            loss = torch.sum(dsquared)
        else:
            loss_fct = nn.MSELoss()
            loss = loss_fct(scores.reshape(outputs.shape), outputs)
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=4)
    
    def get_eval_dataloader(self, eds):
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size=4)
    
    def weight(self, scores):
        return torch.where(scores==3.0, 3.0, 1.0)
    
class QuickDataset(torch.utils.data.Dataset):
    def __init__(self, dh:DataHolder, clipper:CLIP, split):
        df = dh.get_dataframe()
        df = df[df["split"] == split]
        self.features = [clipper.get_image_features_tensor(Image.open(f)) for f in df['image']]
        self.scores = torch.tensor([float(l) for l in df['label_str']])
        self.allfiles = [f for f in df['image']]

    def __getitem__(self, i):
        x = self.features[i]
        y = self.scores[i]
        return {"x":x, "y":y}

    def __len__(self):
        return len(self.features)
    
    def files(self, filter):
        return [x for x in self.allfiles if os.path.basename(os.path.split(x)[0])==filter]
    
def prepare(df):
    ds = datasets.Dataset.from_pandas(df)
    prepared_ds = ds.cast_column("image", datasets.Image())
    return prepared_ds



def train_predictor(training_args, top_level_images="training/aesthetic", pretrained="models/sac+logos+ava1-l14-linearMSE.safetensors"):

    with Timer('load models'):
        predictor = AestheticPredictor(pretrained=pretrained)
        clipper = CLIP()
        data = DataHolder(top_level=top_level_images, model_folder=None, fraction_for_test=0.2, test_pick_seed=42)
        tds = QuickDataset(data, clipper, "train")
        eds = QuickDataset(data, clipper, "test")

    score3before = evaluate_files(eds.files('3'), predictor, clipper)
    score7before = evaluate_files(eds.files('7'), predictor, clipper)
    for x in [score3before, score7before, ]:
        print(sum(y[0] for y in x)/len(x))

    class EvaluationCallback(TrainerCallback):
        def __init__(self, every):
            self.last = 0
            self.every = every

        def on_epoch_end(self, arguments, state, control, **kwargs):
            if state.epoch - self.last < self.every: return
            self.last = state.epoch
            score3 = evaluate_files(eds.files('3'), predictor, clipper)
            mean3 = sum(y[0] for y in score3)/len(score3)
            score7 = evaluate_files(eds.files('7'), predictor, clipper)
            mean7 = sum(y[0] for y in score7)/len(score7)
            print("\n\n======\nEpoch {:>5.1f}: folder 3 average score {:>5.3f}, folder 7 average score {:>5.3f} \n======\n".format(state.epoch,mean3,mean7))

    with Timer('train model'):
        
        t = CustomTrainer(predictor, 
                          args=TrainingArguments( remove_unused_columns=False, push_to_hub=False, **training_args ), 
                          callbacks=[EvaluationCallback(every=2)],
                          )
        t.train_dataset = tds
        t.eval_dataset = eds
        t.applyweight = False
        t.train()



if __name__=='__main__':
    train_args = {"num_train_epochs":30,"learning_rate":5e-4, "output_dir":"temp_model_save"}
    train_predictor(train_args)