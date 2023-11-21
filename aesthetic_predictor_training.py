import torch
import torch.nn as nn
from safetensors.torch import save_file
import os, random, statistics
from src.time_context import Timer
from src.data_holder import DataHolder
from transformers import TrainerControl, TrainerState, TrainingArguments, TrainerCallback
from pandas import DataFrame
try:
    from renumics import spotlight
    have_spotlight = True
except:
    have_spotlight = False

from arguments import training_args, args
from src.clip import CLIP
from src.aesthetic_predictor import AestheticPredictor
from src.ap_trainers import CustomTrainer
    
class QuickDataset(torch.utils.data.Dataset):
    def __init__(self, df:DataFrame):
        self.df = df
        self.map = list(range(len(self)))
        self.shuffle()

    def __getitem__(self, i):
        x = self.df['features'].array[self.map[i]]
        y = torch.tensor(self.df['score'].array[self.map[i]], dtype=torch.float)
        return {"x":x, "y":y}

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        random.shuffle(self.map)

    def update_prediction(self, predictor:AestheticPredictor):
        self.df['predicted_score'] = (predictor.evaluate_files(self.df['image'], eval_mode=True))

class EvaluationCallback(TrainerCallback):
        def __init__(self, every, datasets, labels, shuffles):
            self.last = 0
            self.every = every or 0
            self.datasets = list(zip(datasets,labels,shuffles))

        def do_eval(self, state, predictor:AestheticPredictor):
            for dataset, label, _ in self.datasets:
                with torch.no_grad():
                    dataset.update_prediction(predictor)
                print(f"\n\n====\nEpoch {str(state.epoch)} ({label}): ")
                report(dataset, "folder {:>1} average score {:>5.3f} +/ {:>5.3f}")
                print("====")

        def on_epoch_end(self, arguments, state: TrainerState, control, **kwargs):
            for dataset, _, shuffle in self.datasets:
                if shuffle: dataset.shuffle()
            if state.epoch - self.last < self.every: return
            self.last = state.epoch
            self.do_eval(state, kwargs['model'])

        def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.epoch != self.last: self.do_eval(state, kwargs['model'])

def ab_report(ds:QuickDataset, prnt="AB: {:>5}/{:<5} correct ({:>5.2f}%)"):
    right = 0
    wrong = 0
    true_predicted = list(zip(ds.df['score'], ds.df['predicted_score']))
    for i, a in enumerate(true_predicted):
        for b in true_predicted[i+1:]:
            if a[0]==b[0]: continue
            if (a[0]<b[0] and a[1]<b[1]) or (a[0]>b[0] and a[1]>b[1]): right += 1
            else: wrong += 1
    if prnt: print(prnt.format(right,right+wrong,100*right/(right+wrong)))
    return right/(right+wrong)
    

def report(eds:QuickDataset, prnt:str, prntrmse:str="rms error {:>6.3}"):
    loss_fn = torch.nn.MSELoss()
    if prnt:
        for x in sorted(eds.df['label_str'].unique()):
            df:DataFrame = eds.df[eds.df['label_str']==x]
            std = statistics.stdev(df['predicted_score'].to_numpy()) if len(df)>1 else 0
            print(prnt.format(x,statistics.mean(df['predicted_score'].to_numpy()),std))
    rmse = loss_fn(torch.tensor(eds.df['score'].to_numpy()), torch.tensor(eds.df['predicted_score'].to_numpy()))
    if prntrmse: print(prntrmse.format(rmse))
    ab_report(eds)
    return rmse

def train_predictor():
    pretrained = os.path.join(args['load_model'],"model.safetensors") if ('load_model' in args and args['load_model']) else args['base_model']
    top_level_images = args['top_level_image_directory']

    with Timer('load models'):
        clipper = CLIP()
        predictor = AestheticPredictor(pretrained=pretrained, relu=args['aesthetic_model_relu'], clipper=clipper)

    with Timer('Prepare images'):
        data = DataHolder(top_level=top_level_images)
        df = data.get_dataframe()
        ds = QuickDataset(df)
        with Timer('CLIP'):
            df['features'] = [clipper.prepare_from_file(f, device="cpu") for f in df['image']]
            clipper.save_cache()
        df['score'] = [float(l) for l in df['label_str']]
        tds = QuickDataset(df[df["split"] == 'train'])
        eds = QuickDataset(df[df["split"] == 'test'])

    with torch.no_grad():
        ds.update_prediction(predictor)
    report(ds, "Start (train+eval): folder {:>1} average score {:>5.3f} +/ {:>5.3f}")

    with Timer('train model'):
        train_args = TrainingArguments( remove_unused_columns=False, push_to_hub=False, output_dir=args['save_model'], **training_args )
        callback = EvaluationCallback(every=args['eval_every_n_epochs'], datasets=[ds,eds], labels=["all","test"], shuffles=[False,True])

        CustomTrainer.trainer(  loss = args['loss_model'], model = predictor, 
                                train_dataset = tds, eval_dataset = eds, 
                                args = train_args, callbacks = [callback,], 
                            ).train()

        save_file(predictor.state_dict(),os.path.join(args['save_model'],"model.safetensors"))

    if have_spotlight and 'spotlight' in args['mode']: 
        ds.update_prediction(predictor)
        try:
            spotlight.show(ds.df)
        except:
            pass

    if args['mode']=='meta':
        eds.update_prediction(predictor)
        tds.update_prediction(predictor)
        return report(eds,None,None), report(tds,None,None), ab_report(eds,None)
        

def print_args():
    print("args:")
    for a in args:
        print("{:>30} : {:<40}".format(a, str(args[a])))
    print("trainging_args")
    for a in training_args:
        print("{:>30} : {:<40}".format(a, str(training_args[a])))

if __name__=='__main__':
    print_args()

    if args['mode']=='meta':
        with open("meta.txt",'w') as f:
            print("epochs,lr,batch,train_loss,eval_loss,ab", file=f)
            for lr in args['meta_lr'] if args['meta_lr'] else [training_args['learning_rate'],]:
                for epochs in args['meta_epochs'] if args['meta_epochs'] else [training_args['num_train_epochs'],]:
                    for batch in args['meta_batch'] if args['meta_batch'] else [training_args['per_device_train_batch_size'],]:
                        training_args['num_train_epochs'] = epochs
                        training_args['learning_rate'] = lr
                        training_args['per_device_train_batch_size'] = batch
                        eval_loss, train_loss, ab = train_predictor()
                        print(f"{epochs},{lr},{batch},{train_loss},{eval_loss},{ab}",file=f, flush=True)
    else:
        train_predictor()