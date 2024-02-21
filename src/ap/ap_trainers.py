import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, EvalPrediction
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.dataset import QuickDataset
import random
from src.time_context import Timer

class EvaluationCallback(TrainerCallback):
    def __init__(self, datasets_to_shuffle:list[QuickDataset], datasets_to_score:list[tuple[str, QuickDataset]]):
        self.datasets_to_shuffle = datasets_to_shuffle
        self.datasets_to_score = datasets_to_score

    def _do_eval(self, state: TrainerState, predictor:AestheticPredictor):
        if not len(self.datasets_to_score): return
        was_train = predictor.training
        predictor.eval()
        Timer.message(" Epoch {:>6.1f}".format(state.epoch))        
        for label, dataset in self.datasets_to_score:
            with torch.no_grad():
                dataset.update_prediction(predictor)
            Timer.message("{:8}: mse {:>6.3f}".format(label,dataset.get_mse()))
        if was_train: predictor.train()

    def on_epoch_end(self, arguments, state: TrainerState, control, **kwargs):
        for dataset in self.datasets_to_shuffle: dataset.shuffle()
        if 'model' in kwargs: self._do_eval(state, kwargs['model'])

    def compute_metrics(self, predictions:EvalPrediction):
        pass

class CustomTrainer(Trainer):
    def __init__(self, model, train_dataset, eval_dataset, **kwargs):
        super().__init__(model, **kwargs)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args:TrainingArguments = kwargs['args']

    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size = self.training_args.per_device_train_batch_size)
    
    def get_eval_dataloader(self, eds):
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size = self.training_args.per_device_eval_batch_size)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        targets = inputs['y']
        outputs = model(inputs['x'])
        weight = inputs.get('weight',None)
        loss = self._compute_loss(targets, outputs, weight)
        return (loss, outputs) if return_outputs else loss

    @classmethod
    def trainer(cls, loss, **kwargs):
        if loss=='mse':
            return MSELossTrainer(**kwargs)
        if loss=='ab':
            return RankingLossTrainer(**kwargs)
        if loss=='nll':
            return NegativeLogLikelihoodLossTrainer(**kwargs)
        if loss=='wmes':
            return WMSELossTrainer(**kwargs)
        raise NotImplementedError(loss)

class MSELossTrainer(CustomTrainer):
    loss_fn = torch.nn.MSELoss()
    def _compute_loss(self, scores, outputs, weight):
        return self.loss_fn(scores, torch.squeeze( outputs ))
    
class WMSELossTrainer(CustomTrainer):
    def _compute_loss(self, scores, outputs, weight):
        return torch.sum( torch.multiply(torch.square(scores-torch.squeeze( outputs )), weight ) )/len(scores)
    
class NegativeLogLikelihoodLossTrainer(CustomTrainer):
    loss_fn = torch.nn.GaussianNLLLoss()
    def _compute_loss(self, scores, outputs, weight):
        return self.loss_fn(outputs[:,0],scores,torch.square(outputs[:,1]))

    
class UnequalSampler:
    def __init__(self, batch_size:int, dataset:torch.utils.data.Dataset):
        self.batch_size = batch_size
        self.dataset = dataset
        self.len = len(self.dataset)//self.batch_size
        if self.len==0: self.len=1
        self.left = self.len

    def __len__(self):
        return self.len
    
    def __iter__(self):
        half_batch = self.batch_size//2
        result = [0]*self.batch_size    
        while self.left:
            self.left -= 1
            for i in range(half_batch):
                x = random.randrange(len(self.dataset))
                y = random.randrange(len(self.dataset))
                item1 = self.dataset.__getitem__(x)
                item2 = self.dataset.__getitem__(y)
                while (item1['y']==item2['y']): 
                    y = random.randrange(len(self.dataset))
                    item2 = self.dataset.__getitem__(y)
                result[i] = x
                result[i+half_batch] = y
            yield result
        self.left = self.len

class RankingLossTrainer(CustomTrainer):
    loss_fn = torch.nn.MarginRankingLoss(1.0)

    def _compute_loss(self, targets, outputs):
        half_batch = len(targets)//2
        flat_outputs = torch.squeeze(outputs)
        input1 = flat_outputs[:half_batch]
        input2 = flat_outputs[half_batch:]
        diff_target = torch.sign( targets[:half_batch] - targets[half_batch:] )
        return self.loss_fn(input1, input2, diff_target)
    
    def get_train_dataloader(self):
        self.sampler = UnequalSampler(self.training_args.per_device_train_batch_size, self.train_dataset)
        dl = torch.utils.data.DataLoader(self.train_dataset, batch_sampler = self.sampler)
        return dl
    
    def evaluate(self, eval_dataset:QuickDataset=None, ignore_keys=None, metric_key_prefix=None):
        ev:QuickDataset = eval_dataset or self.eval_dataset
        return { "eval_ab" : ev.get_ab(), "eval_rmse" : ev.get_rmse(), "eval_nll" : ev.get_nll }

