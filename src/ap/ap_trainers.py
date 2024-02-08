import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, EvalPrediction
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.dataset import QuickDataset
import random, re
from src.time_context import Timer

class EvaluationCallback(TrainerCallback):
    def __init__(self, datasets:list[QuickDataset]):
        self.datasets = datasets

    def do_eval(self, state, predictor:AestheticPredictor):
        for dataset, label, _ in self.datasets:
            with torch.no_grad():
                was_train = predictor.training
                predictor.eval()
                dataset.update_prediction(predictor)
                if was_train: predictor.train()
            Timer.message("==== Epoch {:>3} ({:8}): rmse {:>6.3f} ab {:>5.2f}%".format(state.epoch,label,dataset.get_rmse(),100*dataset.get_ab_score()))

    def on_epoch_end(self, arguments, state: TrainerState, control, **kwargs):
        for dataset in self.datasets: dataset.shuffle()

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
        loss = self._compute_loss(targets, outputs)
        return (loss, outputs) if return_outputs else loss

    @classmethod
    def trainer(cls, loss, **kwargs):
        if loss=='mse':
            return MSELossTrainer(**kwargs)
        if loss=='ranking':
            return RankingLossTrainer(**kwargs)
        if loss=='nll':
            return NegativeLogLikelihoodLossTrainer(**kwargs)
        raise NotImplementedError(loss)

class MSELossTrainer(CustomTrainer):
    loss_fn = torch.nn.MSELoss()
    def _compute_loss(self, scores, outputs):
        return self.loss_fn(scores, torch.squeeze( outputs ))
    
class NegativeLogLikelihoodLossTrainer(CustomTrainer):
    def __init__(self, model, special_lr_parameters=dict[re.Pattern, float], **kwargs):
        super().__init__(model, **kwargs)
        self.special_lr_parameters = {x:special_lr_parameters[x] for x in special_lr_parameters}
        self.special_lr_parameters[re.compile('')] = 1.0

    loss_fn = torch.nn.GaussianNLLLoss()
    def _compute_loss(self, scores, outputs):
        return self.loss_fn(outputs[:,0],scores,torch.square(outputs[:,1]))
    
    def first_match(self, n):
        for i,r in enumerate(self.special_lr_parameters):
            if r.search(n): return i

    def create_optimizer(self):
        all_params = list((p,self.first_match(n)) for n,p in self.model.named_parameters() if p.requires_grad)

        optimizer_grouped_parameters = [
            {
                "params": [ p for p, m in all_params if m==i ],
                "lr": self.args.learning_rate * self.special_lr_parameters[r],
            } for i, r in enumerate(self.special_lr_parameters)
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer
    
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
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix=None):
        ev = eval_dataset or self.eval_dataset
        return { "eval_ranking" : ev.get_ab_score(), "eval_loss" : ev.get_rmse() }

