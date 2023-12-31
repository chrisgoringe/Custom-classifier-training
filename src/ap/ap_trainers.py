import torch
from transformers import Trainer, TrainingArguments
import random

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
        raise NotImplementedError(loss)

class MSELossTrainer(CustomTrainer):
    loss_fn = torch.nn.MSELoss()
    def _compute_loss(self, scores, outputs):
        return self.loss_fn(scores, torch.squeeze( outputs ))
    
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

    def set_epoch(self,x):
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
        bs = UnequalSampler(self.training_args.per_device_train_batch_size, self.train_dataset)
        dl = torch.utils.data.DataLoader(self.train_dataset, batch_sampler = bs)
        dl.set_epoch = bs.set_epoch
        return dl

