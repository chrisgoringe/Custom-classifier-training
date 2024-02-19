import pandas as pd
from .aesthetic_predictor import AestheticPredictor
from .feature_extractor import FeatureExtractor
from .image_scores import ImageScores
import random, statistics
import torch
from torch.utils.data import Dataset
import scipy.stats
import os, json
from typing import Self, Callable

pd.options.mode.copy_on_write = True

class QuickDataset(Dataset, ImageScores):
    def __init__(self, **kwargs):
        Dataset.__init__(self)
        ImageScores.__init__(self, **kwargs)
        self.map = [i for i in range(len(self._df))]
        self.shuffle()
        self.children = []

    def subset(self, test:Callable, item:str='relative_path') -> Self:
        qd = QuickDataset(top_level_directory=self.tld, df=self._df.loc[test(self._df[item])])
        self.children.append(qd)
        return qd

    def allocate_split(self, fraction_for_eval:float=0.25, eval_pick_seed:int=42, replace=False):
        if self.has_item('split') and not replace:
            print("Not replacing existing splits")
            return
        random.seed(eval_pick_seed)
        self.add_item('split', lambda a: "eval" if random.random() < fraction_for_eval else "train")

    def extract_features(self, feature_extractor:FeatureExtractor):
        feature_extractor.precache(self.image_files(fullpath=True))
        self.add_item('features', feature_extractor.get_features_from_file, fullpath=True, cast=lambda a:a.cpu())

    def update_prediction(self, predictor:AestheticPredictor):
        data = torch.stack(self.item('features')).to(predictor.device)
        p = predictor(data).cpu()
        #p = predictor.evaluate_files(self._df['image'], output_value=None)
        self.add_item('model_score', list(float(pi[0]) for pi in p))
        self.add_item('sigma', list(float(abs(pi[1])) if len(pi)>1 else 1.0 for pi in p))
        for child in self.children:
            child.add_item(label='model_score', values=lambda a:self.element(label='model_score', file=a))
            child.add_item(label='sigma', values=lambda a:self.element(label='sigma', file=a))

    def __getitem__(self, i):
        x = self._df['features'].array[self.map[i]]
        y = torch.tensor(self._df['score'].array[self.map[i]], dtype=torch.float)
        return {"x":x, "y":y}

    def __len__(self):
        return len(self.map)
    
    def shuffle(self):
        random.shuffle(self.map)

    def get_metadata(self):
        scores = self.item('model_score')
        return {
            "n_images"              : str(len(self.map)),
            "mean_predicted_score"  : str(statistics.mean(scores)),
            "stdev_predicted_score" : str(statistics.stdev(scores)),
        }
    
    def get_ab(self):
        raise NotImplementedError()
        
    def get_mse(self):
        loss_fn = torch.nn.MSELoss()
        rmse = loss_fn(torch.tensor(self.item('score')), torch.tensor(self.item('model_score')))
        return float(rmse)
    
    def get_nll(self):
        loss_fn = torch.nn.GaussianNLLLoss()
        nll = loss_fn(torch.tensor(self.item('model_score')), torch.tensor(self.item('score')), torch.square(torch.tensor(self.item('sigma'))))
        return float(nll)
    
    def get_spearman(self):
        return scipy.stats.spearmanr(self.item('model_score'), self.item('score')).statistic

     
