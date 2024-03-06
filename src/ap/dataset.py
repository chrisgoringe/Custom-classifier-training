import pandas as pd
from .aesthetic_predictor import AestheticPredictor
from .feature_extractor import FeatureExtractor
from .image_scores import ImageScores
import random, statistics
import torch
from torch.utils.data import Dataset
import scipy.stats

pd.options.mode.copy_on_write = True

class QuickDataset(Dataset, ImageScores):
    exclude_from_save = ImageScores.exclude_from_save + ('features',)
    def __init__(self, dtype = torch.float, **kwargs):
        Dataset.__init__(self)
        ImageScores.__init__(self, **kwargs)
        self.map = [i for i in range(len(self._df))]
        self.shuffle()
        self.has_weights = 'weight' in self._df.columns
        self.dtype = dtype

    def subset(self, test, item:str='relative_path'):
        qd = QuickDataset(top_level_directory=self.tld, df=self._df.loc[test(self._df[item])])
        return qd

    def allocate_split(self, fraction_for_eval:float=0.25, eval_pick_seed:int=42, replace=False):
        if self.has_item('split') and not replace:
            print("Not replacing existing splits")
            return
        random.seed(eval_pick_seed)
        self.add_item('split', lambda a: "eval" if random.random() < fraction_for_eval else "train")

    def extract_features(self, feature_extractor:FeatureExtractor, just_precache=False, clear_cache_after=False):
        feature_extractor.precache(self.image_files(fullpath=True))
        if just_precache: return
        self.add_item('features', feature_extractor.get_features_from_file, fullpath=True, cast=lambda a:a.cpu())
        if clear_cache_after: feature_extractor.clear_cache()

    def update_prediction(self, predictor:AestheticPredictor):
        data = torch.stack(self.item('features')).to(predictor.device).to(self.dtype)
        predictions = predictor(data).cpu()
        for i in range(predictor.output_channels):
            label = 'model_score' if i==0 else f"model_score_{i}"
            p = predictions[:,i].numpy()
            self.add_item(label, p)

    def __getitem__(self, i):
        x = self._df['features'].array[self.map[i]].to(self.dtype)
        y = torch.tensor(self._df['score'].array[self.map[i]], dtype=self.dtype)
        if self.has_weights:
            w = torch.tensor(self._df['weight'].array[self.map[i]], dtype=self.dtype)
            return {"x":x, "y":y, "weight":w}
        else: return {"x":x, "y":y}

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
        
    def get_mse(self, **kwargs):
        loss_fn = torch.nn.MSELoss()
        mse = loss_fn(torch.tensor(self.item('score')), torch.tensor(self.item('model_score')))
        return float(mse)
    
    def get_wmse(self, **kwargs):
        sc = self.item('score')
        ms = self.item('model_score')
        we = self.item('weight')
        wmse = sum( (s-m)*(s-m)*w for s,m,w in zip(sc,ms,we) )/len(self)
        return float(wmse)
    
    def get_nll(self, **kwargs):
        loss_fn = torch.nn.GaussianNLLLoss()
        nll = loss_fn(torch.tensor(self.item('model_score')), torch.tensor(self.item('score')), torch.square(torch.tensor(self.item('sigma'))))
        return float(nll)
    
    def get_spearman(self, **kwargs):
        return scipy.stats.spearmanr(self.item('model_score'), self.item('score')).statistic
    
    def get_pearson(self, **kwargs):
        return scipy.stats.pearsonr(self.item('model_score'), self.item('score')).statistic
    
    def get_accuracy(self, divider, **kwargs):
        if divider is None: divider = statistics.median(self.item('score'))
        x = sum(( (a>=divider and b>=divider) or (a<divider and b<divider) ) for a,b in zip(self.item('model_score'), self.item('score')))
        return (x/(len(self)))

     
