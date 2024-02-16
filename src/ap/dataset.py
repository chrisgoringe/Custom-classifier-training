from pandas import DataFrame
from .aesthetic_predictor import AestheticPredictor
import random, statistics
import torch
import scipy.stats
from .image_scores import get_ab
        
class QuickDataset(torch.utils.data.Dataset):
    def __init__(self, df:DataFrame, split:str=None):
        self._df = df
        self.map = [i for i in range(len(df)) if (not split or df['split'].array[i]==split)]
        self.shuffle()

    def __getitem__(self, i):
        x = self._df['features'].array[self.map[i]]
        y = torch.tensor(self._df['score'].array[self.map[i]], dtype=torch.float)
        return {"x":x, "y":y}

    def __len__(self):
        return len(self.map)
    
    def shuffle(self):
        random.shuffle(self.map)

    def update_prediction(self, predictor:AestheticPredictor):
        p = predictor.evaluate_files(self._df['image'], output_value=None)
        self._df['predicted_score'] = list(pi[0] for pi in p)
        self._df['sigma'] = list(abs(pi[1]) if len(pi)>1 else 1.0 for pi in p)

    def get_metadata(self):
        scores = self.column('predicted_score', float)
        return {
            "n_images"              : str(len(self.map)),
            "mean_predicted_score"  : str(statistics.mean(scores)),
            "stdev_predicted_score" : str(statistics.stdev(scores)),
        }

    def column(self, col:str, convert:callable=lambda a:a) -> list:
        return [convert(self._df[col].array[x]) for x in self.map]
    
    #def column_where(self, col:str, match_col:str, match:str, convert:callable=lambda a:a) -> list:
    #    return [convert(self._df[col].array[x]) for x in self.map if self._df[match_col].array[x]==match] 
    
    #def columns(self, *args) -> list:
    #    cols = []
    #    for x in self.map:
    #        cols.append(tuple(self._df[col].array[x] for col in args))
    #    return cols
    
    #def get_ab_score(self):
    #    print(f"get_ab_score deprecated - use get_ab")
    #    return self.get_ab()
    
    def get_ab(self):
        return get_ab(self.column('score'), self.column('predicted_score'))
        
    def get_mse(self):
        loss_fn = torch.nn.MSELoss()
        rmse = loss_fn(torch.tensor(self.column('score')), torch.tensor(self.column('predicted_score')))
        return float(rmse)
    
    def get_nll(self):
        loss_fn = torch.nn.GaussianNLLLoss()
        nll = loss_fn(torch.tensor(self.column('predicted_score')), torch.tensor(self.column('score')), torch.square(torch.tensor(self.column('sigma'))))
        return float(nll)
    
    def get_spearman(self):
        return scipy.stats.spearmanr(self.column('predicted_score'), self.column('score')).statistic