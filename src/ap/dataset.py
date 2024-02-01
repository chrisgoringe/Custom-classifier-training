from pandas import DataFrame
from .aesthetic_predictor import AestheticPredictor
import random, statistics
import torch
        
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
        self._df['predicted_score'] = (predictor.evaluate_files(self._df['image'], eval_mode=True))

    def get_metadata(self):
        scores = self.column('predicted_score')
        return {
            "n_images"              : str(len(self.map)),
            "mean_predicted_score"  : str(statistics.mean(scores)),
            "stdev_predicted_score" : str(statistics.stdev(scores)),
        }

    def column(self, col:str) -> list:
        return [self._df[col].array[x] for x in self.map]
    
    def column_where(self, col:str, match_col:str, match:str) -> list:
        return [self._df[col].array[x] for x in self.map if self._df[match_col].array[x]==match] 
    
    def columns(self, *args) -> list:
        cols = []
        for x in self.map:
            cols.append(tuple(self._df[col].array[x] for col in args))
        return cols
    
    def get_ab_score(self):
        right = 0
        wrong = 0
        true_predicted = self.columns('score','predicted_score')
        for i, a in enumerate(true_predicted):
            for b in true_predicted[i+1:]:
                if a[0]==b[0] or a[1]==b[1]: continue
                if (a[0]<b[0] and a[1]<b[1]) or (a[0]>b[0] and a[1]>b[1]): right += 1
                else: wrong += 1
        return right/(right+wrong) if (right+wrong) else 0
        
    def get_rmse(self):
        loss_fn = torch.nn.MSELoss()
        rmse = loss_fn(torch.tensor(self.column('score')), torch.tensor(self.column('predicted_score')))
        return float(rmse)