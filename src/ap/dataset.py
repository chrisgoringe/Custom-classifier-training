import pandas as pd
from .aesthetic_predictor import AestheticPredictor
from .feature_extractor import FeatureExtractor
from .image_scores import ImageScores
import random, statistics
import torch
import scipy.stats
from .image_scores import get_ab
import os, json
        
class QuickDataset(torch.utils.data.Dataset):

    @classmethod
    def from_scorefile(cls, image_folder:str, fraction_for_eval:float=0.25, eval_pick_seed:int=42, scorefile:str=None):
        df = pd.DataFrame(columns=["image","score","split"])
        if eval_pick_seed: random.seed(eval_pick_seed)
        split = lambda : "eval" if random.random() < fraction_for_eval else "train"
        with open(os.path.join(image_folder,scorefile),'r') as f:
            image_scores = json.load(f)
            if "ImageRecords" in image_scores:
                for f in image_scores["ImageRecords"]:
                    df.loc[len(df)] = [os.path.join(image_folder,f), float(image_scores["ImageRecords"][f]['score']), split()]
            else:
                image_scores.pop('#meta#',None)
                for f in image_scores:
                    df.loc[len(df)] = [os.path.join(image_folder,f), float(image_scores[f][0] if isinstance(image_scores[f],list) else image_scores[f]), split()]
        return QuickDataset(df, image_folder=image_folder)
    
    @classmethod
    def subset(cls, qd, split):
        return QuickDataset(qd._df, split=split, image_folder=qd.image_folder)

    def __init__(self, df:pd.DataFrame, split:str=None, image_folder=None):
        self._df = df
        self.map = [i for i in range(len(df)) if (not split or df['split'].array[i]==split)]
        self.image_folder = image_folder
        self.shuffle()

    def __getitem__(self, i):
        x = self._df['features'].array[self.map[i]]
        y = torch.tensor(self._df['score'].array[self.map[i]], dtype=torch.float)
        return {"x":x, "y":y}

    def __len__(self):
        return len(self.map)
    
    def shuffle(self):
        random.shuffle(self.map)

    @property
    def images(self):
        return self._df['image']

    def update_prediction(self, predictor:AestheticPredictor):
        data = torch.stack(list(self._df['features'])).to(predictor.device)
        p = predictor(data).cpu()
        #p = predictor.evaluate_files(self._df['image'], output_value=None)
        self._df['predicted_score'] = list(pi[0] for pi in p)
        self._df['sigma'] = list(abs(pi[1]) if len(pi)>1 else 1.0 for pi in p)

    def extract_features(self, feature_extractor:FeatureExtractor):
        feature_extractor.precache(self.images)
        self._df['features'] = [feature_extractor.get_features_from_file(f, device="cpu") for f in self.images]

    def get_metadata(self):
        scores = self.column('predicted_score', float)
        return {
            "n_images"              : str(len(self.map)),
            "mean_predicted_score"  : str(statistics.mean(scores)),
            "stdev_predicted_score" : str(statistics.stdev(scores)),
        }
    
    def dictionary(self, col): 
        return { os.path.relpath(f, self.image_folder) : s for f, s in zip(self._df['image'], self._df[col]) }
    
    def get_image_scores(self) -> ImageScores:
        return ImageScores( image_scores=self.dictionary('score'), top_level_directory=self.image_folder, normalisation=False )
    
    def save_split(self, splitfile):
        if not splitfile: return
        with open(os.path.join(self.image_folder, splitfile), 'w') as fhdl:
            print(json.dumps(self.dictionary('split'), indent=2), file=fhdl)

    def column(self, col:str, convert:callable=lambda a:a) -> list:
        return [convert(self._df[col].array[x]) for x in self.map]
     
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