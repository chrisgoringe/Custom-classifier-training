import os, json, statistics
import pandas as pd
#from typing import Self, Callable
from collections.abc import Iterable
from numpy import ndarray

class ImageScores:
    exclude_from_save = ('path', 'relative_path',)
    def __init__(self, top_level_directory:str, files:list=None, scores:list=None, df:pd.DataFrame=None):
        self.tld = top_level_directory
        self._df = pd.DataFrame(columns=['relative_path', 'path', 'score']) if df is None else df
        if files:
            self._df['relative_path'] = list(os.path.normpath(f) for f in files)
            if scores: 
                self._df['score'] = scores
            else:
                self._df['score'] = [0]*len(files)
        self._df['path'] = list(os.path.normpath(os.path.join(self.tld,f)) for f in self._df['relative_path'])
        self._df.set_index('relative_path',drop=False,inplace=True)
        pass
        
    def subset(self, test, item:str='relative_path'):
        return ImageScores(top_level_directory=self.tld, df=self._df.loc[test(self._df[item])])

    @classmethod
    def from_scorefile(cls, top_level_directory:str, scorefilename):
        if os.path.splitext(scorefilename)[1]==".json":
            with open(os.path.join(top_level_directory,scorefilename),'r') as f:
                image_scores_dict = json.load(f)
                if "ImageRecords" in image_scores_dict:
                    files = list(k for k in image_scores_dict["ImageRecords"])
                    scores = list(float(image_scores_dict["ImageRecords"][k]['score']) for k in files)
                    imsc = cls(top_level_directory=top_level_directory, files=files, scores=scores)
                    for item in image_scores_dict.get("Additionals",['comparisons',]):
                        imsc.add_item(item, list(image_scores_dict["ImageRecords"][k].get(item,0) for k in files)) 
        elif os.path.splitext(scorefilename)[1]==".csv":
            imsc = cls(top_level_directory=top_level_directory, df=pd.read_csv(os.path.join(top_level_directory,scorefilename)))
        return imsc
    
    @classmethod
    def from_evaluator(cls, evaluator, images:list[str], top_level_directory, fullpath=True):
        scores = list(float(evaluator(os.path.join(top_level_directory,k) if fullpath else k)) for k in images)
        return cls(top_level_directory=top_level_directory, files=images, scores=scores) 

    @classmethod
    def from_directory(cls, top_level_directory, evaluator=lambda a:0):
        images = []
        valid_image = lambda f : os.path.splitext(f)[1] in ['.png','.jpg','.jpeg']
        def recursively_add_images(d=""):
            for thing in os.listdir(os.path.join(top_level_directory,d)):
                if thing.startswith("."): continue
                thingpath = os.path.join(top_level_directory,d,thing)
                if valid_image(thingpath): images.append(os.path.join(d,thing))
                if os.path.isdir(thingpath): recursively_add_images(os.path.join(d,thing))
        recursively_add_images()
        return cls.from_evaluator(evaluator, images, top_level_directory)
    
    @classmethod
    def from_baid(cls, top_level_directory, include=('eval', 'train', 'test')):
        images = []
        scores = []
        splits = []
        for split in include:
            with open(os.path.join(top_level_directory,f"{split}_set.csv"), 'r') as f:
                for line in f.readlines():
                    image, score = line.split(",")
                    if image!='image':
                        images.append(f"images/{image}")
                        scores.append(float(score))
                        splits.append(split)
        imsc = cls(top_level_directory=top_level_directory, files=images, scores=scores)
        imsc.add_item('split',splits)
        return imsc
    
    def add_item(self, label, values, fullpath=False, cast=lambda a:a):
        if isinstance(values, dict):
            self._df[label] = list(cast(values[f]) for f in self.image_files(fullpath=fullpath))
        elif callable(values):
            self._df[label] = list(cast(values(f)) for f in self.image_files(fullpath=fullpath))
        elif isinstance(values, ImageScores):
            self._df[label] = list( cast(values.score(f)) for f in self.image_files() )
        elif isinstance(values,ndarray):
            self._df[label] = values
        elif isinstance(values, Iterable):
            self._df[label] = list(cast(v) for v in values)
        else:
            raise NotImplementedError()
        
    def normalise(self, label, mean, stdev=None):
        if stdev:
            self._df[label] = self._df[label] - statistics.mean(self._df[label])
            self._df[label] = self._df[label] * stdev/statistics.stdev(self._df[label])
        self._df[label] = self._df[label] + mean - statistics.mean(self._df[label])

    def save_as_scorefile(self, scorefilepath):
        self._df.to_csv(open(scorefilepath, 'w', newline=''), columns=(c for c in self._df.columns if c not in self.exclude_from_save))
    
    def set_scores(self, evaluator:callable, fullpath=True):
        self._df['score'] = list(float(evaluator(k)) for k in self.image_files(fullpath))
        self.sort()

    def sort(self, by="score", add_rank_column=None, resort_after=True):
        self._df.sort_values(by=by, ascending=False, inplace=True)
        if add_rank_column: self._df[add_rank_column] = range(len(self._df))
        if resort_after and by!='score': self.sort()
    
    def image_files(self, fullpath=False):
        return self._df['path' if fullpath else 'relative_path'] 
    
    def element(self, label:str, file:str, is_fullpath=False) -> float:
        file = os.path.normpath(os.path.relpath(file, self.tld) if is_fullpath else file)
        return self._df.loc[file][label]

    def score(self, file:str, is_fullpath=False) -> float:
        return self.element('score', file, is_fullpath)
    
    def item(self, label) -> list:
        return list(self._df[label])
    
    def has_item(self, label):
        return label in self._df.columns
    
    def scores(self) -> list[float]:
        return self.item('score')
    
    def _dictionary(self, column:str):
        return {f:v for f,v in zip(self._df['relative_path'], self._df[column])}

    def scores_dictionary(self) -> dict[str,float]:
        return self._dictionary('score')
    
    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df
    
#if __name__=='__main__':
#    x = ImageScores.from_baid(top_level_directory=r"E:\BAID")
#    x.save_as_scorefile(r"E:\BAID\scores.csv")