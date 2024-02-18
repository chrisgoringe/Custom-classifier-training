import os, json
import pandas as pd
from typing import Self, Callable

class ImageScores:
    def __init__(self, top_level_directory:str, files:list=None, scores:list=None, df:pd.DataFrame=None):
        self.tld = top_level_directory
        self._df = pd.DataFrame(columns=['relative_path', 'path', 'score']) if df is None else df
        if files:
            self._df['relative_path'] = list(os.path.normpath(f) for f in files)
            self._df['path'] = list(os.path.normpath(os.path.join(self.tld,f)) for f in self._df['relative_path'])
            if scores: 
                self._df['scores'] = scores
            else:
                scores._df['scores'] = [0]*len(files)
        
    def subset(self, test:Callable, item:str='relative_path') -> Self:
        return ImageScores(top_level_directory=self.tld, df=self._df.loc[test(self._df[item])])

    @classmethod
    def from_scorefile(cls, top_level_directory:str, scorefilename) -> Self:
        if os.path.splitext(scorefilename)[1]==".json":
            with open(os.path.join(top_level_directory,scorefilename),'r') as f:
                image_scores_dict = json.load(f)
                if "ImageRecords" in image_scores_dict:
                    files = list(k for k in image_scores_dict["ImageRecords"])
                    scores = list(float(image_scores_dict["ImageRecords"][k]['score']) for k in files)
                    imsc = ImageScores(top_level_directory, files=files, scores=scores)
                    for item in image_scores_dict.get("Additionals",['comparisons,']):
                        imsc.add_item(item, list(image_scores_dict["ImageRecords"][k].get(item,0) for k in files))
                    return imsc
        elif os.path.splitext(scorefilename)[1]==".csv":
            imsc = ImageScores(top_level_directory)
            imsc._df = pd.read_csv(os.path.join(top_level_directory,scorefilename))
    
    @classmethod
    def from_evaluator(cls, evaluator:Callable, images:list[str], top_level_directory, fullpath=True) -> Self:
        scores = list(float(evaluator(os.path.join(top_level_directory,k) if fullpath else k)) for k in images)
        return ImageScores(top_level_directory, files=images, scores=scores) 

    @classmethod
    def from_directory(cls, top_level_directory, evaluator:Callable=lambda a:0) -> Self:
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
    
    def add_item(self, label, values:dict|list|Callable|Self, fullpath=False):
        if isinstance(values, dict):
            self._df[label] = list(values[f] for f in self.image_files(fullpath=fullpath))
        elif isinstance(values, list):
            self._df[label] = list(v for v in values)
        elif callable(values):
            self._df[label] = list(values(f) for f in self.image_files(fullpath=fullpath))
        elif isinstance(values, ImageScores):
            self._df[label] = list( values.score(f) for f in self.image_files() )
        else:
            raise NotImplementedError()

    def save_as_scorefile(self, scorefilepath):
        self._df.to_csv(open(scorefilepath, 'w', newline=''), columns=(c for c in self._df.columns if c!='path'))
    
    def set_scores(self, evaluator:callable, fullpath=True):
        self._df['scores'] = list(float(evaluator(k)) for k in self.image_files(fullpath))
        self.sort()

    def sort(self, by="score", add_rank_column=None, resort_after=True):
        self._df.sort_values(by=by, ascending=False, inplace=True)
        if add_rank_column: self._df[add_rank_column] = range(len(self._df))
        if resort_after and by!='score': self.sort()
    
    def image_files(self, fullpath=False):
        return self._df['path' if fullpath else 'relative_path'] 
    
    def _element(self, column:str, file:str, is_fullpath=False) -> float:
        file = os.path.normpath(os.path.relpath(file, self.tld) if is_fullpath else file)
        return self._df.loc[file][column]

    def score(self, file:str, is_fullpath=False) -> float:
        return self._element('score', file, is_fullpath)
    
    def _dictionary(self, column:str):
        return {f:v for f,v in zip(self._df['relative_path'], self._df[column])}

    def scores_dictionary(self) -> dict[str,float]:
        return self._dictionary('score')