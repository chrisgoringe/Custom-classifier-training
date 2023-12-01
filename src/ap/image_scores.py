import os, json, statistics, re

def compress_rank(ranks:list):
    ordered = sorted(ranks)
    return [ordered.index(r) for r in ranks]

class ImageScores:
    def __init__(self, image_scores:dict[str, float], image_directory:str):
        self.image_scores:dict[str, float] = image_scores
        self.normaliser = self.create_normaliser()
        self.set_rankings()
        self.image_directory = image_directory

    @classmethod
    def from_scorefile(cls, image_directory:str):
        with open(os.path.join(image_directory,"score.json"),'r') as f:
            image_scores = json.load(f)
            image_scores.pop("#meta#",{})
            for k in image_scores: image_scores[k] = float(image_scores[k][0])
        return ImageScores(image_scores, image_directory)
    
    @classmethod
    def from_evaluator(cls, evaluator:callable, images:list[str], image_directory):
        image_scores = {k:float(evaluator(os.path.join(image_directory,k))) for k in images}
        return ImageScores(image_scores, image_directory)
    
    def set_scores(self, evaluator:callable):
        for k in self.image_scores: self.image_scores[k] = float(evaluator(os.path.join(self.image_directory,k)))
        self.normaliser = self.create_normaliser()
        self.set_rankings()

    def create_normaliser(self) -> callable:
        mean = statistics.mean(self.image_scores[k] for k in self.image_scores)
        stdev = statistics.stdev(self.image_scores[k] for k in self.image_scores)
        return lambda a : float( (a-mean)/stdev )
    
    def image_files(self, fullpath=False) -> list[str]:
        if fullpath:
            return list(os.path.join(self.image_directory,k) for k in self.image_scores)
        return list(k for k in self.image_scores) 

    def _create_condition(self, match:str, regex:bool) -> callable:
        if match:
            if regex:
                r = re.compile(match)
                return lambda a : r.match(a)
            else:
                return lambda a : match in a
        else:
            return lambda a : True

    def set_rankings(self):
        ordered = [(f,self.image_scores[f]) for f in self.image_scores]
        ordered.sort(key=lambda a:a[1], reverse=True)
        self.ranked = {f:0 for f in self.image_scores}
        for i, f in enumerate(f for f, _ in ordered):
            self.ranked[f] = i

    def ranks(self):
        return [self.ranked[f] for f in self.ranked]

    def scores(self, match:str=None, regex=True, normalised=True, rankings=False) -> list[float]:
        condition = self._create_condition(match, regex)
        if rankings:
            ranks = [self.ranked[f] for f in self.image_scores if condition(f)]
            return compress_rank(ranks)
        else:
            normaliser = self.normaliser if normalised else lambda a : a
            return [normaliser(self.image_scores[f]) for f in self.image_scores if condition(f)]

    def scores_dictionary(self, match:str=None, regex=True, normalised=True) -> dict[str,float]:
        condition = self._create_condition(match, regex)
        normaliser = self.normaliser if normalised else lambda a : a
        return {f:normaliser(self.image_scores[f]) for f in self.image_scores if condition(f)}