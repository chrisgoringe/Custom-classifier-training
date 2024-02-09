import os, json, statistics, re, torch

def compress_rank(ranks:list):
    ordered = sorted(ranks)
    return [ordered.index(r) for r in ranks]

def valid_image(filepath:str):
    if os.path.basename(filepath).startswith("."): return False
    _,ext = os.path.splitext(filepath)
    return (ext in ['.png','.jpg','.jpeg'])

def valid_directory(dir_path:str):
    if not os.path.isdir(dir_path): return False
    if os.path.basename(dir_path).startswith("."): return False
    return True

class ImageScores:
    def __init__(self, image_scores:dict[str, float], top_level_directory:str, normalisation, comparisons:dict[str,int]={}):
        clean = lambda d : { os.path.normpath(f) : d[f] for f in d }
        self.image_scores = clean(image_scores)
        self.normaliser = self.create_normaliser() if normalisation else lambda a:a
        self.set_rankings()
        self.top_level_directory = top_level_directory
        self.comparisons = clean(comparisons)

    def save_as_scorefile(self, scorefilepath):
        saveable = { "ImageRecords" : { f : {
                                                "relative_filepath": f,
                                                "comparisons": self.comparisons.get(f,0),
                                                "score": self.image_scores[f]            
                                            } for f in self.image_scores } 
        }
        with open(scorefilepath, 'w') as f:
            print(json.dumps(saveable,indent=2), file=f)

    @classmethod
    def from_scorefile(cls, top_level_directory:str, scorefilename, normalisation=False):
        with open(os.path.join(top_level_directory,scorefilename),'r') as f:
            image_scores_dict = json.load(f)
            if "ImageRecords" in image_scores_dict:
                image_scores = {k : float(image_scores_dict["ImageRecords"][k]['score']) for k in image_scores_dict["ImageRecords"]}
                comparisons = {k : float(image_scores_dict["ImageRecords"][k].get('comparisons',0)) for k in image_scores_dict["ImageRecords"]}
            else:
                image_scores_dict.pop("#meta#",{})
                image_scores = {k : float(image_scores_dict[k][0]) if isinstance(image_scores_dict[k],list) else image_scores_dict[k] for k in image_scores_dict}
                comparisons = {k : int(image_scores_dict[k][1]) if isinstance(image_scores_dict[k],list) else 0 for k in image_scores_dict}
        return ImageScores(image_scores, top_level_directory, normalisation=normalisation, comparisons=comparisons)
    
    @classmethod
    def from_evaluator(cls, evaluator:callable, images:list[str], top_level_directory, normalisation=False, fullpath=True):
        image_scores = {k:float(evaluator(os.path.join(top_level_directory,k) if fullpath else k)) for k in images}
        return ImageScores(image_scores, top_level_directory, normalisation=normalisation)
    
    @classmethod
    def from_directory(cls, top_level_directory, evaluator:callable=lambda a:0, normalisation=False):
        images = []
        for thing in os.listdir(top_level_directory):
            if valid_image(os.path.join(top_level_directory,thing)): images.append(thing)
            if valid_directory(os.path.join(top_level_directory,thing)):
                for subthing in os.listdir(os.path.join(top_level_directory,thing)):
                    if valid_image(os.path.join(top_level_directory,thing,subthing)): images.append(os.path.join(thing,subthing))
        return cls.from_evaluator(evaluator, images, top_level_directory, normalisation)
    
    def set_scores(self, evaluator:callable, normalisation=False):
        for k in self.image_scores: self.image_scores[k] = float(evaluator(os.path.join(self.top_level_directory,k)))
        self.normaliser = self.create_normaliser() if normalisation else lambda a:a
        self.set_rankings()

    def create_normaliser(self) -> callable:
        raise Exception("really?")
        mean = statistics.mean(self.image_scores[k] for k in self.image_scores)
        stdev = statistics.stdev(self.image_scores[k] for k in self.image_scores)
        return lambda a : float( (a-mean)/stdev )
    
    def image_files(self, fullpath=False) -> list[str]:
        if fullpath:
            return list(os.path.join(self.top_level_directory,k) for k in self.image_scores)
        return list(k for k in self.image_scores) 

    def _create_condition_stack(self, *args) -> callable:
        def condition(a):
            for cond in args: 
                if isinstance(cond,list):
                    for c in cond:
                        if not c(a): return False
                else:
                    if not cond(a): return False
            return True
        return condition
    
    def _create_condition(self, match:str, regex:bool, directory:str) -> callable:
        conds = []
        if match:
            if regex:
                r = re.compile(match)
                conds.append(lambda a : r.match(a))
            else:
                conds.append(lambda a : match in a)
        if directory:
            conds.append(lambda a:os.path.split(a)[0]==directory)
        return self._create_condition_stack(conds)

    def set_rankings(self):
        ordered = [(f,self.image_scores[f]) for f in self.image_scores]
        ordered.sort(key=lambda a:a[1], reverse=True)
        self.ranked = {f:0 for f in self.image_scores}
        for i, f in enumerate(f for f, _ in ordered):
            self.ranked[f] = i

    def ranks(self):
        return [self.ranked[f] for f in self.ranked]
    
    def score(self, file:str, normalised=True) -> float:
        normaliser = self.normaliser if normalised else lambda a : a
        return normaliser(self.image_scores[file])

    def scores(self, match:str=None, regex=True, normalised=True, rankings=False, compressed=True, directory=None) -> list[float]:
        condition = self._create_condition(match, regex, directory)
        if rankings:
            ranks = [self.ranked[f] for f in self.image_scores if condition(f)]
            return compress_rank(ranks) if compressed else ranks
        else:
            normaliser = self.normaliser if normalised else lambda a : a
            return [normaliser(self.image_scores[f]) for f in self.image_scores if condition(f)]

    def scores_dictionary(self, match:str=None, regex=True, normalised=True, directory=None) -> dict[str,float]:
        condition = self._create_condition(match, regex, directory)
        normaliser = self.normaliser if normalised else lambda a : a
        return {f:normaliser(self.image_scores[f]) for f in self.image_scores if condition(f)}
    
