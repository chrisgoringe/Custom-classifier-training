import os, json, re

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

def get_ab(scores, predicted_scores):
    right = 0
    total = 0
    for i in range(len(scores)):
        for j in range(i+1,len(scores)):
            if (predicted_scores[i]<predicted_scores[j] and scores[i]<scores[j]) or \
                (predicted_scores[i]>predicted_scores[j] and scores[i]>scores[j]): right += 1
            total += 1
    return right/total if total else 0

def clean(d):
    return { os.path.normpath(f) : d[f] for f in d }

class ImageScores:
    def __init__(self, image_scores:dict[str, float], top_level_directory:str):
        self.image_scores = clean(image_scores)
        self.top_level_directory = top_level_directory
        self.more = { }

    def add_item(self, label, values:dict):
        self.more[label] = clean(values)

    def saveable_dictionary(self, f):
        d = { "score": self.image_scores[f], }
        for item in self.more: d[item] = self.more[item].get(f,None)
        return d

    def save_as_scorefile(self, scorefilepath):
        saveable = { 
            "ImageRecords" : { f : self.saveable_dictionary(f) for f in self.image_scores } ,
            "Additionals" : list(item for item in self.more)
            }
        with open(scorefilepath, 'w') as f:
            print(json.dumps(saveable,indent=2), file=f)

    def subset(self, item:str, test:callable):
        assert item in self.more
        image_scores = {k : float(self.image_scores[k]) for k in self.image_scores if test(self.more[item][k])}
        new = ImageScores(image_scores, self.top_level_directory)
        for item in self.more:
            new.add_item(item, list(self.more[item][f] for f in image_scores))
        return new

    @classmethod
    def from_scorefile(cls, top_level_directory:str, scorefilename):
        with open(os.path.join(top_level_directory,scorefilename),'r') as f:
            image_scores_dict = json.load(f)
            if "ImageRecords" in image_scores_dict:
                image_scores = {k : float(image_scores_dict["ImageRecords"][k]['score']) for k in image_scores_dict["ImageRecords"]}
                imsc = ImageScores(image_scores, top_level_directory)
                for item in image_scores_dict.get("Additionals",['comparisons,']):
                    imsc.add_item(item, {k : float(image_scores_dict["ImageRecords"][k].get(item,0)) for k in image_scores_dict["ImageRecords"]})
                return imsc
            else:
                print("Old style score files are deprecated!")
                image_scores_dict.pop("#meta#",{})
                image_scores = {k : float(image_scores_dict[k][0]) if isinstance(image_scores_dict[k],list) else image_scores_dict[k] for k in image_scores_dict if keep(k)}
                comparisons = {k : int(image_scores_dict[k][1]) if isinstance(image_scores_dict[k],list) else 0 for k in image_scores_dict if keep(k)}
                imsc = ImageScores(image_scores, top_level_directory)
                imsc.add_item('comparisons', comparisons)
                return imsc
    
    @classmethod
    def from_evaluator(cls, evaluator:callable, images:list[str], top_level_directory, fullpath=True):
        image_scores = {k:float(evaluator(os.path.join(top_level_directory,k) if fullpath else k)) for k in images}
        return ImageScores(image_scores, top_level_directory)
    
    @classmethod
    def from_directory(cls, top_level_directory, evaluator:callable=lambda a:0):
        images = []
        for thing in os.listdir(top_level_directory):
            if valid_image(os.path.join(top_level_directory,thing)): images.append(thing)
            if valid_directory(os.path.join(top_level_directory,thing)):
                for subthing in os.listdir(os.path.join(top_level_directory,thing)):
                    if valid_image(os.path.join(top_level_directory,thing,subthing)): images.append(os.path.join(thing,subthing))
        return cls.from_evaluator(evaluator, images, top_level_directory)
    
    def set_scores(self, evaluator:callable):
        for k in self.image_scores: self.image_scores[k] = float(evaluator(os.path.join(self.top_level_directory,k)))
    
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
        if directory is not None:
            conds.append(lambda a:os.path.split(a)[0]==directory)
        return self._create_condition_stack(conds)

    def ranks(self):
        ordered = [(f,self.image_scores[f]) for f in self.image_scores]
        ordered.sort(key=lambda a:a[1], reverse=True)
        ranks = { f:i for i,f in enumerate(ordered)}
        return list(ranks[f] for f in self.image_scores)
    
    def score(self, file:str) -> float:
        return self.image_scores[os.path.normpath(file)]

    def scores(self, match:str=None, regex=True, directory=None) -> list[float]:
        condition = self._create_condition(match, regex, directory)
        return [self.image_scores[f] for f in self.image_scores if condition(f)]

    def scores_dictionary(self, match:str=None, regex=True, directory=None) -> dict[str,float]:
        condition = self._create_condition(match, regex, directory)
        return {f:self.image_scores[f] for f in self.image_scores if condition(f)}