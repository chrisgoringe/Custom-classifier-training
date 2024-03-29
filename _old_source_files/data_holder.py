import os, json
import pandas as pd
import random, json
from .ap.image_scores import ImageScores

def valid_image(filepath:str):
    if os.path.basename(filepath).startswith("."): return False
    _,ext = os.path.splitext(filepath)
    return (ext in ['.png','.jpg','.jpeg'])

def valid_directory(dir_path:str):
    if not os.path.isdir(dir_path): return False
    if os.path.basename(dir_path).startswith("."): return False
    for f in os.listdir(dir_path):
        if valid_image(os.path.join(dir_path,f)): return True
    return False

class DataHolder:
    def __init__(self, top_level:str, fraction_for_eval:float=0.25, eval_pick_seed:int=42, use_score_file:str=None):
        self.df = pd.DataFrame(columns=["image","score","split"])
        self.labels = []
        self.fraction_for_eval = fraction_for_eval
        self.top_level = top_level
        if eval_pick_seed: random.seed(eval_pick_seed)

        if use_score_file and os.path.exists(os.path.join(top_level,use_score_file)):
            self.dataframe_from_scorefile(top_level, use_score_file)
        elif os.path.exists(os.path.join(top_level,'score.csv')) and use_score_file:
            self.dataframe_from_csv(top_level)
        else:
            raise Exception("No score file")

        self.describe()

    def save_split(self, filename="split.json"):
        if filename:
            with open(os.path.join(self.top_level, filename), 'w') as fhdl:
                print(json.dumps(self.dictionary('split'), indent=2), file=fhdl)
        
    def split(self) -> str:
        return "eval" if random.random() < self.fraction_for_eval else "train"
    
    def dictionary(self, column):
        return { os.path.relpath(f, self.top_level) : s for f, s in zip(self.df['image'], self.df[column]) }
    
    #def weights(self):
    #    return {label:len(self.df)/len(self.df[self.df['label_str']==label]) for label in self.labels} if self.labels else {}
    
    def describe(self):
        if self.labels:
            for label in self.labels:
                dfl = self.df[self.df['score']==label]
                print("{:>10} contains {:>4} images, {:>4} train and {:>3} evaluation".format(label, len(dfl), len(dfl[dfl['split']=='train']), len(dfl[dfl['split']=='eval']))) 
    
    def dataframe_from_scorefile(self, image_folder, scorefile):
        with open(os.path.join(image_folder,scorefile),'r') as f:
            image_scores = json.load(f)
            if "ImageRecords" in image_scores:
                for f in image_scores["ImageRecords"]:
                    self.df.loc[len(self.df)] = [os.path.join(image_folder,f), float(image_scores["ImageRecords"][f]['score']), self.split()]
            else:
                image_scores.pop('#meta#',None)
                for f in image_scores:
                    self.df.loc[len(self.df)] = [os.path.join(image_folder,f), float(image_scores[f][0] if isinstance(image_scores[f],list) else image_scores[f]), self.split()]

    def dataframe_from_csv(self, image_folder):
        with open(os.path.join(image_folder,"score.csv"),'r') as f:
            for line in f.readlines():
                try:
                    score, file = line.split(",")
                    score = score.strip()
                    score = float(score)
                    file = file.strip()
                except:
                    print(f"Ignoring {line} because it isn't in the form score,filename")
                    continue
                fullfile = os.path.join(image_folder,file)
                if os.path.exists(fullfile):
                    self.df.loc[len(self.df)] = [fullfile, score, self.split()]
                else:
                    print(f"{file} not found")

    def dataframe_from_directory_structure(self, image_folder):
        self.labels = sorted([d for d in os.listdir(image_folder) if valid_directory(os.path.join(image_folder,d))])
        for label in self.labels:
            try:
                score = float(label.split("_")[0])
            except:
                print(f"Ignoring folder {label} because it isn't in the form number_name")
                continue
            dir = os.path.join(image_folder,label)
            for file in os.listdir(dir):
                if valid_image(os.path.join(dir,file)):
                    self.df.loc[len(self.df)] = [os.path.join(dir,file), score, self.split()]

    def get_dataframe(self) -> pd.DataFrame:
        return self.df 
    
    def get_image_scores(self) -> ImageScores:
        return ImageScores( image_scores=self.dictionary('score'), top_level_directory=self.top_level, normalisation=False )