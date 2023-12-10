import os, json
import pandas as pd
import random, json


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
    def __init__(self, top_level:str, save_model_folder:str=None, fraction_for_test:float=0.25, test_pick_seed:int=42, use_score_file=True):
        self.df = pd.DataFrame(columns=["image","label_str","split"])
        self.labels = []
        self.fraction_for_test = fraction_for_test
        random.seed(test_pick_seed)

        if os.path.exists(os.path.join(top_level,'score.json')) and use_score_file:
            self.dataframe_from_scorefile(top_level)
        elif os.path.exists(os.path.join(top_level,'score.csv')) and use_score_file:
            self.dataframe_from_csv(top_level)
        else:
            self.dataframe_from_directory_structure(top_level)
            if save_model_folder:    
                with open(os.path.join(save_model_folder,"categories.json"),'w') as f:
                    json.dump({"categories":self.labels}, f)

        self.describe()
        
    def split(self) -> str:
        return "test" if random.random() < self.fraction_for_test else "train"
    
    def weights(self):
        return {label:len(self.df)/len(self.df[self.df['label_str']==label]) for label in self.labels} if self.labels else {}
    
    def describe(self):
        if self.labels:
            for label in self.labels:
                dfl = self.df[self.df['label_str']==label]
                print("{:>10} contains {:>4} images, {:>4} train and {:>3} evaluation".format(label, len(dfl), len(dfl[dfl['split']=='train']), len(dfl[dfl['split']=='test']))) 
    
    def dataframe_from_scorefile(self, image_folder):
        with open(os.path.join(image_folder,"score.json"),'r') as f:
            image_scores = json.load(f)
            image_scores.pop('#meta#',None)
            for f in image_scores:
                self.df.loc[len(self.df)] = [os.path.join(image_folder,f), str(image_scores[f][0] if isinstance(image_scores[f],list) else image_scores[f]), self.split()]

    def dataframe_from_csv(self, image_folder):
        with open(os.path.join(image_folder,"score.csv"),'r') as f:
            for line in f.readlines():
                score, file = line.split(",")
                score = score.strip()
                try:
                    float(score)
                except:
                    continue
                file = file.strip()
                fullfile = os.path.join(image_folder,file)
                if os.path.exists(fullfile):
                    self.df.loc[len(self.df)] = [fullfile, score, self.split()]
                else:
                    print(f"{file} not found")

    def dataframe_from_directory_structure(self, image_folder):
        self.labels = sorted([d for d in os.listdir(image_folder) if valid_directory(os.path.join(image_folder,d))])
        for label in self.labels:
            dir = os.path.join(image_folder,label)
            for file in os.listdir(dir):
                if valid_image(os.path.join(dir,file)):
                    self.df.loc[len(self.df)] = [os.path.join(dir,file), label, self.split()]

    def get_dataframe(self) -> pd.DataFrame:
        return self.df 
    
