import os, json
import pandas as pd
import random

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
    def __init__(self, top_level:str, model_folder:str, fraction_for_test:float, test_pick_seed:int):
        self.directories = [os.path.join(top_level,d) for d in os.listdir(top_level) if valid_directory(os.path.join(top_level,d))]
        if model_folder:
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            with open(os.path.join(model_folder,"categories.json"),'w') as f:
                json.dump({"categories":[os.path.basename(d) for d in self.directories]}, f)
        self.fraction_for_test = fraction_for_test
        self.test_pick_seed = test_pick_seed
        self.df = None

    def split(self) -> str:
        return "test" if random.random() < self.fraction_for_test else "train"
    
    def image_stats(self):
        pass

    def get_dataframe(self) -> pd.DataFrame:
        if self.df is None:
            self.df = pd.DataFrame(columns=["image","label_str","label","split"])
            self.accum = 0.0
            random.seed(self.test_pick_seed)
            self.sizes = []
            for i, dir in enumerate(self.directories):
                before = len(self.df)
                for file in os.listdir(dir):
                    if valid_image(os.path.join(dir,file)):
                        self.df.loc[len(self.df)] = [os.path.join(dir,file), os.path.basename(dir), i, self.split()]
                print(f"{os.path.basename(dir)} contains {len(self.df)-before} images")
                self.sizes.append(len(self.df)-before)
            test_images = len(self.df[self.df["split"]=="test"])
            print(f"{len(self.df)} total images ({test_images} test, {len(self.df)-test_images} train)")
        return self.df 