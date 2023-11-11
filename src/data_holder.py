import os, json
import pandas as pd
import random

class DataHolder:
    def __init__(self, top_level:str, model_folder:str, fraction_for_test:float, test_pick_seed:int):
        self.directories = [os.path.join(top_level,d) for d in os.listdir(top_level) if os.path.isdir(os.path.join(top_level,d))]
        if model_folder:
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            with open(os.path.join(model_folder,"categories.json"),'w') as f:
                json.dump({"categories":[os.path.basename(d) for d in self.directories]}, f)
        self.fraction_for_test = fraction_for_test
        self.test_pick_seed = test_pick_seed

    def split(self) -> str:
        return "test" if random.random() < self.fraction_for_test else "train"

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=["image","label_str","label","split"])
        self.accum = 0.0
        random.seed(self.test_pick_seed)
        for i, dir in enumerate(self.directories):
            for file in os.listdir(dir):
                df.loc[len(df)] = [os.path.join(dir,file), os.path.basename(dir), i, self.split()]
        return df