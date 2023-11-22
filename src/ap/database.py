import random, json, os, math, shutil, regex
from PIL import Image
import time

class Database:
    def __init__(self, img_dir, args, k=0.7):
        self.image_directory = img_dir
        self.args = args
        self.load()
        self.recursive_add()
        self.keys = list(self.image_scores.keys())
        self.k = k
        self.stats = [0,0,0]
        self.started = time.monotonic()
        self.validate()

    def load(cls):
        try:
            with open(os.path.join(cls.image_directory,"score.json"),'r') as f:
                cls.image_scores = json.load(f)
                cls.meta = cls.image_scores.pop("#meta#",{})
            shutil.copyfile(os.path.join(cls.image_directory,"score.json"), os.path.join(cls.image_directory,"score-backup.json"))
        except:
            print("Didn't reload scores")
            cls.image_scores = {}
            cls.meta = {}

    def save(self):
        with open(os.path.join(self.image_directory,"score.json"),'w') as f:
            self.image_scores['#meta#'] = self.meta
            self.replace_missing()
            print(json.dumps(self.image_scores, indent=2),file=f)
            self.image_scores.pop('#meta#')
            self.remove_missing()

    def validate(self):
        self.missing_files = {f:self.image_scores[f] for f in self.image_scores if not os.path.exists(os.path.join(self.image_directory, f))}
        if self.missing_files:
            print(f"{len(self.missing_files)} image file{'s are' if len(self.missing_files)>1 else ' is'} in score.json but not found.")
            print("They will be kept in score.json but not made available to training.")
        if self.args['ignore_score_zero']:
            self.zeroes = {f:self.image_scores[f] for f in self.image_scores if self.image_scores[f]==0}
            print(f"{len(self.zeroes)} image file{'s are' if len(self.zeroes)>1 else ' is'} in score.json but have score 0.")
            print("They will be kept in score.json but not made available to training.")
        else:
            self.zeroes = {}
        self.remove_missing()

    def remove_missing(self):
        for file in self.missing_files: self.image_scores.pop(file)
        for file in self.zeroes: self.image_scores.pop(file,None)  # in case it's in both

    def replace_missing(self):
        for file in self.zeroes: self.image_scores[file] = self.zeroes[file]
        for file in self.missing_files: self.image_scores[file] = self.missing_files[file]

    def scores_for_matching(self, reg):
        r = regex.compile(reg)
        return [self.image_scores[f] for f in self.image_scores if r.match(f)]

    def pick_images_and_scores(self):
        self.im1, self.im2 = random.sample(self.keys,2)
        self.s1 = self.image_scores[self.im1]
        self.s2 = self.image_scores[self.im2]
        return (Image.open(os.path.join(self.image_directory,self.im1)), 
                Image.open(os.path.join(self.image_directory,self.im2)))

    def recursive_add(self, dir=None):
        dir = dir or self.image_directory
        for f in os.listdir(dir):
            full = os.path.join(dir,f) 
            if os.path.isdir(full): self.recursive_add(full)
            if os.path.splitext(f)[1] == ".png" and dir!=self.image_directory: 
                rel = os.path.relpath(full, self.image_directory)
                if not rel in self.image_scores: self.image_scores[rel] = 0

    def choice_made(self, k):
        if k=='1':
            delta = self.s1 - self.s2
            p = 1.0/(1.0+math.pow(10,-delta))
            self.image_scores[self.im1] += self.k * (1-p)
            self.image_scores[self.im2] -= self.k * (1-p)
        elif k=='2':
            delta = self.s2 - self.s1
            p = 1.0/(1.0+math.pow(10,-delta))
            self.image_scores[self.im2] += self.k * (1-p)
            self.image_scores[self.im1] -= self.k * (1-p)
        if delta>0: self.stats[0] += 1
        if delta<0: self.stats[1] += 1
        if delta==0: self.stats[2] += 1
        self.meta['evaluations'] = self.meta.get('evaluations',0) + 1
        self.save()

    def report(self):
        z = sum(self.image_scores[x]==0 for x in self.image_scores)
        print("{:>4} image pairs in {:>6.1f} s".format(sum(self.stats), time.monotonic()-self.started))
        print(f"{z}/{len(self.image_scores)} of the images are rated zero")
        if (self.stats[0]+self.stats[1]):
            print("{:>3} choices matched prediction, {:>3} contradicted prediction [{:>3} not predicted] = ({:>5.2f}%) ".format(
            *self.stats, 100*self.stats[0]/(self.stats[0]+self.stats[1])))
        print("A total of {:>6} comparisons have been made for {:>5} images ({:>5.2f} per image)".format(
            self.meta['evaluations'], len(self.image_scores), 2*self.meta['evaluations']/len(self.image_scores)))