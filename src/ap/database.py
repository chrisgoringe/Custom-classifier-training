import random, json, os, math
from PIL import Image
import time

class Database:
    def __init__(self, img_dir, k=0.7):
        self.image_directory = img_dir
        self.load()
        self.recursive_add()
        self.keys = list(self.image_scores.keys())
        self.k = k
        self.stats = [0,0,0]
        self.started = time.monotonic()

    def load(cls):
        try:
            with open(os.path.join(cls.image_directory,"score.json"),'r') as f:
                cls.image_scores = json.load(f)
                cls.meta = cls.image_scores.pop("#meta#",{})
        except:
            print("Didn't reload scores")
            cls.image_scores = {}
            cls.meta = {}

    def save(self):
        with open(os.path.join(self.image_directory,"score.json"),'w') as f:
            self.image_scores['#meta#'] = self.meta
            print(json.dumps(self.image_scores, indent=2),file=f)
            self.image_scores.pop('#meta#')

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
            if os.path.splitext(f)[1] == ".png": 
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
        print("{:>4} images in {:>6.1f} s".format(sum(self.stats), time.monotonic()-self.started))
        print(f"{z}/{len(self.image_scores)} of the images are rated zero")
        print("{:>3} choices matched prediction, {:>3} contradicted prediction [{:>3} not predicted] = ({:>5.2f}%) ".format(
            *self.stats, 100*self.stats[0]/(self.stats[0]+self.stats[1])))
        print("A total of {:>6} comparisons have been made for {:>5} images ({:>5.2f} per image)".format(
            self.meta['evaluations'], len(self.image_scores), 2*self.meta['evaluations']/len(self.image_scores)))