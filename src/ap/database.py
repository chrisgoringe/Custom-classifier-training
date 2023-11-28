import random, json, os, math, shutil, regex, statistics
from PIL import Image
import time

class Database:
    def __init__(self, img_dir, args, k=0.7, low_count_weight=0):
        self.image_directory = img_dir
        self.args = args
        self.weight_fn = lambda a:math.pow(1-low_count_weight,a)
        self.load()
        self.recursive_add()
        self.keys = list(self.image_scores.keys())
        self.k = k
        self.stats = [0,0,0]     # agreed with db, disagreed with db, no prediction
        self.started = time.monotonic()
        self.validate()
        self.model_score_stats = [0,0,0,0,0]  
        self.model_scores = None
        self.set_scaling_function()
        
    def load(self):
        try:
            self.image_compare_count = {}
            with open(os.path.join(self.image_directory,"score.json"),'r') as f:
                self.image_scores = json.load(f)
                self.meta = self.image_scores.pop("#meta#",{})
                for im in self.image_scores:
                    if isinstance(self.image_scores[im],list):
                        self.image_compare_count[im] = self.image_scores[im][1]
                        self.image_scores[im] = self.image_scores[im][0]
                    else:
                        self.image_compare_count[im] = 0
            shutil.copyfile(os.path.join(self.image_directory,"score.json"), os.path.join(self.image_directory,"score-backup.json"))
        except:
            print(f"Database didn't reload scores from {os.path.join(self.image_directory,'score.json')}")
            self.image_scores = {}
            self.meta = {}
            self.image_compare_count = {}
        
    def save(self):
        with open(os.path.join(self.image_directory,"score.json"),'w') as f:
            self.replace_missing()
            to_save = {f:(self.image_scores[f],self.image_compare_count[f]) for f in self.image_scores}
            to_save['#meta#'] = self.meta
            print(json.dumps(to_save, indent=2),file=f)
            self.remove_missing()

    def set_model_score(self, scorer):
        self.model_scores = {f:scorer(os.path.join(self.image_directory,f)) for f in self.image_scores}

    def validate(self):
        self.missing_files = {f:self.image_scores[f] for f in self.image_scores if not os.path.exists(os.path.join(self.image_directory, f))}
        if self.missing_files:
            print(f"{len(self.missing_files)} image file{'s are' if len(self.missing_files)>1 else ' is'} in score.json but not found.")
            print("They will be kept in score.json but not made available to training.")
        if self.args['ignore_score_zero']:
            self.zeroes = {f:self.image_scores[f] for f in self.image_scores if self.image_scores[f]==0}
            if self.zeroes:
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

    def set_scaling_function(self):
        non_zeros = list(self.image_scores[f] for f in self.image_scores if self.image_scores[f]!=0)
        if len(non_zeros)>1: 
            mean = statistics.mean(non_zeros)
            stdev = statistics.stdev(non_zeros)
            self.scale = lambda a : (a-mean)/stdev
        else:
            self.scale = lambda a : a
    
    def scores_for_matching(self, reg):
        r = regex.compile(reg)
        return [self.image_scores[f] for f in self.image_scores if r.match(f)]
    
    def model_scores_for_matching(self,reg):
        r = regex.compile(reg)
        return [self.model_scores[f] for f in self.model_scores if r.match(f)]

    def pick_images(self, number):
        weights = list(self.weight_fn(self.image_compare_count[k]) for k in self.keys)
        self.ims = random.choices(self.keys,weights=weights,k=number)
        for i in range(number-1): 
            if self.ims[i] in self.ims[i+1:]: return self.pick_images(number)
        return (Image.open(os.path.join(self.image_directory,im)) for im in self.ims)

    def recursive_add(self, dir=None):
        dir = dir or self.image_directory
        for f in os.listdir(dir):
            full = os.path.join(dir,f) 
            if os.path.isdir(full): self.recursive_add(full)
            if os.path.splitext(f)[1] == ".png": 
                rel = os.path.relpath(full, self.image_directory)
                if not rel in self.image_scores: 
                    self.image_scores[rel] = 0
                    self.image_compare_count[rel] = 0

    def choices_made(self):
        return sum(self.stats)

    def choice_made(self, k):
        for i in range(len(self.ims)):
            if i==k: continue
            self.pair_choice_made(k,i)
        self.save()

    def pair_choice_made(self, win, loss):
        self.image_compare_count[self.ims[win]] += 1
        self.image_compare_count[self.ims[loss]] += 1
        
        db_delta = self.image_scores[self.ims[win]] - self.image_scores[self.ims[loss]]

        p = 1.0/(1.0+math.pow(10,-db_delta))
        self.image_scores[self.ims[win]] += self.k * (1-p)
        self.image_scores[self.ims[loss]] -= self.k * (1-p)

        if p>0.5: self.stats[0] += 1
        elif p<0.5: self.stats[1] += 1
        else: self.stats[2] += 1
        self.meta['evaluations'] = self.meta.get('evaluations',0) + 1

        if self.model_scores is not None:
            m_delta = (self.model_scores[self.ims[win]] - self.model_scores[self.ims[loss]])
            if db_delta!=0 and m_delta!=0:
                if m_delta>0 and db_delta>0: self.model_score_stats[0] += 1   # all agreedb, model, human
                elif db_delta<0 and m_delta<0: self.model_score_stats[3] += 1 # choice odd one out
                elif m_delta<0: self.model_score_stats[2] += 1                # model odd one out
                else: self.model_score_stats[1] += 1                          # db odd one out                       
            else:
                self.model_score_stats[4] += 1                                # one of db or model said draw
        

    def report(self):
        z = sum(self.image_compare_count[x]==0 for x in self.image_scores)
        print("{:>4} comparisons in {:>6.1f} s".format(sum(self.stats), time.monotonic()-self.started))
        print(f"{z}/{len(self.image_scores)} of the images have no comparisons yet")

        db_choice_match_percentage = 100*self.stats[0]/(self.stats[0]+self.stats[1])
        total_ever = sum(self.image_compare_count[x] for x in self.image_compare_count)

        if (self.stats[0]+self.stats[1]):
            print("{:>3} choices matched db, {:>3} contradicted db [{:>3} not predicted] = ({:>5.2f}%) ".format(
            *self.stats, db_choice_match_percentage))
        print("A total of {:>6} comparisons have been made for {:>5} images ({:>5.2f} per image)".format(
            self.meta['evaluations'], len(self.image_scores), 2*self.meta['evaluations']/len(self.image_scores)))
        if self.model_score_stats is not None and sum(self.model_score_stats): 
            strng = "all agree {:>3} ; model-choice agree {:>3} ; db-choice agree {:>3} ; db-model agree {:>3} "
            strng=strng.format(self.model_score_stats[0], self.model_score_stats[0]+self.model_score_stats[1], self.model_score_stats[0]+self.model_score_stats[2], self.model_score_stats[0]+self.model_score_stats[3] ) 
        else:
            strng = "{:>6} tests: {:>5.2f}%"
            strng=strng.format(total_ever, db_choice_match_percentage) 
        print(strng)
        print(strng, file=open("ab_stats.txt",'+a'))