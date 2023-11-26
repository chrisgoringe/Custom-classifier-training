import random, json, os, math, shutil, regex
from PIL import Image
import time

class Database:
    def __init__(self, img_dir, args, k=0.7, weight_fn=lambda a:math.pow(0.8,a)):
        self.image_directory = img_dir
        self.args = args
        self.load()
        self.recursive_add()
        self.keys = list(self.image_scores.keys())
        self.weights = list(weight_fn(self.image_compare_count[k]) for k in self.keys)
        self.k = k
        self.stats = [0,0,0]     # agreed with db, disagreed with db, no prediction
        self.started = time.monotonic()
        self.validate()
        self.model_score_stats = [0,0,0,0,0]  
        self.model_scores = None
        
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
            print("Didn't reload scores")
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

    def scores_for_matching(self, reg):
        r = regex.compile(reg)
        return [self.image_scores[f] for f in self.image_scores if r.match(f)]
    
    def model_scores_for_matching(self,reg):
        r = regex.compile(reg)
        return [self.model_scores[f] for f in self.model_scores if r.match(f)]

    def pick_images_and_scores(self):
        self.im1, self.im2 = random.choices(self.keys,weights=self.weights,k=2)
        if self.im1==self.im2: return self.pick_images_and_scores()
        return (Image.open(os.path.join(self.image_directory,self.im1)), 
                Image.open(os.path.join(self.image_directory,self.im2)))

    def recursive_add(self, dir=None):
        dir = dir or self.image_directory
        for f in os.listdir(dir):
            full = os.path.join(dir,f) 
            if os.path.isdir(full): self.recursive_add(full)
            if os.path.splitext(f)[1] == ".png" and dir!=self.image_directory: 
                rel = os.path.relpath(full, self.image_directory)
                if not rel in self.image_scores: 
                    self.image_scores[rel] = 0
                    self.image_compare_count[rel] = 0

    def choices_made(self):
        return sum(self.stats)

    def choice_made(self, k):
        self.image_compare_count[self.im1] += 1
        self.image_compare_count[self.im2] += 1
        
        db_delta = self.image_scores[self.im1] - self.image_scores[self.im2]
        if k=='1':
            p = 1.0/(1.0+math.pow(10,-db_delta))
            self.image_scores[self.im1] += self.k * (1-p)
            self.image_scores[self.im2] -= self.k * (1-p)
        elif k=='2':
            p = 1.0/(1.0+math.pow(10,+db_delta))
            self.image_scores[self.im2] += self.k * (1-p)
            self.image_scores[self.im1] -= self.k * (1-p)
        if p>0.5: self.stats[0] += 1
        elif p<0.5: self.stats[1] += 1
        else: self.stats[2] += 1
        self.meta['evaluations'] = self.meta.get('evaluations',0) + 1
        if self.model_scores is not None:
            if db_delta!=0 and self.model_scores[self.im1] != self.image_scores[self.im2]:
                m_delta = 1 if (self.model_scores[self.im1] - self.image_scores[self.im2])>1 else -1
                db_delta = 1 if db_delta>0 else -1
                choice_delta = 1 if k=='1' else -1
                if choice_delta==m_delta and choice_delta==db_delta: self.model_score_stats[0] += 1 # all agreedb, model, human
                elif choice_delta==m_delta: self.model_score_stats[1] += 1                          # db odd one out
                elif choice_delta==db_delta: self.model_score_stats[2] += 1                         # model odd one out
                else: self.model_score_stats[3] += 1                                                # choice odd one out
            else:
                self.model_score_stats[4] += 1                                                      # one of db or model said draw
        self.save()

    def report(self):
        z = sum(self.image_scores[x]==0 for x in self.image_scores)
        print("{:>4} image pairs in {:>6.1f} s".format(sum(self.stats), time.monotonic()-self.started))
        print(f"{z}/{len(self.image_scores)} of the images are rated zero")
        db_choice_match_percentage = 100*self.stats[0]/(self.stats[0]+self.stats[1])
        total_ever = sum(self.image_compare_count[x] for x in self.image_compare_count)
        if (self.stats[0]+self.stats[1]):
            print("{:>3} choices matched db, {:>3} contradicted db [{:>3} not predicted] = ({:>5.2f}%) ".format(
            *self.stats, db_choice_match_percentage))
        print("A total of {:>6} comparisons have been made for {:>5} images ({:>5.2f} per image)".format(
            self.meta['evaluations'], len(self.image_scores), 2*self.meta['evaluations']/len(self.image_scores)))
        if self.model_score_stats is not None and sum(self.model_score_stats): 
            strng = "{:>3} all agree; {:>3} db odd-one-out; {:>3} model odd-one-out; {:>3} human odd-one-out; {:>3} model or db draw; {:>5.2f}%"
            strng=strng.format(*self.model_score_stats, db_choice_match_percentage) 
            print(strng)
            print(strng, file=open("ab_stats.txt",'+a'))
        strng = "{:>6} tests: {:>5.2f}%"
        strng=strng.format(total_ever, db_choice_match_percentage) 
        print(strng)
        print(strng, file=open("ab_stats.txt",'+a'))