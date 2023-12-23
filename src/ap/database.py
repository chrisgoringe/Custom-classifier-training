import random, json, os, math, shutil, re, statistics, time
from PIL import Image

class ImageChooser:
    def __init__(self, database):
        self.database = database

    def first_image_weight(self, k):
        return 1.0
    
    def other_image_weight(self, k, first_image):
        return self.first_image_weight(k)

    def first_image_weights(self):
        return list(self.first_image_weight(k) for k in self.database.keys)
    
    def other_image_weights(self, first_image):
        return list(self.other_image_weight(k, first_image) for k in self.database.keys)
    
    def pick_images(self, number):
        im1 = random.choices(self.database.keys, weights=self.first_image_weights(), k=1)[0]
        imx = list(random.choices(self.database.keys, weights=self.other_image_weights(im1), k=1)[0] for _ in range(number-1))
        ims = [im1,*imx]
        for i,im in enumerate(ims):
            if im in ims[i+1:]:
                return self.pick_images(number)
        return ims
            
class WeightedImageChooser(ImageChooser):
    def __init__(self, database, low_count_weight, controversy_weight):
        super().__init__(database)
        self.low_count_weight = low_count_weight
        self.controvery_weight = controversy_weight

    def first_image_weight(self, k):
        w = 1
        if self.low_count_weight:
            w *= math.pow(1-self.low_count_weight,self.database.image_compare_count[k])
        if self.controvery_weight:
            w *= math.pow(1+self.controvery_weight,abs(self.database.model_scores[k]-self.database.image_scores[k]))
        return w

class Database:
    def __init__(self, img_dir, args={}, k=0.7, low_count_weight=0, controversy_weight=0, threshold=None):
        self.image_directory = img_dir
        self.args = args
        self.threshold = threshold
        self.image_chooser = WeightedImageChooser(self, low_count_weight, controversy_weight)
        self.load()
        self.recursive_add()
        
        self.k = k
        self.stats = [0,0,0]     # agreed with db, disagreed with db, no prediction
        self.started = time.monotonic()
        self.validate()
        self.keys = list(self.image_scores.keys())
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

    def __len__(self):
        return len(self.image_scores)
        
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
        if self.args.get('ignore_score_zero',False):
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
        r = re.compile(reg)
        return [self.image_scores[f] for f in self.image_scores if r.match(f)]
    
    def model_scores_for_matching(self,reg):
        r = re.compile(reg)
        return [self.model_scores[f] for f in self.model_scores if r.match(f)]
    
    def sorted_list(self, best_first=False):
        image_list = list((f, self.image_scores[f], self.image_compare_count[f]) for f in self.image_scores)
        image_list.sort(key=lambda a:a[1], reverse=best_first)
        return image_list

    def pick_images(self, number):
        self.ims = self.image_chooser.pick_images(number)
        return (Image.open(os.path.join(self.image_directory,im)) for im in self.ims)

    def recursive_add(self, dir=None):
        dir = dir or self.image_directory
        for f in os.listdir(dir):
            full = os.path.join(dir,f) 
            if os.path.isdir(full): 
                self.recursive_add(full)
            else:
                try:
                    i = Image.open(full)
                    rel = os.path.relpath(full, self.image_directory)
                    if not rel in self.image_scores: 
                        self.image_scores[rel] = 0
                        self.image_compare_count[rel] = 0
                except:
                    print(f"{full} didn't load as an image")

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
        if self.threshold:
            z = sum(self.image_compare_count[x]<self.threshold for x in self.image_scores)
            print(f"{z}/{len(self.image_scores)} of the images have fewer than {self.threshold} comparisons")
        print("A total of {:>6} comparisons have been made for {:>5} images ({:>5.2f} per image)".format(
            self.meta['evaluations'], len(self.image_scores), 2*self.meta['evaluations']/len(self.image_scores)))
        
        if (self.stats[0]+self.stats[1])>self.stats[2]:
            db_choice_match_percentage = 100*self.stats[0]/(self.stats[0]+self.stats[1])
            total_ever = sum(self.image_compare_count[x] for x in self.image_compare_count)
            print("{:>3} choices matched db, {:>3} contradicted db [{:>3} not predicted] = ({:>5.2f}%) ".format(
            *self.stats, db_choice_match_percentage))
            strng = "{:>6} tests: {:>5.2f}%".format(total_ever, db_choice_match_percentage) 
            print(strng)
            print(strng, file=open("ab_stats.txt",'+a'))
        else:
            print("Not enough data yet to compare choices with database")

    def all_paths(self):
        return list(os.path.join(self.image_directory,f) for f in self.image_scores)
 