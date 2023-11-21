import customtkinter
import random, json, os, math
from PIL import Image
import time

class Database:
    def __init__(self, img_dir, k=0.5):
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
        except:
            print("Didn't reload scores")
            cls.image_scores = {}

    def save(self):
        with open(os.path.join(self.image_directory,"score.json"),'w') as f:
            print(json.dumps(self.image_scores, indent=2),file=f)

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
        self.save()

    def report(self):
        z = sum(self.image_scores[x]==0 for x in self.image_scores)
        print("{:>4} images in {:>6.1f} s".format(sum(self.stats), time.monotonic()-self.started))
        print(f"{z}/{len(self.image_scores)} are rated zero")
        print("{:>3} choices matched prediction, {:>3} contradicted prediction [{:>3} not predicted] = ({:>5.2f}%) ".format(
            *self.stats, 100*self.stats[0]/(self.stats[0]+self.stats[1])))

class TheApp:
    def __init__(self, h, db:Database):
        self.app = customtkinter.CTk()
        self.app.title("H.A.S.")
        self.app.geometry(f"{h*2}x{h}")
        self.a = customtkinter.CTkLabel(self.app, text="")
        self.b = customtkinter.CTkLabel(self.app, text="")
        self.a.grid(row=0, column=0)
        self.b.grid(row=0, column=1)
        self.db = db
        self.h = h
        self.app.bind("<KeyRelease>", self.keyup)
        self.pick_images()
        
    def pick_images(self):
        im1, im2 = self.db.pick_images_and_scores()
        self.a.configure(image = customtkinter.CTkImage(light_image=im1, size=(int(self.h*im1.width/im2.height),self.h)))
        self.b.configure(image = customtkinter.CTkImage(light_image=im2, size=(int(self.h*im2.width/im2.height),self.h)))

    def keyup(self,k):
        k = k.char
        if (k=='q'): 
            self.db.report()
            self.app.quit()
        elif k=='1' or k=='2': 
            self.db.choice_made(k)
            self.pick_images()
        elif k=='r':
            self.db.report()

def main():
    db = Database("C:/Users/chris/Documents/GitHub/ComfyUI_windows_portable/ComfyUI/output/compare")
    TheApp(600, db).app.mainloop()

if __name__=='__main__':
    main()