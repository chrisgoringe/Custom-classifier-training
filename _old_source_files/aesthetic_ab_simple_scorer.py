import customtkinter
from src.ap.database import Database
from src.ap.image_scores import ImageScores
import os, statistics

do_score = True
do_analyse = True

args = {
    # Where are the images?
    'directory':r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output\compare2",

    # How strongly to prefer images that have been shown less. Values 0-0.9999 
    # 0 = totally random, 0.999 = very very strong preference
    'low_count_weight' : 0.4,   

    # Height of the window on your screen  
    'ab_scorer_size' : 800,

    # maximum aspect ratio of the images (width/height)
    'ab_max_width_ratio' : 0.8,

    'show_scores_at_end' : False,
    'save_csv'  : True,

    # tell me how many images have fewer than this number of comparisons at the end
    'threshold' : 5,

    # How many images to offer
    'ab_image_count' : 2,

    # How many comparisons to do
    'max_comparisons' : 100,

    # Leave this as False
    'ignore_score_zero' : False,
}

class TheApp:
    def __init__(self, height, ratio, image_count, db:Database):
        self.app = customtkinter.CTk()
        self.app.title("H.A.S.")
        self.app.geometry(f"{height*ratio*image_count}x{height}")
        self.ab = [customtkinter.CTkLabel(self.app, text="") for _ in range(image_count)]
        for i, label in enumerate(self.ab):
            label.grid(row=0, column=2*i)
            if i: self.app.grid_columnconfigure(2*i-1, weight=1)

        self.db = db
        self.height = height
        self.image_count = image_count
        self.app.bind("<KeyRelease>", self.keyup)
        self.pick_images()
        
    def pick_images(self):
        ims = self.db.pick_images(self.image_count)
        for i, im in enumerate(ims):
            self.ab[i].configure(image = customtkinter.CTkImage(light_image=im, size=(int(self.height*im.width/im.height),self.height)))

    def keyup(self,k):
        k = k.char
        if (k=='q'): 
            self.db.report()
            self.app.quit()
        elif k in "123456789"[:self.image_count+1]: 
            if int(k)<=self.image_count: 
                self.db.choice_made(int(k)-1)
            self.pick_images()
        elif k=='r':
            self.db.report()
        if self.db.choices_made()==args['max_comparisons']*(self.image_count-1):
            self.db.report()
            self.app.quit()
        if self.db.choices_made() % 10 == 0:
            print(self.db.choices_made())

def main():
    assert os.path.exists(args['directory']), f"{args['directory']} not found"
    db = Database(args['directory'], args, low_count_weight=args['low_count_weight'], threshold=args['threshold'])
    print(f"{len(db)} images")

    TheApp(args['ab_scorer_size'], args['ab_max_width_ratio'], args['ab_image_count'], db).app.mainloop()
    results = db.sorted_list()
    if args['show_scores_at_end']: 
        for result in results:
            print("{:>30} : {:>6.3f} ({:>3} tests)".format(*result))
    if args['save_csv']:
        with open(os.path.join(args['directory'],'scores.csv'),'w') as f:
            for result in results:
                print("{:>30},{:>6.3f}".format(*result), file=f)

def analyse():
    database_scores:ImageScores = ImageScores.from_scorefile(args['directory'])
    named_results = []
    for thing in os.listdir(args['directory']):
        if os.path.isdir(os.path.join(args['directory'],thing)):
            scores = database_scores.scores(normalised=True, directory=thing)
            results = (len(scores),statistics.mean(scores),statistics.stdev(scores))
            named_results.append( (thing, *results) )
    named_results.sort(key=lambda a:a[2])
    for named_result in named_results:
        print("{:>60} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}".format(*named_result)) 
    
if __name__=='__main__':
    if do_score: main()
    if do_analyse: analyse()
