import customtkinter
from arguments import args
from src.ap.database import Database
import statistics

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
    db = Database(args['top_level_image_directory'])
    TheApp(args['ab_scorer_size'], db).app.mainloop()

def analyse():
    db = Database(args['top_level_image_directory'])
    for r in args['ab_analysis_regexes']:
        scores = db.scores_for_matching(r)
        if len(scores)>1:
            print("{:>20} : {:>3} images, score {:>6.3f} +/- {:>4.2f}".format(f"/{r}/",len(scores),statistics.mean(scores),statistics.stdev(scores)))

if __name__=='__main__':
    if args['ab_analysis_regexes']:
        analyse()
    else:
        main()