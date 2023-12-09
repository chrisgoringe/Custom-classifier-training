import customtkinter
from src.ap.database import Database
import os

args = {
    # Where are the images?
    'top_level_image_directory':'PATH/GOES/HERE',

    # How strongly to prefer images that have been shown less. Values 0-0.9999 
    # 0 = totally random, 0.999 = very very strong preference
    'low_count_weight' : 0.4,   

    # Height of the window on your screen  
    'ab_scorer_size' : 600,

    # maximum aspect ratio of the images (width/height)
    'ab_max_width_ratio' : 1.0,

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
        elif k in "1234567"[:self.image_count]: 
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
    assert os.path.exists(args['top_level_image_directory']), f"{args['top_level_image_directory']} not found"
    db = Database(args['top_level_image_directory'], args, low_count_weight=args['low_count_weight'], controversy_weight=0)

    TheApp(args['ab_scorer_size'], args['ab_max_width_ratio'], args['ab_image_count'], db).app.mainloop()

if __name__=='__main__':
    main()