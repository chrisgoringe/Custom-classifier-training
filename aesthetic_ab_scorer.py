import customtkinter
from arguments import args, get_args
from src.ap.database import Database
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.clip import CLIP
from src.time_context import Timer
import os

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
        if self.db.choices_made()==args['max_comparisons']:
            self.db.report()
            self.app.quit()
        if self.db.choices_made() % 10 == 0:
            print(self.db.choices_made())

def main():
    get_args(aesthetic_ab=True, aesthetic_model=True, show_training_args=False)
    assert os.path.exists(args['top_level_image_directory']), f"{args['top_level_image_directory']} not found"
    db = Database(args['top_level_image_directory'], args)
    if args['use_model_scores_for_stats']:
        assert args['load_model']
        with Timer("Evaluate with model"):
            ap = AestheticPredictor(clipper=CLIP(image_directory=db.image_directory), 
                                    pretrained=os.path.join(args['load_model'],'model.safetensors'), 
                                    relu=args['aesthetic_model_relu'])
            db.set_model_score(ap.evaluate_file)

    TheApp(args['ab_scorer_size'], args['ab_max_width_ratio'], args['ab_image_count'], db).app.mainloop()

if __name__=='__main__':
    main()