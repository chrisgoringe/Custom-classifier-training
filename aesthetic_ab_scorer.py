import customtkinter
from arguments import args, get_args
from src.ap.database import Database
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.clip import CLIP
from src.time_context import Timer
import os

class TheApp:
    def __init__(self, h, db:Database):
        self.app = customtkinter.CTk()
        self.app.title("H.A.S.")
        self.app.geometry(f"{h*2}x{h}")
        self.a = customtkinter.CTkLabel(self.app, text="")
        self.b = customtkinter.CTkLabel(self.app, text="")
        self.app.grid_columnconfigure(1, weight=1)
        self.a.grid(row=0, column=0)
        self.b.grid(row=0, column=2, sticky="e")
        self.db = db
        self.h = h
        self.app.bind("<KeyRelease>", self.keyup)
        self.pick_images()
        
    def pick_images(self):
        im1, im2 = self.db.pick_images_and_scores()
        self.a.configure(image = customtkinter.CTkImage(light_image=im1, size=(int(self.h*im1.width/im1.height),self.h)))
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
        if self.db.choices_made()==args['max_comparisons']:
            self.db.report()
            self.app.quit()
        if self.db.choices_made() % 10 == 0:
            print(self.db.choices_made())

def main():
    get_args(aesthetic_ab=True, aesthetic_model=True, show_training_args=False)
    db = Database(args['top_level_image_directory'], args)
    if args['use_model_scores_for_stats']:
        assert args['load_model']
        with Timer("Evaluate with model"):
            ap = AestheticPredictor(clipper=CLIP(image_directory=db.image_directory), 
                                    pretrained=os.path.join(args['load_model'],'model.safetensors'), 
                                    dropouts=args['aesthetic_model_dropouts'], 
                                    relu=args['aesthetic_model_relu'])
            db.set_model_score(ap.evaluate_file)

    TheApp(args['ab_scorer_size'], db).app.mainloop()

if __name__=='__main__':
    main()