import customtkinter
from arguments import args, get_args
from src.ap.database import Database

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
    get_args(aesthetic_ab=True)
    db = Database(args['top_level_image_directory'])
    TheApp(args['ab_scorer_size'], db).app.mainloop()

if __name__=='__main__':
    main()