from src.ap.image_scores import ImageScores
import os
from src.ap.aesthetic_predictor import AestheticPredictor

class Args:
    directory = ""
    load_model = ""

def print_image_scores():
    image_score_file = ImageScores.from_directory(Args.directory)

    ap = AestheticPredictor.from_pretrained(pretrained=Args.load_model)
    ap.precache(image_score_file.image_files(fullpath=True))

    image_score_file.set_scores(ap.evaluate_file)
    unsorted_scores = image_score_file.scores_dictionary()
    model_scores = sorted((unsorted_scores[f], f) for f in unsorted_scores)

    with open(os.path.join(Args.directory, "model_scores.csv"),'w') as f:
        for score, filename in model_scores:
            print(f"{filename},{score}",file=f)     

if __name__=='__main__':
    print_image_scores()