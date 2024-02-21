from src.ap.image_scores import ImageScores
import os, sys
from src.ap.aesthetic_predictor import AestheticPredictor
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser("Score a set of images by a series of AB comparisons")
    parser.add_argument('-d', '--directory', help="Top level directory", required=True)
    parser.add_argument('-m', '--model', help="Path to model", required=True)
    parser.add_argument('-o', '--outfile', default="", help="Save to file (.csv) in directory")

    global Args
    Args, unknowns = parser.parse_known_args()
    if unknowns: print(f"\nIgnoring unknown argument(s) {unknowns}")

def print_image_scores():
    image_score_file = ImageScores.from_directory(Args.directory)
    ap = AestheticPredictor.from_pretrained(pretrained=Args.model)
    ap.precache(image_score_file.image_files(fullpath=True))
    image_score_file.set_scores(ap.evaluate_file)
    
    unsorted_scores = image_score_file.scores_dictionary()
    model_scores = sorted((unsorted_scores[f], f) for f in unsorted_scores)

    f = open(os.path.join(Args.directory, Args.outfile),'w') if Args.outfile else sys.stdout
    try:
        print("relative_path,score", file=f)
        for score, filename in model_scores:
            print(f"{filename},{score}",file=f)     
    finally:
        if Args.outfile: f.close()

if __name__=='__main__':
    parse_arguments()
    print_image_scores()