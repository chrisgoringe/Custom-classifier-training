from src.ap.aesthetic_predictor import AestheticPredictor
from renumics import spotlight
from src.ap.image_scores import ImageScores
from pandas import DataFrame
import torch, json, os, argparse

Args = None

def parse_arguments():
    parser = argparse.ArgumentParser("Score a set of images by a series of AB comparisons")
    parser.add_argument('-d', '--directory', help="Top level directory", required=True)
    parser.add_argument('--scores', default="scores.json", help="Filename of scores file (default scores.json)")
    parser.add_argument('-m', '--model', default="", help="Model (if any) to load and run on images (default is not to load a model)")
    parser.add_argument('--model_scores', default="", help="Filename of model scores file (ignored if model is specified)")
    parser.add_argument('--split', default="split.json", help="Filename of split file (default split.json)")

    global Args
    Args, unknowns = parser.parse_known_args()
    if unknowns: print(f"\nIgnoring unknown argument(s) {unknowns}")

def main():
    database_scores = ImageScores.from_scorefile(top_level_directory=Args.directory, scorefilename=Args.scores)
    
    if Args.model:
        with torch.no_grad():
            predictor = AestheticPredictor.from_pretrained(pretrained=os.path.join(Args.directory,Args.model), image_directory=Args.directory)
            predictor.eval()
            predictor.precache(database_scores.image_files(fullpath=True))
            model_scores = ImageScores.from_evaluator(predictor.evaluate_file, database_scores.image_files(), Args.directory)
    elif Args.model_scores:
        model_scores = ImageScores.from_scorefile(Args.directory, Args.model_scores)
    else:
        print("Need to specify a model or a model scorefile")
        return

    splits = json.load(open(os.path.join(Args.directory, Args.split))) if Args.split else {}

    df = DataFrame()
    df['image'] = database_scores.image_files(fullpath=True)
    if splits: df['split'] = list(splits.get(os.path.relpath(f,Args.directory),"") for f in df['image'])
    df['db_score'] = database_scores.scores()
    df['model_score'] = model_scores.scores()
    df['error'] = list(abs(x) for x in df['db_score']-df['model_score'])
    df['db_rank'] = database_scores.ranks()
    df['model_rank'] = model_scores.ranks()
    df['rank error'] = list(abs(x) for x in df['db_rank']-df['model_rank'])
            
    spotlight.show(df)

if __name__=='__main__':
    parse_arguments()
    main()