from src.ap.aesthetic_predictor import AestheticPredictor
from renumics import spotlight
from src.ap.image_scores import ImageScores
import torch, json, os, argparse

def parse_arguments():
    parser = argparse.ArgumentParser("Score a set of images by a series of AB comparisons")
    parser.add_argument('-d', '--directory', help="Top level directory", required=True)
    parser.add_argument('--scores', default="scores.json", help="Filename of scores file (default scores.json)")
    parser.add_argument('-m', '--model', default="", help="Model (if any) to load and run on images (default is not to load a model)")
    parser.add_argument('--model_scores', default="", help="Filename of model scores file (ignored if model is specified)")
    parser.add_argument('--split', default="split.json", help="Filename of split file (default split.json)")
    parser.add_argument('--save', default=None, help="Filename to save in (as csv, default None)")

    global Args
    Args, unknowns = parser.parse_known_args()
    if unknowns: print(f"\nIgnoring unknown argument(s) {unknowns}")

def main():
    database_scores = ImageScores.from_scorefile(top_level_directory=Args.directory, scorefilename=Args.scores)

    if database_scores.has_item('model_score'):
        print(f"Using model_score from {Args.scores}")
    else:
        if Args.model:
            print(f"Using {Args.model} to generate model_scores")
            with torch.no_grad():
                predictor = AestheticPredictor.from_pretrained(pretrained=os.path.join(Args.directory,Args.model), image_directory=Args.directory)
                predictor.eval()
                predictor.precache(database_scores.image_files(fullpath=True))
                database_scores.add_item('model_score', predictor.evaluate_file, fullpath=True, cast=float)
        elif Args.model_scores:
            print(f"Using model_scores from {Args.model_scores}")
            database_scores.add_item('model_score', ImageScores.from_scorefile(dir, Args.model_scores))
        else:
            print("Need to specify a model or a model scorefile")
            return

    splits = json.load(open(os.path.join(Args.directory, Args.split))) if Args.split else {}

    if splits: database_scores.add_item('split',splits)
    database_scores.add_item('error',lambda a:database_scores.element('score',a)-database_scores.element('model_score',a), cast=float)
    database_scores.sort(add_rank_column='rank')
    database_scores.sort(by='model_score', add_rank_column='model_rank', resort_after=True)
    database_scores.add_item('rank_error',lambda a:abs(database_scores.element('rank',a)-database_scores.element('model_rank',a)), cast=int)

    if Args.save: database_scores.save_as_scorefile(os.path.join(Args.directory, Args.save))
            
    spotlight.show(database_scores.dataframe)

if __name__=='__main__':
    parse_arguments()
    main()