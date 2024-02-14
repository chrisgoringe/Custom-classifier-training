import statistics, os, argparse
import scipy.stats
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.image_scores import ImageScores, get_ab
import torch
from src.ap.create_scorefiles import create_scorefiles

def parse_arguments():
    parser = argparse.ArgumentParser("Score a set of images by a series of AB comparisons")
    parser.add_argument('-d', '--directory', help="Top level directory", required=True)
    parser.add_argument('--scores', default="scores.json", help="Filename of scores file (default scores.json)")
    parser.add_argument('-m', '--model', default="", help="Model (if any) to load and run on images")
    parser.add_argument('--model_scores', default="", help="Filename of model scores file (if not running a model)")
    parser.add_argument('--split', default="split.json", help="Filename of split file (default split.json)")
    parser.add_argument('--include_train_split', action="store_true", help="Include training split in analysis")
    parser.add_argument('--save_scores_and_errors', action="store_true", help="Save score and error files from running model")
    parser.add_argument('--save_model_scorefile', default="model_scores.json")
    parser.add_argument('--save_error_scorefile', default="error_scores.json")

    Args.namespace, unknowns = parser.parse_known_args()
    if unknowns: print(f"\nIgnoring unknown argument(s) {unknowns}")

class _Args(object):
    instance = None
    def __init__(self):
        self.namespace = None

    def __getattr__(self, attr):
        return getattr(self.namespace,attr)
    
_Args.instance = _Args.instance or _Args()
Args = _Args.instance

regexes = []    # Zero or more regexes (as strings to be compiled). The analysis will run on (subject to the test_split constraint)
                # - all files 
                # - for each subfolder, just the files in it
                # - for each regex, just the files whose path matches the regex

def compare(label:str, database_scores:ImageScores, model_scores:ImageScores, **kwargs):
    scores = database_scores.scores(normalised=False, **kwargs)
    dbranks = database_scores.scores(rankings=True, **kwargs)
    if len(scores)<2:
        print("{:>20} : {:>5} images, too few for statistics".format(label, len(scores)))
        return
    results = (len(scores),statistics.mean(scores),statistics.stdev(scores))
    if model_scores:    
        mscores = model_scores.scores(normalised=False, **kwargs)
        mdranks = model_scores.scores(rankings=True, **kwargs)
        spearman = scipy.stats.spearmanr(dbranks,mdranks)
        pearson = scipy.stats.pearsonr(scores,mscores)
        results += (statistics.mean(mscores),statistics.stdev(mscores),spearman.statistic, spearman.pvalue, pearson.statistic, pearson.pvalue)
        results += (100*get_ab(mscores, scores),)
        print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}, model score {:>6.3f} +/- {:>4.2f}, spearman {:>6.4f} (p={:>8.2}), pearson {:>6.4f} (p={:>8.2}), AB {:>6.2f}%".format(label,*results))
    else:
        print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}".format(label,*results))

def analyse():
    if not Args.include_train_split: print(f"\nAnalysing test split only")
    else: print("\mAnalysing all images")
    dir = Args.directory
    print(f"database scores from {Args.scores}")
    database_scores:ImageScores = ImageScores.from_scorefile(dir, Args.scores, splitfile=Args.split, split='test')

    if Args.model:
        print(f"using {Args.model} to evaluate images")
        ap = AestheticPredictor.from_pretrained(pretrained=os.path.join(Args.directory, Args.model), image_directory=Args.directory)
        ap.eval()
        with torch.no_grad():
            ap.precache(database_scores.image_files(fullpath=True))
            model_scores:ImageScores = ImageScores.from_evaluator(ap.evaluate_file, database_scores.image_files(), dir)
    elif Args.model_scores:
        print(f"loading model scores from {Args.model_scores}")
        model_scores = ImageScores.from_scorefile(dir, Args.model_scores, splitfile=Args.split, split=not Args.include_train_split)
    else:
        model_scores = None

    compare("All", database_scores, model_scores)
    for r in regexes: compare(f"/{r}/", database_scores, model_scores, match=r, regex=True )
    for d in [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]: compare(d, database_scores, model_scores, directory=d)

    if Args.save_scores_and_errors and Args.model:
        create_scorefiles(ap, database_scores, Args.save_model_scorefile, Args.save_error_scorefile)
            
if __name__=='__main__':
    parse_arguments()
    assert not (Args.model and Args.model_scores), "Can't do both load_and_run_model and load_model_scorefile"
    analyse()