import statistics, os, argparse, json, re
import scipy.stats
#from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.image_scores import ImageScores
#import torch
from src.ap.create_scorefiles import create_scorefiles

def parse_arguments():
    parser = argparse.ArgumentParser("Score a set of images by a series of AB comparisons")
    parser.add_argument('-d', '--directory', help="Top level directory", required=True)
    parser.add_argument('--scores', default="scores.json", help="Filename of scores file (default scores.json)")
    parser.add_argument('-m', '--model', default="", help="Model (if any) to load and run on images (default is not to load a model)")
    parser.add_argument('--model_scores', default="", help="Filename of model scores file (ignored if model is specified)")
    parser.add_argument('--split', default="split.json", help="Filename of split file (default split.json)")
    parser.add_argument('--include_train_split', action="store_true", help="Include training split in analysis (default is eval images only)")
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

regexes = []    # Zero or more regexes (as strings to be compiled). The analysis will run on (subject to the eval constraint)
                # - all files 
                # - for each subfolder, just the files in it
                # - for each regex, just the files whose path matches the regex

def get_ab(scores, predicted_scores):
    right = 0
    total = 0
    for i in range(len(scores)):
        for j in range(i+1,len(scores)):
            if (predicted_scores[i]<predicted_scores[j] and scores[i]<scores[j]) or \
                (predicted_scores[i]>predicted_scores[j] and scores[i]>scores[j]): right += 1
            total += 1
    return right/total if total else 0

def compare(label:str, scores:list, model_scores:list):
    if len(scores)<2:
        print("{:>20} : {:>5} images, too few for statistics".format(label, len(scores)))
        return
    results = (len(scores),statistics.mean(scores),statistics.stdev(scores))
    if model_scores:    
        spearman = scipy.stats.spearmanr(scores,model_scores)
        pearson = scipy.stats.pearsonr(scores,model_scores)
        results += (statistics.mean(model_scores),statistics.stdev(model_scores),spearman.statistic, spearman.pvalue, pearson.statistic, pearson.pvalue)
        results += (100*get_ab(model_scores, scores),)
        print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}, model score {:>6.3f} +/- {:>4.2f}, spearman {:>6.4f} (p={:>8.2}), pearson {:>6.4f} (p={:>8.2}), AB {:>6.2f}%".format(label,*results))
    else:
        print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}".format(label,*results))

def analyse():
    dir = Args.directory
    print(f"database scores from {Args.scores}")
    database_scores:ImageScores = ImageScores.from_scorefile(dir, Args.scores)
    if not Args.include_train_split:
        print(f"\nAnalysing eval split only")
        if Args.split:
            with open(os.path.join(Args.directory, Args.split), 'r') as fhdl:
                database_scores.add_item('split', json.load(fhdl))
        database_scores = database_scores.subset(item='split',test=lambda a:a=='eval')
    else:
        print("\mAnalysing all images")
    
    if Args.model:
        print(f"using {Args.model} to evaluate images")
        ap = AestheticPredictor.from_pretrained(pretrained=os.path.join(Args.directory, Args.model), image_directory=Args.directory)
        ap.eval()
        ap.precache(database_scores.image_files(fullpath=True))
        database_scores.add_item('model_score', ap.evaluate_file)
    elif Args.model_scores:
        print(f"loading model scores from {Args.model_scores}")
        database_scores.add_item('model_score', ImageScores.from_scorefile(dir, Args.model_scores))

    compare("All", database_scores.scores(), database_scores.item('model_score'))
    for r in regexes: 
        reg = re.compile(r)
        sub = database_scores.subset(lambda a : reg.match(a))
        compare(f"/{r}/", sub.scores(), sub.item('model_scores'))
    for d in [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]: 
        sub = database_scores.subset(lambda a : a.startswith(d+os.pathsep))
        compare(d, sub.scores(), sub.item('model_scores'))

    if Args.save_scores_and_errors and Args.model:
        create_scorefiles(ap, database_scores, Args.save_model_scorefile, Args.save_error_scorefile)
            
if __name__=='__main__':
    parse_arguments()
    assert not (Args.model and Args.model_scores), "Can't do both load_and_run_model and load_model_scorefile"
    analyse()