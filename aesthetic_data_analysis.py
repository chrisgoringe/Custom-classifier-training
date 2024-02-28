import statistics, os, argparse, re
import scipy.stats
from src.ap.image_scores import ImageScores
import torch


def parse_arguments():
    parser = argparse.ArgumentParser("Statistics of scored files")
    parser.add_argument('-d', '--directory', help="Top level directory", required=True)
    parser.add_argument('--scores', default="scores.csv", help="Filename of scores file (default scores.csv)")
    parser.add_argument('--include_train_split', action="store_true", help="Include training split in analysis (default is eval images only)")
    parser.add_argument('--regex', default="", help="Only include images matching this regex")
    parser.add_argument('--directories', action="store_true", help="Perform separate analysis for each subdirectory")

    global Args
    Args, unknowns = parser.parse_known_args()
    if unknowns: print(f"\nIgnoring unknown argument(s) {unknowns}")

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
    if model_scores is not None:    
        spearman = scipy.stats.spearmanr(scores,model_scores)
        pearson = scipy.stats.pearsonr(scores,model_scores)
        results += (statistics.mean(model_scores),statistics.stdev(model_scores),spearman.statistic, spearman.pvalue, pearson.statistic, pearson.pvalue)
        results += (100*get_ab(model_scores, scores),)
        print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}, model score {:>6.3f} +/- {:>4.2f}, spearman {:>6.4f} (p={:>8.2}), pearson {:>6.4f} (p={:>8.2}), AB {:>6.2f}%".format(label,*results))
    else:
        print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}".format(label,*results))

def analyse():
    database_scores:ImageScores = ImageScores.from_scorefile(Args.directory, Args.scores)
    
    if not Args.include_train_split:
        print(f"\nAnalysing eval split only")
        database_scores = database_scores.subset(item='split',test=lambda a:a=='eval')

    if Args.regex: 
        reg = re.compile(Args.regex)
        sub = database_scores.subset(lambda a : reg.match(a))
        compare(f"/{Args.regex}/", sub.item('score'), sub.item('model_scores'))
    else:
        reg = None
        compare("All", database_scores.item('score'), database_scores.item('model_score'))

    if Args.directories:
        for d in [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]: 
            sub = database_scores.subset(lambda a : a.startswith(d+os.pathsep))
            if reg: sub = sub.subset(lambda a : reg.match(a))
            compare(d+(f" /{Args.regex}/" if reg else ""), sub.item('score'), sub.item('model_scores'))
            
if __name__=='__main__':
    parse_arguments()
    with torch.no_grad():
        analyse()