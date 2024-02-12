import statistics, os
import scipy.stats
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.image_scores import ImageScores, get_ab
import torch
from src.ap.create_scorefiles import create_scorefiles

class Args:
    top_level_image_directory = r"training4"
    scorefile = "scores.json"
    splitfile = "split.json"
    test_split_only = True

    load_and_run_model = True
    model = r"training4\model.safetensors"
    save_model_score_and_errors = False
    save_model_scorefile = "model_scores.json"
    save_error_scorefile = "error_scores.json"

    load_model_scorefile = False
    model_scorefile = "model_scores.json"

    regexes = []

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
    if Args.test_split_only: print(f"\nAnalysing test split only")
    else: print("\mAnalysing all images")
    dir = Args.top_level_image_directory
    print(f"database scores from {Args.scorefile}")
    database_scores:ImageScores = ImageScores.from_scorefile(dir, Args.scorefile, splitfile=Args.splitfile, split='test')

    if Args.load_and_run_model:
        print(f"using {Args.model} to evaluate images")
        ap = AestheticPredictor.from_pretrained(pretrained=Args.model, image_directory=Args.top_level_image_directory)
        ap.eval()
        with torch.no_grad():
            ap.precache(database_scores.image_files(fullpath=True))
            model_scores:ImageScores = ImageScores.from_evaluator(ap.evaluate_file, database_scores.image_files(), dir)
    elif Args.load_model_scorefile:
        print(f"loading model scores from {Args.model_scorefile}")
        model_scores = ImageScores.from_scorefile(dir, Args.model_scorefile, splitfile=Args.splitfile, split=Args.analyse_split)
    else:
        model_scores = None

    compare("All", database_scores, model_scores)
    for r in Args.regexes: compare(f"/{r}/", database_scores, model_scores, match=r, regex=True )
    for d in [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]: compare(d, database_scores, model_scores, directory=d)

    if Args.save_model_score_and_errors and Args.load_and_run_model:
        create_scorefiles(ap, database_scores, Args.save_model_scorefile, Args.save_error_scorefile)
            
if __name__=='__main__':
    assert not (Args.load_and_run_model and Args.load_model_scorefile), "Can't do both load_and_run_model and load_model_scorefile"
    analyse()