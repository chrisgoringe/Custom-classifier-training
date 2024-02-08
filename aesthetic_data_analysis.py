from arguments import args, get_args
import statistics, os
import scipy.stats
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.feature_extractor import FeatureExtractor
from src.ap.image_scores import ImageScores
import torch

def get_ab_score(scores, true_scores):
    right = 0
    wrong = 0
    for i in range(len(scores)):
        a = (scores[i], true_scores[i])
        for j in range(i+1,len(scores)):
            b = (scores[j], true_scores[j])
            if a[0]==b[0] or a[1]==b[1]: continue
            if (a[0]<b[0] and a[1]<b[1]) or (a[0]>b[0] and a[1]>b[1]): right += 1
            else: wrong += 1
            
    return right/(right+wrong) if (right+wrong) else 0
        
def get_rmse(scores, true_scores):
    loss_fn = torch.nn.MSELoss()
    rmse = loss_fn(torch.tensor(scores), torch.tensor(true_scores))
    return float(rmse)

def analyse():
    get_args(aesthetic_analysis=True, aesthetic_model=True, show_training_args=False, show_args=False)
    dir = args['top_level_image_directory']
    database_scores:ImageScores = ImageScores.from_scorefile(dir, args['scorefile'])

    regexes = ['',] + [r for r in args['ab_analysis_regexes']] + [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]

    if args['load_model_path']:
        feature_extractor=FeatureExtractor.get_feature_extractor(image_directory=dir, pretrained=args['clip_model'])
        ap = AestheticPredictor(feature_extractor=feature_extractor, pretrained=args['load_model_path'])
        ap.eval()
        with torch.no_grad():
            feature_extractor.precache(database_scores.image_files(fullpath=True))
            model_scores:ImageScores = ImageScores.from_evaluator(ap.evaluate_file, database_scores.image_files(), dir)
            model_sigma:ImageScores = ImageScores.from_evaluator(ap.evaluate_file_sigma, database_scores.image_files(), dir)
        with open("scores_and_sigmas.csv",'w') as fhdl:
            print(f"file, true score, model score, model sigma", file=fhdl)
            for f in database_scores.image_files():
                print(f"{f},{database_scores.score(f)},{model_scores.score(f)},{model_sigma.score(f)}", file=fhdl)
    else:
        model_scores = None

    for r in regexes:
        scores = database_scores.scores(r, regex=(r in args['ab_analysis_regexes']), normalised=False)
        dbranks = database_scores.scores(r, regex=(r in args['ab_analysis_regexes']), rankings=True)
        if len(scores)<2:
            print("{:>20} : too few matches")
            continue
        results = (len(scores),statistics.mean(scores),statistics.stdev(scores))
        if model_scores:    
            mscores = model_scores.scores(r, regex=(r in args['ab_analysis_regexes']), normalised=False)
            mdranks = model_scores.scores(r, regex=(r in args['ab_analysis_regexes']), rankings=True)
            spearman = scipy.stats.spearmanr(dbranks,mdranks)
            pearson = scipy.stats.pearsonr(scores,mscores)
            results += (statistics.mean(mscores),statistics.stdev(mscores),spearman.statistic, spearman.pvalue, pearson.statistic, pearson.pvalue)
            results += (100*get_ab_score(mscores, scores),)
            print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}, model score {:>6.3f} +/- {:>4.2f}, spearman {:>6.4f} (p={:>8.2}), pearson {:>6.4f} (p={:>8.2}), AB {:>6.2f}%".format(r,*results))
        else:
            print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}".format(r,*results))

        
            
if __name__=='__main__':
    analyse()