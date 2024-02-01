from arguments import args, get_args
import statistics, os
import scipy.stats
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.feature_extractor import FeatureExtractor
from src.ap.image_scores import ImageScores

def analyse():
    get_args(aesthetic_analysis=True, aesthetic_model=True, show_training_args=False, show_args=False)
    dir = args['top_level_image_directory']
    database_scores:ImageScores = ImageScores.from_scorefile(dir, args['scorefile'])

    regexes = ['',] + [r for r in args['ab_analysis_regexes']] + [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]

    if args['load_model_path']:
        ap = AestheticPredictor(feature_extractor=FeatureExtractor.get_feature_extractor(image_directory=dir, pretrained=args['clip_model']), 
                                pretrained=args['load_model_path'])
        model_scores:ImageScores = ImageScores.from_evaluator(ap.evaluate_file, database_scores.image_files(), dir)
    else:
        model_scores = None

    for r in regexes:
        scores = database_scores.scores(r, regex=(r in args['ab_analysis_regexes']), normalised=True)
        dbranks = database_scores.scores(r, regex=(r in args['ab_analysis_regexes']), rankings=True)
        if len(scores)<2:
            print("{:>20} : too few matches")
            continue
        results = (len(scores),statistics.mean(scores),statistics.stdev(scores))
        if model_scores:    
            mscores = model_scores.scores(r, regex=(r in args['ab_analysis_regexes']), normalised=True)
            mdranks = model_scores.scores(r, regex=(r in args['ab_analysis_regexes']), rankings=True)
            spearman = scipy.stats.spearmanr(dbranks,mdranks)
            pearson = scipy.stats.pearsonr(scores,mscores)
            results += (statistics.mean(mscores),statistics.stdev(mscores),spearman.statistic, spearman.pvalue, pearson.statistic, pearson.pvalue)
            print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}, model score {:>6.3f} +/- {:>4.2f}, spearman {:>6.4f} (p={:>8.2}), pearson {:>6.4f} (p={:>8.2})".format(r,*results))
        else:
            print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}".format(r,*results))         
            
if __name__=='__main__':
    analyse()