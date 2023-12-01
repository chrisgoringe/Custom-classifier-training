from arguments import args, get_args
import statistics, os
import scipy.stats
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.clip import CLIP
from src.ap.image_scores import ImageScores

def analyse():
    get_args(aesthetic_analysis=True, aesthetic_model=True, show_training_args=False, show_args=False)
    dir = args['top_level_image_directory']
    database_scores:ImageScores = ImageScores.from_scorefile(dir)

    regexes = ['',] + [r for r in args['ab_analysis_regexes']] + [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]

    if args['use_model_scores_for_stats']:
        assert args['load_model'], "Need to load a model if use_model_scores_for_stats is true"
        ap = AestheticPredictor(clipper=CLIP(image_directory=args['top_level_image_directory']), 
                                pretrained=args['load_model_path'], 
                                dropouts=args['aesthetic_model_dropouts'])
        model_scores:ImageScores = ImageScores.from_evaluator(ap.evaluate_file, database_scores.image_files(), database_scores.image_directory)
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
            scores = model_scores.scores(r, regex=(r in args['ab_analysis_regexes']), normalised=True)
            mdranks = model_scores.scores(r, regex=(r in args['ab_analysis_regexes']), rankings=True)
            spearman = scipy.stats.spearmanr(dbranks,mdranks)
            results += (statistics.mean(scores),statistics.stdev(scores),spearman.statistic, spearman.pvalue)
            print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}, model score {:>6.3f} +/- {:>4.2f}, spearman {:>6.4f} (p={:>8.2})".format(r,*results))
        else:
            print("{:>20} : {:>5} images, db score {:>6.3f} +/- {:>4.2f}".format(r,*results))         
            
if __name__=='__main__':
    analyse()