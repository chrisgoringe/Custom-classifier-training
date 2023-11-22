from arguments import args, get_args
from src.ap.database import Database
import statistics

def analyse():
    get_args(aesthetic_analysis=True)
    db = Database(args['top_level_image_directory'])
    for r in args['ab_analysis_regexes']:
        scores = db.scores_for_matching(r)
        if len(scores)>1:
            print("{:>20} : {:>3} images, score {:>6.3f} +/- {:>4.2f}".format(f"/{r}/",len(scores),statistics.mean(scores),statistics.stdev(scores)))
