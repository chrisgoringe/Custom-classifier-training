from arguments import args, get_args
from src.ap.database import Database
import statistics, os
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.clip import CLIP

def literal_to_regex(s:str):
    for char in '\$()*+.?[^{|':
        s = s.replace(char,"\\"+char)
    return s

def create_normaliser(all_scores):
    all_s = [float(a) for a in all_scores]
    mean = statistics.mean(all_s)
    stdev = statistics.stdev(all_s)
    return lambda a : float( (a-mean)/stdev )

def apply_normalise(scores, normaliser):
    return list(normaliser(a) for a in scores)

def analyse():
    get_args(aesthetic_analysis=True, aesthetic_model=True, show_training_args=False)
    dir = args['top_level_image_directory']
    db = Database(dir, args)
    db_norm = create_normaliser(db.scores_for_matching(''))
    regexes = [r for r in args['ab_analysis_regexes']] + [f"^{literal_to_regex(d)}" for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]

    if args['use_model_scores_for_stats']:
        assert args['load_model'], "Need to load a model if use_model_scores_for_stats is true"
        ap = AestheticPredictor(clipper=CLIP(image_directory=db.image_directory), 
                                pretrained=os.path.join(args['load_model'],'model.safetensors'), 
                                dropouts=args['aesthetic_model_dropouts'], 
                                relu=args['aesthetic_model_relu'])
        db.set_model_score(ap.evaluate_file)
        model_norm = create_normaliser(db.model_scores_for_matching(''))

    for r in regexes:
        scores = apply_normalise(db.scores_for_matching(r), db_norm)
        if len(scores)>1:
            results = (len(scores),statistics.mean(scores),statistics.stdev(scores))
        if args['use_model_scores_for_stats']:    
            scores = apply_normalise(db.model_scores_for_matching(r), model_norm)
            if len(scores)>1:
                results += (statistics.mean(scores),statistics.stdev(scores))
            print("{:>20} : {:>3} images, db score {:>6.3f} +/- {:>4.2f}, model score {:>6.3f} +/- {:>4.2f}".format(f"/{r}/",*results))
        else:
            print("{:>20} : {:>3} images, db score {:>6.3f} +/- {:>4.2f}".format(f"/{r}/",*results))

            
            
if __name__=='__main__':
    analyse()