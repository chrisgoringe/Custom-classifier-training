from arguments import args, get_args
from src.ap.database import Database
import statistics, os
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.clip import CLIP

def analyse():
    get_args(aesthetic_analysis=True, aesthetic_model=True)
    db = Database(args['top_level_image_directory'], args)
    if args['use_model_scores_for_stats']:
        assert args['load_model'], "Need to load a model if use_model_scores_for_stats is true"
        ap = AestheticPredictor(clipper=CLIP(image_directory=db.image_directory), 
                                pretrained=os.path.join(args['load_model'],'model.safetensors'), 
                                dropouts=args['aesthetic_model_dropouts'], 
                                relu=args['aesthetic_model_relu'])
        db.set_model_score(ap.evaluate_file)

    for r in args['ab_analysis_regexes']:
        scores = db.scores_for_matching(r)
        if len(scores)>1:
            print("{:>20} : {:>3} images,    db score {:>6.3f} +/- {:>4.2f}".format(f"/{r}/",len(scores),statistics.mean(scores),statistics.stdev(scores)))

    if args['use_model_scores_for_stats']:
        for r in args['ab_analysis_regexes']:
            scores = db.model_scores_for_matching(r)
            if len(scores)>1:
                print("{:>20} : {:>3} images, model score {:>6.3f} +/- {:>4.2f}".format(f"/{r}/",len(scores),statistics.mean(scores),statistics.stdev(scores)))

if __name__=='__main__':
    analyse()