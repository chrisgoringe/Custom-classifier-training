from arguments import args, get_args
from src.ap.database import Database
import os, statistics, math
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.clip import CLIP

def print_image_scores():
    get_args(aesthetic_analysis=True, aesthetic_model=True)
    db = Database(args['top_level_image_directory'], args)

    assert args['load_model'], "Need to load a model if use_model_scores_for_stats is true"
    ap = AestheticPredictor(clipper=CLIP(image_directory=db.image_directory), 
                            pretrained=os.path.join(args['load_model'],'model.safetensors'), 
                            dropouts=args['aesthetic_model_dropouts'], 
                            relu=args['aesthetic_model_relu'])
    db.set_model_score(ap.evaluate_file)
    model_scores = sorted((ap.scale(db.model_scores[f]), f) for f in db.model_scores)
    dbmean = statistics.mean(db.image_scores[f] for f in db.image_scores)
    dbstd = statistics.stdev(db.image_scores[f] for f in db.image_scores)
    scale = lambda a : (a-dbmean)/dbstd
    image_scores = {f:scale(db.image_scores[f]) for f in db.image_scores}

    for model_score, filename in model_scores:
        if abs(model_score - image_scores[filename])>1:
            print("{:>70} : model {:>5.3f} ; db {:>5.3f}".format(filename[-70:], model_score, image_scores[filename]))

if __name__=='__main__':
    print_image_scores()