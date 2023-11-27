from arguments import args, get_args
from src.ap.database import Database
import os, statistics, math
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.clip import CLIP

def print_image_scores():
    get_args(aesthetic_analysis=True, aesthetic_model=True)
    db = Database(args['top_level_image_directory'], args)

    assert args['load_model'], "Need to load a model if use_model_scores_for_stats is true"
    clipper = CLIP(image_directory=db.image_directory)
    ap = AestheticPredictor(clipper=clipper, 
                            pretrained=os.path.join(args['load_model'],'model.safetensors'), 
                            dropouts=args['aesthetic_model_dropouts'], 
                            relu=args['aesthetic_model_relu'])
    db.set_model_score(ap.evaluate_file)
    clipper.save_cache()
    model_scores = sorted((ap.scale(db.model_scores[f]), f) for f in db.model_scores)

    for score, filename in model_scores:
        print("{:>70} : model {:>6.3f} ; db {:>6.3f}".format(filename[-70:], score, db.scale(db.image_scores[filename])))    

if __name__=='__main__':
    print_image_scores()