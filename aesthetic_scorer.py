from arguments import args, get_args
from src.ap.database import Database
import os
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.clip import CLIP

def print_image_scores():
    get_args(aesthetic_analysis=True, aesthetic_model=True)
    db = Database(args['top_level_image_directory'], args)

    assert args['load_model'], "Need to load a model if use_model_scores_for_stats is true"
    clipper = CLIP.get_clip(pretrained=args['clip_model'], device="cuda", image_directory=db.image_directory)
    clipper.precache(list(os.path.join(args['top_level_image_directory'],f) for f in db.image_scores))
    ap = AestheticPredictor(clipper=clipper, pretrained=args['load_model_path'], input_size=args['input_size'])
    db.set_model_score(ap.evaluate_file)
    clipper.save_cache()
    model_scores = sorted((ap.scale(db.model_scores[f]), f) for f in db.model_scores)

    with open(os.path.join(args['top_level_image_directory'], "model_scores.csv"),'w') as f:
        for score, filename in model_scores:
            print(f"{filename},{score}",file=f)    

if __name__=='__main__':
    print_image_scores()