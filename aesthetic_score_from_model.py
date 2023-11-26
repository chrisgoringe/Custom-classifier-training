from arguments import args, get_args

from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.database import Database
from src.ap.clip import CLIP
from src.time_context import Timer

import os, statistics

def main():
    get_args(aesthetic_model=True)
    assert args['top_level_image_directory'], "Need an image directory"
    assert args['load_model'], "Need to load a model"

    with Timer("Load database and models"):
        db = Database(args['top_level_image_directory'], args)

        clipper = CLIP(image_directory=args['top_level_image_directory'])
        predictor = AestheticPredictor(pretrained=os.path.join(args['load_model'],"model.safetensors"), 
                                    relu=args.get('aesthetic_model_relu',True), clipper=clipper)
        
    with Timer("Predict scores for all images"):
        all_predicted_scores = predictor.evaluate_files([os.path.join(args['top_level_image_directory'],f) for f in db.image_scores], 
                                            as_sorted_tuple=True, eval_mode=True)
        all_predicted_scores = [(float(a[0]), os.path.relpath(a[1],args['top_level_image_directory'])) for a in all_predicted_scores ]
        clipper.save_cache()
    
    with Timer("Analyse statistics") as logger:
        new_predictions = { a[1]: a[0] for a in all_predicted_scores if db.image_scores[a[1]]==0 }
        old_predictions = { a[1]: a[0] for a in all_predicted_scores if db.image_scores[a[1]]!=0 }
        assert len(new_predictions)>1 and len(old_predictions)>1 , "Need at least two old images and two new images"
        mean_old_predictions = statistics.mean(old_predictions[a] for a in old_predictions)
        std_old_predictions = statistics.stdev(old_predictions[a] for a in old_predictions)

        old_db_scores = [db.image_scores[f] for f in db.image_scores if db.image_scores[f]!=0]
        std_db = statistics.stdev(old_db_scores)
        mean_db = statistics.mean(old_db_scores)

        prediction_to_db = lambda a : (std_db*(a - mean_old_predictions)/std_old_predictions) + mean_db
    
        logger("For existing images, model score mean {:>6.3f} stdev {:>6.3f}, database score mean {:>6.3f} stdev {:>6.3f}".format(
            mean_old_predictions, std_old_predictions, mean_db, std_db))
        
        logger("Scaling of model score a to database score is therefore lambda a : ({:>6.3f} * (a -{:>6.3f}) /{:>6.3f}) +{:>6.3f}".format(
            std_db, mean_old_predictions, std_old_predictions, mean_db ))

    with Timer("Set scores for unscored images") as logger:

        for f in new_predictions:
            db.image_scores[f] = prediction_to_db(new_predictions[f])
        logger(f"Added scores for {len(new_predictions)} unscored images")

    with Timer("Save updated database"):
        db.meta['model_evals'] = db.meta.get('model_evals',0) + len(new_predictions)
        db.save()

if __name__=='__main__':
    with Timer("Main"): main()
