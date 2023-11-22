from arguments import args

from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.database import Database
from src.ap.clip import CLIP
from src.time_context import Timer

import os, statistics, math

def main():
    assert args['top_level_image_directory'], "Need an image directory"
    assert args['load_model'], "Need to load a model"

    with Timer("Load database and models"):
        db = Database(args['top_level_image_directory'])

        clipper = CLIP(image_directory=args['top_level_image_directory'])
        predictor = AestheticPredictor(pretrained=os.path.join(args['load_model'],"model.safetensors"), 
                                    relu=args['aesthetic_model_relu'], clipper=clipper)
    
    with Timer("Analyse existing non-zero scores") as logger:
        all_non_zero_scores = [db.image_scores[f] for f in db.image_scores if db.image_scores[f]!=0]
        if len(all_non_zero_scores)>1:
            std_existing = statistics.stdev(all_non_zero_scores)
            logger(f"Std dev. of {len(all_non_zero_scores)} existing scores = {std_existing}")
            logger(f"(mean of existing scores = {statistics.mean(all_non_zero_scores)})")
        else:
            logger("No images have been scored: setting std dev. to 0.5")
            std_existing = 0.5

    with Timer("Predict scores for all images"):
        all_predicted_scores = predictor.evaluate_files([os.path.join(args['top_level_image_directory'],f) for f in db.image_scores], 
                                            as_sorted_tuple=True, eval_mode=True)
        clipper.save_cache()
        
        all_predicted_scores = [(a[0], os.path.relpath(a[1],args['top_level_image_directory'])) for a in all_predicted_scores ]
        mean = statistics.mean([a[0] for a in all_predicted_scores])
    
    with Timer("Offset by mean and apply tanh") as logger:
        new_predictions = { a[1]: math.tanh(a[0]-mean) for a in all_predicted_scores if db.image_scores[a[1]]==0 }
        std_predictions = statistics.stdev([new_predictions[a] for a in new_predictions])
        mean_predictions = statistics.mean([new_predictions[a] for a in new_predictions])
        logger(f"Std dev. of prediction scores = {std_predictions}")
        logger(f"(mean of prediction scores = {mean_predictions})")

    with Timer("Set scores for unscored images") as logger:
        scaling = std_existing/std_predictions
        logger(f"scaling new predictions by {scaling}")
        for f in new_predictions:
            db.image_scores[f] = new_predictions[f]*scaling
        logger(f"Added scores for {len(new_predictions)} unscored images")

    with Timer("Save updated database"):
        db.meta['model_evals'] = db.meta.get('model_evals',0) + len(new_predictions)
        db.save()

if __name__=='__main__':
    with Timer("Main"): main()
