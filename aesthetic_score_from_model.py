from arguments import args, get_args

from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.image_scores import ImageScores
from src.ap.feature_extractor import FeatureExtractor
from src.time_context import Timer

import os, statistics

def main():
    get_args(aesthetic_model=True, show_training_args=False)
    assert args['top_level_image_directory'], "Need an image directory"
    assert args['load_model'], "Need to load a model"

    with Timer("Load database and models"):
        image_score_file = ImageScores.from_scorefile(args['top_level_image_directory'], args['scorefile'])
        image_scores = image_score_file.scores_dictionary()

        feature_extractor = FeatureExtractor.get_feature_extractor(image_directory=args['top_level_image_directory'], pretrained=args['clip_model'])
        feature_extractor.precache([os.path.join(args['top_level_image_directory'],f) for f in image_scores])
        predictor = AestheticPredictor(pretrained=args['load_model_path'], feature_extractor=feature_extractor)
        
    with Timer("Predict scores for all images"):
        all_predicted_scores = predictor.evaluate_files([os.path.join(args['top_level_image_directory'],f) for f in image_scores], 
                                            as_sorted_tuple=True, eval_mode=True)
        all_predicted_scores = [(float(a[0]), os.path.relpath(a[1],args['top_level_image_directory'])) for a in all_predicted_scores ]
    
    with Timer("Analyse statistics") as logger:
        new_predictions = { a[1]: a[0] for a in all_predicted_scores if image_scores[a[1]]==0 }

        old_predictions = { a[1]: a[0] for a in all_predicted_scores if image_scores[a[1]]!=0 }
        mean_old_predictions = statistics.mean(old_predictions[a] for a in old_predictions) if old_predictions else 0
        std_old_predictions = statistics.stdev(old_predictions[a] for a in old_predictions) if len(old_predictions)>1 else 1

        old_db_scores = [image_scores[f] for f in image_scores if image_scores[f]!=0]
        mean_db = statistics.mean(old_db_scores) if old_db_scores else 0
        std_db = statistics.stdev(old_db_scores) if len(old_db_scores) else 1
        
        prediction_to_db = lambda a : (std_db*(a - mean_old_predictions)/std_old_predictions) + mean_db
    
        logger("For existing images, model score mean {:>6.3f} stdev {:>6.3f}, database score mean {:>6.3f} stdev {:>6.3f}".format(
            mean_old_predictions, std_old_predictions, mean_db, std_db))
        
        logger("Scaling of model score a to database score is therefore lambda a : ({:>6.3f} * (a -{:>6.3f}) /{:>6.3f}) +{:>6.3f}".format(
            std_db, mean_old_predictions, std_old_predictions, mean_db ))

    with Timer("Set scores for unscored images") as logger:

        for f in new_predictions:
            image_scores[f] = prediction_to_db(new_predictions[f])
        logger(f"Added scores for {len(new_predictions)} unscored images")

    with Timer("Save updated database"):
        savefile = os.path.splitext(args['scorefile'])[0]+"_new.json"
        image_score_file.save_as_scorefile(os.path.join(args['top_level_image_directory'], savefile))

if __name__=='__main__':
    with Timer("Main"): main()
