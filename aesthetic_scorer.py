from arguments import args, get_args
from src.ap.image_scores import ImageScores
import os
from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.feature_extractor import FeatureExtractor

def print_image_scores():
    get_args(aesthetic_analysis=True, aesthetic_model=True)
    image_score_file = ImageScores.from_directory(args['top_level_image_directory'])

    feature_extractor = FeatureExtractor.get_feature_extractor(pretrained=args['clip_model'], device="cuda", image_directory=args['top_level_image_directory'])
    feature_extractor.precache(image_score_file.image_files(fullpath=True))

    ap = AestheticPredictor(feature_extractor=feature_extractor, pretrained=args['load_model_path'])

    image_score_file.set_scores(ap.evaluate_file)
    unsorted_scores = image_score_file.scores_dictionary()
    model_scores = sorted((unsorted_scores[f], f) for f in unsorted_scores)

    with open(os.path.join(args['top_level_image_directory'], "model_scores.csv"),'w') as f:
        for score, filename in model_scores:
            print(f"{filename},{score}",file=f)     

if __name__=='__main__':
    print_image_scores()