from src.time_context import Timer
with Timer("python imports"):
    from arguments import args, get_args
    from src.ap.feature_extractor import FeatureExtractor
    from src.ap.aesthetic_predictor import AestheticPredictor
    from renumics import spotlight
    from src.ap.image_scores import ImageScores
    from pandas import DataFrame
    import torch, json, os

def main():
    get_args(aesthetic_model=True, show_training_args=False, show_args=False)
    top_level_images = args['top_level_image_directory']
    assert args["load_model_path"], "Need to specify load_model (the directory where the model is located, or the safetensors file itself)"
    
    with Timer('load models'):
        feature_extractor = FeatureExtractor.get_feature_extractor(pretrained=args['clip_model'], image_directory=top_level_images)
        predictor = AestheticPredictor(pretrained=args['load_model_path'], feature_extractor=feature_extractor, input_size=feature_extractor.number_of_features)
        predictor.eval()

    with Timer("prepare data"):
        with torch.no_grad():
            df = DataFrame()
            database_scores = ImageScores.from_scorefile(top_level_images, args['scorefile'])
            feature_extractor.precache(database_scores.image_files(fullpath=True))
            model_scores = ImageScores.from_evaluator(predictor.evaluate_file, database_scores.image_files(), top_level_images)
            splits = {}
            if 'splitfile' in args and args['splitfile']:
                split_path = os.path.join(top_level_images, args['splitfile'])
                if os.path.exists(split_path):
                    splits = json.load(open(split_path))

            df['image'] = database_scores.image_files(fullpath=True)
            df['split'] = list(splits.get(os.path.relpath(f,top_level_images),"") for f in df['image'])
            df['db_score'] = database_scores.scores()
            df['model_score'] = model_scores.scores()
            df['error'] = list(abs(x) for x in df['db_score']-df['model_score'])
            df['db_rank'] = database_scores.ranks()
            df['model_rank'] = model_scores.ranks()
            
    spotlight.show(df)

if __name__=='__main__':
    main()