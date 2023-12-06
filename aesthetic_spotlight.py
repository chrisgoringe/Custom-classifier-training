from src.time_context import Timer
with Timer("python imports"):
    from arguments import args, get_args
    from src.ap.clip import CLIP
    from src.ap.aesthetic_predictor import AestheticPredictor
    from renumics import spotlight
    from src.ap.image_scores import ImageScores
    from pandas import DataFrame

def main():
    get_args(aesthetic_model=True, show_training_args=False, show_args=False)
    top_level_images = args['top_level_image_directory']
    
    with Timer('load models'):
        clipper = CLIP.get_clip(pretrained=args['clip_model'], image_directory=top_level_images)
        predictor = AestheticPredictor(pretrained=args['load_model_path'], clipper=clipper)

    with Timer("prepare data"):
        df = DataFrame()#columns=['image','db_score','model_score','db_rank','model_rank'])
        database_scores = ImageScores.from_scorefile(top_level_images)
        model_scores = ImageScores.from_evaluator(predictor.evaluate_file, database_scores.image_files(), top_level_images)
        df['image'] = database_scores.image_files(fullpath=True)
        df['db_score'] = database_scores.scores()
        df['model_score'] = model_scores.scores()
        df['db_rank'] = database_scores.ranks()
        df['model_rank'] = model_scores.ranks()
        clipper.save_cache()

    spotlight.show(df)

if __name__=='__main__':
    main()