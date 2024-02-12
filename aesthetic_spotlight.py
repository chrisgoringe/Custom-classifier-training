from src.time_context import Timer
with Timer("python imports"):
    from src.ap.aesthetic_predictor import AestheticPredictor
    from renumics import spotlight
    from src.ap.image_scores import ImageScores
    from pandas import DataFrame
    import torch, json, os

class Args:
    top_level_image_directory = ""
    model = ""
    scorefile = ""
    splitfile = ""

def main():
    top_level_images = Args.top_level_image_directory

    with Timer('load models'):
        predictor = AestheticPredictor.from_pretrained(pretrained=Args.model)
        predictor.eval()

    with Timer("prepare data"):
        with torch.no_grad():
            df = DataFrame()
            database_scores = ImageScores.from_scorefile(top_level_images, Args.scorefile)
            predictor.precache(database_scores.image_files(fullpath=True))
            model_scores = ImageScores.from_evaluator(predictor.evaluate_file, database_scores.image_files(), top_level_images)
            splits = {}
            if Args.splitfile:
                split_path = os.path.join(top_level_images, Args.splitfile)
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