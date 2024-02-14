from src.ap.aesthetic_predictor import AestheticPredictor
from renumics import spotlight
from src.ap.image_scores import ImageScores
from pandas import DataFrame
import torch, json, os

class Args:
    directory = "training4"
    model = r"training4/vitH_tiniest2_model.safetensors"
    scorefile = "scores.json"
    splitfile = "split.json"

def main():
    predictor = None
    if Args.model.endswith(".safetensors"):
        predictor = AestheticPredictor.from_pretrained(pretrained=Args.model, image_directory=Args.directory)
        predictor.eval()

    with torch.no_grad():
        df = DataFrame()
        database_scores = ImageScores.from_scorefile(top_level_directory=Args.directory, scorefilename=Args.scorefile)
        if predictor:
            predictor.precache(database_scores.image_files(fullpath=True))
            model_scores = ImageScores.from_evaluator(predictor.evaluate_file, database_scores.image_files(), Args.directory)
        elif Args.model.endswith(".json"):
            model_scores = ImageScores.from_scorefile(Args.directory, Args.model)
        else:
            print("Args.model should be a model.safetensors file or a score.json file")
            return
        splits = {}
        if Args.splitfile:
            split_path = os.path.join(Args.directory, Args.splitfile)
            if os.path.exists(split_path):
                splits = json.load(open(split_path))

        df['image'] = database_scores.image_files(fullpath=True)
        df['split'] = list(splits.get(os.path.relpath(f,Args.directory),"") for f in df['image'])
        df['db_score'] = database_scores.scores()
        df['model_score'] = model_scores.scores()
        df['error'] = list(abs(x) for x in df['db_score']-df['model_score'])
        df['db_rank'] = database_scores.ranks()
        df['model_rank'] = model_scores.ranks()
        df['rank error'] = list(abs(x) for x in df['db_rank']-df['model_rank'])
            
    spotlight.show(df)

if __name__=='__main__':
    main()