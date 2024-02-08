from .aesthetic_predictor import AestheticPredictor
from .image_scores import ImageScores
import os

def create_scorefiles(ap:AestheticPredictor, database_scores:ImageScores, model_scorefile:str, error_scorefile:str):
    image_list = database_scores.image_files()
    tld = database_scores.top_level_directory
    model_scores:ImageScores = ImageScores.from_evaluator(ap.evaluate_file, image_list, top_level_directory=tld, fullpath=True)
    if model_scorefile: model_scores.save_as_scorefile(os.path.join(tld, model_scorefile))

    error = lambda f : abs(database_scores.score(f)-model_scores.score(f))
    error_scores = ImageScores.from_evaluator(error, image_list, top_level_directory=None, fullpath=False)
    if error_scorefile: error_scores.save_as_scorefile(os.path.join(tld, error_scorefile))