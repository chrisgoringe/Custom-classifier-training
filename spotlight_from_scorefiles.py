from src.ap.image_scores import ImageScores
from pandas import DataFrame
from renumics import spotlight

def display(directory, sources):
    df = DataFrame()
    files = None
    for name, filename in sources:
        scores = ImageScores.from_scorefile(directory, filename)
        scores.set_rankings()
        if not files:
            files = list(f for f in scores.image_scores)
            df['files'] = files
        df[name] = list(scores.image_scores[f] for f in files)
        df[name+" rank"] = scores.ranks()

    spotlight.show(df)

if __name__=='__main__':
    display(directory="training4", sources=(
        ('split', 'split.json'),
        ('true value', 'scores.json'),
        ('t4 model score', 'model_scores.json'),
        ('t4 errors', 'error_scores.json'),
        ('t1 model score', 'scores_from_training1_model.json'),
        ('t1 errors', 'errors_from_training1_model.json'),
    ))