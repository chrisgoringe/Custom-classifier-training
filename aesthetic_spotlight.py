
from renumics import spotlight
from src.ap.image_scores import ImageScores
from src.comment_argument_parser import CommentArgumentParser

def parse_arguments():
    parser = CommentArgumentParser("Score a set of images by a series of AB comparisons", fromfile_prefix_chars='@')
    parser.add_argument('-d', '--directory', help="Top level directory", required=True)
    parser.add_argument('--scores', default="scores.csv", help="Filename of scores file (default scores.csv)")

    global Args
    Args, unknowns = parser.parse_known_args()
    if unknowns: print(f"\nIgnoring unknown argument(s) {unknowns}")

def main():
    database_scores = ImageScores.from_scorefile(top_level_directory=Args.directory, scorefilename=Args.scores)
    database_scores.sort(add_rank_column='rank')
    
    if not database_scores.has_item('model_score'):
        print(f"{Args.scores} doesn't contain model_scores column")
    else:
        database_scores.add_item('error',lambda a:database_scores.element('score',a)-database_scores.element('model_score',a), cast=float)
        database_scores.sort(by='model_score', add_rank_column='model_rank', resort_after=True)
        database_scores.add_item('rank_error',lambda a:abs(database_scores.element('rank',a)-database_scores.element('model_rank',a)), cast=int)
            
    spotlight.show(database_scores.dataframe)

if __name__=='__main__':
    parse_arguments()
    main()