import argparse
from pandas import DataFrame
import optuna

def parse_arguments():
    parser = argparse.ArgumentParser("Save study as csv")
    parser.add_argument('--database', default="sqlite:///db.sqlite")
    parser.add_argument('--study_name', default=None)
    parser.add_argument('--outfile', default=None)

    Args, unknowns = parser.parse_known_args()
    if unknowns: print(f"\nIgnoring unknown argument(s) {unknowns}")
    return Args

def main():
    Args = parse_arguments()
    if Args.study_name and Args.outfile:
        study:optuna.Study = optuna.study.load_study(study_name=Args.study_name, storage=Args.database)
        df:DataFrame = study.trials_dataframe(attrs=('number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'params', 'user_attrs'),)
        df.to_csv(Args.outfile)
    else:
        print("--study_name and --outfile are both required. Available studies:")
        print("\n".join(optuna.study.get_all_study_names(storage=Args.database)))

if __name__=='__main__':
    main()
