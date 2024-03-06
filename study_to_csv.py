import argparse
from pandas import DataFrame
import optuna

def parse_arguments():
    parser = argparse.ArgumentParser("Save study as csv")
    parser.add_argument('--database', default="sqlite:///db.sqlite")
    parser.add_argument('--study_name', default=None, help="Default is the most recent study; --study_name=list to show available studies.")
    parser.add_argument('--outfile', default=None)

    Args, unknowns = parser.parse_known_args()
    if unknowns: print(f"\nIgnoring unknown argument(s) {unknowns}")
    return Args

def main():
    Args = parse_arguments()
    study_name = Args.study_name or optuna.study.get_all_study_names(storage=Args.database)[-1]
    if Args.study_name=='list':
        print("\n".join(optuna.study.get_all_study_names(storage=Args.database)))
    else:
        if Args.outfile:
            study:optuna.Study = optuna.study.load_study(study_name=study_name, storage=Args.database)
            df:DataFrame = study.trials_dataframe(attrs=('number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'params', 'user_attrs'),)
            df.to_csv(Args.outfile)
        else:
            print("--outfile required")

if __name__=='__main__':
    main()
