from src.time_context import Timer

with Timer("Python imports"):
    import torch
    from safetensors.torch import save_file
    import math, random, os, threading, sys
    import cProfile as profile

    from transformers import TrainingArguments
    import optuna
    from optuna_dashboard import run_server

    from src.ap.dataset import QuickDataset

    from arguments import Args 
    from src.ap.feature_extractor import FeatureExtractor
    from src.ap.aesthetic_predictor import AestheticPredictor
    from src.ap.ap_trainers import CustomTrainer, EvaluationCallback
    from src.best_keeper import BestKeeper
    from aesthetic_data_analysis import compare

class SupressServerMessages(object):
    def __init__(self):
        self.stdout= sys.stdout

    def write(self, message:str):
        if message.startswith("127.0.0.1"):
            pass
        else:
            self.stdout.write(message)

    def flush(self):
        pass

sys.stderr = SupressServerMessages()

def combine_metadata(*args):
    metadata = {}
    for dic in args:
        for k in dic:
            metadata[k] = dic[k]
    return metadata

def train_predictor(feature_extractor:FeatureExtractor, ds:QuickDataset, eds:QuickDataset, tds:QuickDataset):
    with Timer('Create model'):
        predictor = AestheticPredictor(pretrained=None, feature_extractor=feature_extractor, **Args.aesthetic_model_extras)

    with Timer('Train model'):
        train_args = TrainingArguments( remove_unused_columns=False, **Args.training_args )
        callback = EvaluationCallback(datasets_to_shuffle=(tds,), datasets_to_score=(('training',tds),('evaluate',eds),))
        trainer = CustomTrainer.trainer(loss = Args.loss_model, model = predictor, 
                                        train_dataset = tds, eval_dataset = None, 
                                        args = train_args, callbacks = [callback,],
                                        **Args.trainer_extras )
        trainer.train()

    with Timer("Evaluate model"):
        predictor.eval()
        with torch.no_grad(): 
            ds.update_prediction(predictor)
            tds.update_prediction(predictor)
            eds.update_prediction(predictor)
        metrics = {}
        for measure in Args.measures:
            for the_set, the_set_name in ((ds, "full"),(tds, "train"),(eds, "eval")):
                with Timer(f"{the_set_name}_{measure}"):
                    metrics[f"{the_set_name}_{measure}"] = the_set.__getattribute__(f"get_{measure}")(divider=Args.accuracy_divider) if len(the_set) else 0

    with Timer("Save model"):
        metadata = combine_metadata( ds.get_metadata(), feature_extractor.get_metadata(), predictor.get_metadata() )
        save_file(predictor.state_dict(),Args.save_model_path,metadata=metadata)

    metrics['extra'] = predictor.info()

    return metrics

def validate():
    assert os.path.isdir(Args.directory), f"{Args.directory} doesn't exist or isn't a directory"
    assert os.path.exists(os.path.join(Args.directory, Args.scores)), f"{os.path.join(Args.directory, Args.scores)} not found"

def main():
    Args.parse_arguments(show=True)
    validate()
    name = f"{Args.get('name','')}_{random.randint(10000,99999)}"

    best_keeper = BestKeeper(save_model_path=Args.save_model_path, minimise=Args.best_minimize)

    with Timer('Build datasets from images') as logger:
        ds = QuickDataset.from_scorefile(top_level_directory=Args.directory, scorefilename=Args.scores)
        ds.allocate_split(fraction_for_eval=Args.fraction_for_eval, eval_pick_seed=Args.eval_pick_seed, replace=Args.ignore_existing_split)

        feature_extractor = FeatureExtractor.get_feature_extractor(pretrained=Args.feature_extractor_model, image_directory=Args.directory, device="cuda", **Args.feature_extractor_extras)
        ds.extract_features(feature_extractor)
        tds = ds.subset(lambda a:a=='train', 'split')
        eds = ds.subset(lambda a:a!='train', 'split')

        logger(f"{len(ds)} images ({len(tds)} training, {len(eds)} evaluation)")

    with Timer("Metaparameter search"):
        ta = Args.training_args
        def objective(trial:optuna.trial.Trial):
            ta['num_train_epochs']            =             Args.meta(trial.suggest_int,  'num_train_epochs',  Args.train_epochs)
            ta['learning_rate']               = math.pow(10,Args.meta(trial.suggest_float,'log_learning_rate', Args.log_lr))
            ta['per_device_train_batch_size'] =             Args.meta(trial.suggest_int,  'batch_size',        Args.batch_size)
            ta['warmup_ratio']                =             Args.meta(trial.suggest_float,'warmup_ratio',      Args.warmup_ratio)
            if Args.loss_model=='nll': ta['per_device_train_batch_size'] = ((ta['per_device_train_batch_size']+1)//2)*2

            Args.set("layers", list( Args.meta(trial.suggest_int,  f"layer_size_{i}",  Args.layer_size) for i in (0,1) ) )
            Args.set("dropouts", list( Args.meta(trial.suggest_float,  f"dropout_{i}",  Args.dropout) for i in (0,1) ) )

            trial.set_user_attr("Input number of features", feature_extractor.number_of_features)
            result = train_predictor(feature_extractor, ds=ds, eds=eds, tds=tds)
            score = result[Args.parameter_for_scoring]
            trial.set_user_attr('extra', str(result.pop('extra','')))
            for r in result: trial.set_user_attr(r, float(result[r]))

            best_keeper.keep_if_best(score)

            return score

        if Args.sampler=="CmaEs": sampler = optuna.samplers.CmaEsSampler()
        elif Args.sampler=="random": sampler = optuna.samplers.RandomSampler()
        elif Args.sampler=="QMC": sampler = optuna.samplers.QMCSampler()
        else: raise NotImplementedError()

        if not Args.no_server:
            print("Starting optuna dashboard server")
            storage:optuna.storages.BaseStorage = optuna.storages.RDBStorage(url=r"sqlite:///db.sqlite")
            study:optuna.study.Study = optuna.create_study(study_name=name, direction=Args.direction, sampler=sampler, storage=storage)
            threading.Thread(target=run_server,kwargs={'storage':storage}).start()

        for k in Args.keys: study.set_user_attr(k, Args.get(k))
        study.set_user_attr("image_count", len(ds))

        study.optimize(objective, n_trials=Args.trials)

        best_filepath = best_keeper.restore_best()

    with Timer('Statistics'):
        predictor = AestheticPredictor.from_pretrained(pretrained=best_filepath, image_directory=Args.directory)
        predictor.eval()
        with torch.no_grad():
            ds.update_prediction(predictor)
        compare("Best model",  eds.scores(), eds.item('model_score'))
        if Args.savefile: ds.save_as_scorefile(os.path.join(Args.directory, Args.savefile))

if __name__=='__main__':
    profile.run('main()', 'profile.stats')