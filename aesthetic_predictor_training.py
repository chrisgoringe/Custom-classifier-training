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
        if not message.startswith("127.0.0.1"): self.stdout.write(message)

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
        callback = EvaluationCallback(datasets_to_shuffle=(tds,), datasets_to_score=(('training',tds),('evaluate',eds),), measures=Args.measures)
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
                    metrics[f"{the_set_name}_{measure}"] = the_set.__getattribute__(f"get_{measure}")(divider=Args.get("accuracy_divider",None)) if len(the_set) else 0

    with Timer("Save model"):
        metadata = combine_metadata( ds.get_metadata(), feature_extractor.get_metadata(), predictor.get_metadata() )
        save_file(predictor.state_dict(),Args.save_model_path,metadata=metadata)

    metrics['extra'] = predictor.info()

    return metrics

def main():
    name = f"{Args.get('name','')}_{random.randint(10000,99999)}"

    best_keeper = BestKeeper(save_model_path=Args.save_model_path, minimise=Args.score_direction=='minimize')

    with Timer('Build datasets'):
        ds = QuickDataset.from_scorefile(top_level_directory=Args.directory, scorefilename=Args.scores)
        ds.allocate_split(fraction_for_eval=Args.fraction_for_eval, eval_pick_seed=Args.eval_pick_seed, replace=Args.ignore_existing_split)
        if Args.normalise_weights and ds.has_item('weight'): ds.normalise('weight', mean=1)

    with Timer('Create feature_extractor') as logger:
        feature_extractor = FeatureExtractor.get_feature_extractor(pretrained=Args.feature_extractor, image_directory=Args.directory, device="cuda", **Args.feature_extractor_extras)
        Args.set('hidden_states_used', feature_extractor.hidden_states_used)
        for k in feature_extractor.metadata: logger("{:>20} : {:<40}".format(k, feature_extractor.metadata[k]))

    with Timer('Extract features:') as logger:
        ds.extract_features(feature_extractor)
        feature_extractor.clear_cache()
        if not Args.trials:
            logger("trials set to zero - exiting")
            return

    with Timer('Create test and evaluation subsets') as logger:
        tds = ds.subset(lambda a:a=='train', 'split')
        eds = ds.subset(lambda a:a!='train', 'split')
        logger(f"{len(ds)} images ({len(tds)} training, {len(eds)} evaluation)")

    with Timer("Metaparameter search") as logger:
        ta = Args.training_args
        def objective(trial:optuna.trial.Trial):
            ta['num_train_epochs']            =             Args.meta(trial.suggest_int,  'num_train_epochs',  Args.train_epochs)
            ta['learning_rate']               = math.pow(10,Args.meta(trial.suggest_float,'log_learning_rate', Args.log_lr))
            ta['per_device_train_batch_size'] =             Args.meta(trial.suggest_int,  'batch_size',        Args.batch_size)
            ta['warmup_ratio']                =             Args.meta(trial.suggest_float,'warmup_ratio',      Args.warmup_ratio)
            if Args.loss_model=='nll': ta['per_device_train_batch_size'] = ((ta['per_device_train_batch_size']+1)//2)*2

            Args.set("layers", [ Args.meta(trial.suggest_int,  f"layer_size_first",  Args.first_layer_size),
                                 Args.meta(trial.suggest_int,  f"layer_size_second",  Args.second_layer_size),])
                     
            Args.set("dropouts", [ Args.meta(trial.suggest_float,  f"dropout_at_input",  Args.input_dropout),
                                   Args.meta(trial.suggest_float,  f"dropout_internal",  Args.dropout), 
                                   Args.meta(trial.suggest_float,  f"dropout_at_output",  Args.output_dropout),])

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
        else: raise NotImplementedError(f"{Args.sampler} not implemented")

        storage:optuna.storages.BaseStorage = optuna.storages.RDBStorage(url=Args.database) if Args.database else None 
        study:optuna.study.Study = optuna.create_study(study_name=name, direction=Args.score_direction, sampler=sampler, storage=storage)
        for k in Args.keys: study.set_user_attr(k, Args.get(k))
        study.set_user_attr("image_count", len(ds))

        if Args.server!='off' and storage is not None:
            logger("Starting optuna dashboard server" + " as daemon" if Args.server=='daemon' else "") 
            threading.Thread(target=run_server,kwargs={'storage':storage}, daemon=(Args.server=='daemon')).start()

        study.optimize(objective, n_trials=Args.trials)

        best_filepath = best_keeper.restore_best()

    with Timer('Statistics'):
        with Timer('Load best model'):
            predictor = AestheticPredictor.from_pretrained(pretrained=best_filepath, image_directory=Args.directory, feature_extractor=feature_extractor)
            predictor.eval()
        with Timer('Update predictions'):
            with torch.no_grad():   
                ds.update_prediction(predictor)
        with Timer('Calculate stats'):
            compare("Best model",  eds.scores(), eds.item('model_score'))
            if Args.savefile: ds.save_as_scorefile(os.path.join(Args.directory, Args.savefile))

if __name__=='__main__':
    profile.run('main()', 'profile.stats')