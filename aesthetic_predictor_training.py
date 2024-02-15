from src.time_context import Timer

with Timer("Python imports"):
    import torch
    from safetensors.torch import save_file
    import math, random, os

    from src.data_holder import DataHolder
    from transformers import TrainingArguments
    import optuna

    from src.ap.dataset import QuickDataset

    from arguments import Args 
    from src.ap.feature_extractor import FeatureExtractor
    from src.ap.aesthetic_predictor import AestheticPredictor
    from src.ap.ap_trainers import CustomTrainer, EvaluationCallback
    from src.best_keeper import BestKeeper
    from src.ap.create_scorefiles import create_scorefiles

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
        trainer = CustomTrainer.trainer(loss = Args.loss_model, model = predictor, 
                                        train_dataset = tds, eval_dataset = eds, 
                                        args = train_args, callbacks = [EvaluationCallback((tds,eds,ds)),],
                                        **Args.trainer_extras )
        trainer.train()

    with Timer("Evaluate model"):
        predictor.eval()
        with torch.no_grad(): ds.update_prediction(predictor)
        metrics = {}
        for measure in Args.measures:
            for the_set, the_set_name in ((ds, "full"),(tds, "train"),(eds, "eval")):
                with Timer(f"{the_set_name}_{measure}"):
                    metrics[f"{the_set_name}_{measure}"] = the_set.__getattribute__(f"get_{measure}")()

    with Timer("Save model"):
        metadata = combine_metadata( ds.get_metadata(), feature_extractor.get_metadata(), predictor.get_metadata() )
        save_file(predictor.state_dict(),Args.save_model_path,metadata=metadata)

    metrics['extra'] = predictor.info()

    return metrics

def validate():
    assert os.path.isdir(Args.directory), f"{Args.directory} doesn't exist or isn't a directory"
    assert os.path.exists(os.path.join(Args.directory, Args.scores)), f"{os.path.join(Args.directory, Args.scores)} not found"

if __name__=='__main__':
    Args.parse_arguments(show=True)
    validate()
    name = f"{Args.get('name','')}_{random.randint(10000,99999)}"

    best_keeper = BestKeeper(save_model_path=Args.save_model_path, minimise=Args.best_minimize)

    with Timer('Build datasets from images') as logger:
        data = DataHolder(top_level=Args.directory, 
                          use_score_file=Args.scores, 
                          fraction_for_test=Args.fraction_for_test,
                          test_pick_seed=Args.test_pick_seed)
        df = data.get_dataframe()
        ds = QuickDataset(df)

        feature_extractor = FeatureExtractor.get_feature_extractor(pretrained=Args.feature_extractor_model, image_directory=Args.directory, device="cuda", **Args.feature_extractor_extras)
        feature_extractor.precache((f for f in df['image']))
        df['features'] = [feature_extractor.get_features_from_file(f, device="cpu") for f in df['image']]
        
        tds = QuickDataset(df, 'train')
        eds = QuickDataset(df, 'test')

        logger(f"{len(ds)} images ({len(tds)} training, {len(eds)} evaluation)")

    with Timer("Metaparameter search"):
        ta = Args.training_args
        ma = Args.metaparameter_args
        def objective(trial:optuna.trial.Trial):
            ta['num_train_epochs']            =             Args.meta(trial.suggest_int,  'num_train_epochs',  Args.train_epochs)
            ta['learning_rate']               = math.pow(10,Args.meta(trial.suggest_float,'log_learning_rate', Args.log_lr))
            ta['per_device_train_batch_size'] =             Args.meta(trial.suggest_int,  'batch_size',        Args.batch_size)
            ta['warmup_ratio']                =             Args.meta(trial.suggest_float,'warmup_ratio',      Args.warmup_ratio)
            if Args.loss_model=='nll': ta['per_device_train_batch_size'] = ((ta['per_device_train_batch_size']+1)//2)*2

            Args.set("dropouts", Args.meta_list(trial.suggest_float, "dropouts", ma["dropouts"] ))
            Args.set("layers", Args.meta_list(trial.suggest_int, "layers", ma["layers"] ))

            trial.set_user_attr("Input number of features", feature_extractor.number_of_features)
            result = train_predictor(feature_extractor, ds, eds, tds)
            score = result[Args.parameter_for_scoring]
            trial.set_user_attr('extra', str(result.pop('extra','')))
            for r in result: trial.set_user_attr(r, float(result[r]))

            best_keeper.keep_if_best(score)

            return score

        if Args.sampler=="CmaEs": sampler = optuna.samplers.CmaEsSampler()
        elif Args.sampler=="random": sampler = optuna.samplers.RandomSampler()
        elif Args.sampler=="QMC": sampler = optuna.samplers.QMCSampler()
        else: raise NotImplementedError()

        study:optuna.study.Study = optuna.create_study(study_name=name, direction=Args.direction, sampler=sampler, storage=r"sqlite:///db.sqlite")
        print(f"optuna-dashboard sqlite:///db.sqlite")
        for k in ma: study.set_user_attr(k, ma[k])
        for k in Args.keys: study.set_user_attr(k, Args.get(k))
        study.set_user_attr("image_count", len(df))

        study.optimize(objective, n_trials=Args.trials)

    with Timer('Save results'):
        best_filepath = best_keeper.restore_best()

        predictor = AestheticPredictor.from_pretrained(pretrained=best_filepath, image_directory=Args.directory)
        predictor.eval()
        with torch.no_grad():
            create_scorefiles(predictor, database_scores=data.get_image_scores(), 
                            model_scorefile=Args.get("model_scorefile",None), 
                            error_scorefile=Args.get("error_scorefile",None))
            data.save_split(Args.get('splitfile',None))
