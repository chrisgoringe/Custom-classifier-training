from src.time_context import Timer

with Timer("Python imports"):
    import torch
    from safetensors.torch import save_file
    import math, random

    from src.data_holder import DataHolder
    from transformers import TrainingArguments
    import optuna

    from src.ap.dataset import QuickDataset

    from arguments import training_args, args, get_args, metaparameter_args, MetaRangeProcessor, aesthetic_model_extras, trainer_extras
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
    pretrained = args['load_model_path']

    with Timer('Create model'):
        predictor = AestheticPredictor(pretrained=pretrained, feature_extractor=feature_extractor, 
                                       dropouts=args['dropouts'], hidden_layer_sizes=args['hidden_layers'],
                                       dropouts_0=args.get('dropouts_0',None), hidden_layer_sizes_0=args.get('hidden_layers_0',None),
                                       output_channels=(2 if args['loss_model']=="nll" else 1),
                                       **aesthetic_model_extras)

    with Timer('Train model'):
        training_args["metric_for_best_model"] = "ab" if args['loss_model']=="ab" else "loss"
        train_args = TrainingArguments( remove_unused_columns=False, **training_args )
        trainer = CustomTrainer.trainer(loss = args['loss_model'], model = predictor, 
                                        train_dataset = tds, eval_dataset = eds, 
                                        args = train_args, callbacks = [EvaluationCallback((tds,eds,ds)),],
                                        **trainer_extras )
        trainer.train()

    with Timer("Evaluate model"):
        predictor.eval()
        with torch.no_grad(): ds.update_prediction(predictor)
        sets = ((ds, "full"),(tds, "train"),(eds, "eval"))
        measures = ('ab','mse','nll')
        metrics = {f"{the_set_name}_{measure}" : the_set.__getattribute__(f"get_{measure}")() for the_set, the_set_name in sets for measure in measures}

    with Timer("Save model"):
        metadata = combine_metadata( ds.get_metadata(), feature_extractor.get_metadata(), predictor.get_metadata() )
        save_file(predictor.state_dict(),args['save_model_path'],metadata=metadata)

    return metrics

if __name__=='__main__':
    get_args(aesthetic_training=True, aesthetic_model=True)
    name = f"{metaparameter_args['name']}_{random.randint(10000,99999)}" if metaparameter_args.get('name',None) else None

    best_keeper = BestKeeper(save_model_path=args['save_model_path'], minimise=(not args['parameter_for_scoring'].endswith("_ab")))

    with Timer('Build datasets from images') as logger:
        data = DataHolder(top_level=args['top_level_image_directory'], 
                          save_model_folder=args['save_model'], 
                          use_score_file=args['scorefile'], 
                          fraction_for_test=args['fraction_for_test'],
                          test_pick_seed=args['test_pick_seed'])
        df = data.get_dataframe()
        ds = QuickDataset(df)

        feature_extractor = FeatureExtractor.get_feature_extractor(pretrained=args['clip_model'], image_directory=args['top_level_image_directory'], device="cuda")
        feature_extractor.precache((f for f in df['image']))
        df['features'] = [feature_extractor.get_features_from_file(f, device="cpu") for f in df['image']]
        
        tds = QuickDataset(df, 'train')
        eds = QuickDataset(df, 'test')

        logger(f"{len(ds)} images ({len(tds)} training, {len(eds)} evaluation)")

    with Timer("Metaparameter search"):
        mrp = MetaRangeProcessor()
        def objective(trial:optuna.trial.Trial):
            training_args['num_train_epochs']            =             mrp.meta(trial.suggest_int,  'num_train_epochs',  metaparameter_args['num_train_epochs'])
            training_args['learning_rate']               = math.pow(10,mrp.meta(trial.suggest_float,'log_learning_rate', metaparameter_args['log_learning_rate']))
            training_args['per_device_train_batch_size'] =         2 * mrp.meta(trial.suggest_int,  'half_batch_size',   metaparameter_args['half_batch_size'])
            training_args['warmup_ratio']                =             mrp.meta(trial.suggest_float,'warmup_ratio',      metaparameter_args['warmup_ratio'])

            for i,k in enumerate(trainer_extras.get('special_lr_parameters',[])):
                trainer_extras['special_lr_parameters'][k] = math.pow(10,mrp.meta(trial.suggest_float, f"special_lr_{i}", metaparameter_args['delta_log_spec_lr']))

            for suffix in ('', '_0'):
                if f"dropouts{suffix}" in metaparameter_args and metaparameter_args[f"dropouts{suffix}"]:
                    args[f"dropouts{suffix}"]      = mrp.meta_list(trial.suggest_float, f"dropouts{suffix}",    metaparameter_args[f"dropouts{suffix}"] )

                if f"hidden_layers{suffix}" in metaparameter_args and metaparameter_args[f"hidden_layers{suffix}"]:
                    args[f"hidden_layers{suffix}"] = mrp.meta_list(trial.suggest_int,   f"hidden_layers{suffix}",metaparameter_args[f"hidden_layers{suffix}"] )

            result = train_predictor(feature_extractor, ds, eds, tds)
            score = result[args['parameter_for_scoring']]
            for r in result: trial.set_user_attr(r, float(result[r]))

            if best_keeper.bad_by(score, args['prune_bad_by'], args['prune_bad_limit']): 
                logger(f"Pruning score {score}")
                raise optuna.TrialPruned()

            best_keeper.keep_if_best(score)

            return score

        sampler = None
        if 'sampler' in metaparameter_args and metaparameter_args['sampler']:
            if metaparameter_args['sampler']=="CmaEs": sampler = optuna.samplers.CmaEsSampler()
            elif metaparameter_args['sampler']=="random": sampler = optuna.samplers.RandomSampler()
            elif metaparameter_args['sampler']=="QMC": sampler = optuna.samplers.QMCSampler()
            else: raise NotImplementedError()

        study:optuna.study.Study = optuna.create_study(study_name=name, direction=args['direction'], sampler=sampler, storage=r"sqlite:///db.sqlite")
        print(f"optuna-dashboard sqlite:///db.sqlite")
        for k in metaparameter_args: study.set_user_attr(k, metaparameter_args[k])
        for k in args: study.set_user_attr(k, args[k])
        study.set_user_attr("image_count", len(df))

        study.optimize(objective, n_trials=metaparameter_args['meta_trials'])

    with Timer('Save results'):
        best_filepath = best_keeper.restore_best()

        predictor = AestheticPredictor.from_pretrained(pretrained=best_filepath, image_directory=args['top_level_image_directory'])
        predictor.eval()
        with torch.no_grad():
            create_scorefiles(predictor, database_scores=data.get_image_scores(), 
                            model_scorefile=args.get("model_scorefile",None), 
                            error_scorefile=args.get("error_scorefile",None))
            data.save_split(args.get('splitfile',None))
