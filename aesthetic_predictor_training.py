from src.time_context import Timer

with Timer("Python imports"):
    import torch
    from safetensors.torch import save_file
    import os, math, shutil, tempfile

    from src.data_holder import DataHolder
    from transformers import TrainingArguments

    from src.ap.dataset import QuickDataset

    from arguments import training_args, args, get_args, metaparameter_args, MetaRangeProcessor
    from src.ap.feature_extractor import FeatureExtractor
    from src.ap.aesthetic_predictor import AestheticPredictor
    from src.ap.ap_trainers import CustomTrainer, EvaluationCallback

def combine_metadata(*args):
    metadata = {}
    for dic in args:
        for k in dic:
            metadata[k] = dic[k]
    return metadata

def train_predictor(feature_extractor:FeatureExtractor, ds:QuickDataset, eds:QuickDataset, tds:QuickDataset):
    pretrained = args['load_model_path']

    with Timer('load models'):
        predictor = AestheticPredictor(pretrained=pretrained, feature_extractor=feature_extractor, 
                                       input_size=feature_extractor.number_of_features,
                                       dropouts=args['dropouts'], hidden_layer_sizes=args['hidden_layers'])

    with Timer('Predict values') as logger:
        with torch.no_grad():
            ds.update_prediction(predictor)
        logger("==== Start (all images): rmse {:>6.3f} ab {:>5.2f}%".format(ds.get_rmse(),100*ds.get_ab_score()))

    with Timer('train model'):
        training_args["metric_for_best_model"] = "ranking" if args['loss_model']=="ranking" else "loss"
        train_args = TrainingArguments( remove_unused_columns=False, **training_args )
        CustomTrainer.trainer(  loss = args['loss_model'], model = predictor, 
                                train_dataset = tds, eval_dataset = eds, 
                                args = train_args, callbacks = [EvaluationCallback((tds,eds,ds)),],
                            ).train()
        ds.update_prediction(predictor)

        metadata = combine_metadata( ds.get_metadata(), feature_extractor.get_metadata(), predictor.get_metadata() )
        save_file(predictor.state_dict(),args['save_model_path'],metadata=metadata)
   
    if args['loss_model']=='ranking':
        return eds.get_ab_score(), tds.get_ab_score(), ds.get_rmse()
    else:
        return eds.get_rmse(), tds.get_rmse(tds), ds.get_ab_score()

class BestKeeper:
    def __init__(self, save_model_path, minimise):
        self.best_score = None
        self.temp_dir = tempfile.TemporaryDirectory()
        self.best_temp = os.path.join(self.temp_dir.name, "bestmodel.safetensors")
        self.save_model_path = save_model_path
        self.minimise = minimise

    def keep_if_best(self, score):
        if self.best_score is None or (score<self.best_score and self.minimise) or (score>self.best_score and not self.minimise):
            shutil.copyfile(self.save_model_path, self.best_temp)
            self.best_score = score

    def restore_best(self):
        shutil.copyfile(self.best_temp, self.save_model_path)

if __name__=='__main__':
    get_args(aesthetic_training=True, aesthetic_model=True)

    with Timer('Extract features from images') as logger:
        feature_extractor = FeatureExtractor.get_feature_extractor(pretrained=args['clip_model'], image_directory=args['top_level_image_directory'], device="cuda")
        data = DataHolder(top_level=args['top_level_image_directory'], save_model_folder=args['save_model'], use_score_file=args['scorefile'])
        df = data.get_dataframe()
        ds = QuickDataset(df)
        with Timer('Feature extraction'):
            feature_extractor.precache((f for f in df['image']))
            df['features'] = [feature_extractor.get_features_from_file(f, device="cpu") for f in df['image']]
        df['score'] = [float(l) for l in df['label_str']]
        tds = QuickDataset(df, 'train')
        eds = QuickDataset(df, 'test')
        logger(f"{len(ds)} images ({len(tds)} training, {len(eds)} evaluation)")

    best_keeper = BestKeeper(save_model_path=args['save_model_path'], minimise=args['loss_model']=='mse')
    with Timer("Metaparameter search"):
        mrp = MetaRangeProcessor()
        import optuna
        def objective(trial:optuna.trial.Trial):
            training_args['num_train_epochs']            =             mrp.meta(trial.suggest_int,  'num_train_epochs',  metaparameter_args['num_train_epochs'])
            training_args['learning_rate']               = math.pow(10,mrp.meta(trial.suggest_float,'log_learning_rate', metaparameter_args['log_learning_rate']))
            training_args['per_device_train_batch_size'] =         2 * mrp.meta(trial.suggest_int,  'half_batch_size',   metaparameter_args['half_batch_size'])
            training_args['warmup_ratio']                =             mrp.meta(trial.suggest_float,'warmup_ratio',      metaparameter_args['warmup_ratio'])

            if 'dropouts' in metaparameter_args and metaparameter_args['dropouts']:
                args['dropouts']      = mrp.meta_list(trial.suggest_float, 'dropout',    metaparameter_args['dropouts'] )

            if 'hidden_layers' in metaparameter_args and metaparameter_args['hidden_layers']:
                args['hidden_layers'] = mrp.meta_list(trial.suggest_int,   'hidden_layer',metaparameter_args['hidden_layers'] )

            result = train_predictor(feature_extractor, ds, eds, tds)
            score = result[0]
            if not score: raise optuna.TrialPruned()

            trial.set_user_attr('train-dataset', float(result[1]))
            trial.set_user_attr('eval_train_difference', float(result[1])-float(result[0]))
            trial.set_user_attr('rmse_loss' if args['loss_model']=='ranking' else 'ranking_score_loss', float(result[2]))
            
            best_keeper.keep_if_best(score)
            if not mrp.any_ranges: 
                print("None of the metaparameters had ranges, so just doing a single trial")
                trial.study.stop()

            return score
        direction='minimize' if args['loss_model']=='mse' else 'maximize'

        sampler = None
        if 'sampler' in metaparameter_args and metaparameter_args['sampler']=="CmaEs": sampler = optuna.samplers.CmaEsSampler()

        study:optuna.study.Study = optuna.create_study(direction=direction, sampler=sampler, storage=r"sqlite:///dfb.sqlite")
        print(f"optuna-dashboard ")
        for k in metaparameter_args: study.set_user_attr(k, metaparameter_args[k])
        for k in args: study.set_user_attr(k, args[k])
        study.set_user_attr("image_count", len(df))
        study.optimize(objective, n_trials=metaparameter_args['meta_trials'])

        best_keeper.restore_best()
