from src.time_context import Timer

with Timer("Python imports"):
    import torch
    from safetensors.torch import save_file
    import os, math, shutil, tempfile

    from src.data_holder import DataHolder
    from transformers import TrainerControl, TrainerState, TrainingArguments, TrainerCallback

    from src.ap.dataset import QuickDataset

    from arguments import training_args, args, get_args, metaparameter_args
    from src.ap.feature_extractor import FeatureExtractor
    from src.ap.aesthetic_predictor import AestheticPredictor
    from src.ap.ap_trainers import CustomTrainer
    
class EvaluationCallback(TrainerCallback):
    def __init__(self, every, datasets, labels, shuffles):
        self.last = 0
        self.every = every or 0
        self.datasets = list(zip(datasets,labels,shuffles))

    def do_eval(self, state, predictor:AestheticPredictor):
        for dataset, label, _ in self.datasets:
            with torch.no_grad():
                was_train = predictor.training
                predictor.eval()
                dataset.update_prediction(predictor)
                if was_train: predictor.train()
            Timer.message("==== Epoch {:>3} ({:8}): rmse {:>6.3f} ab {:>5.2f}%".format(state.epoch,label,get_rmse(dataset),100*get_ab_score(dataset)))

    def on_epoch_end(self, arguments, state: TrainerState, control, **kwargs):
        for dataset, _, shuffle in self.datasets:
            if shuffle: dataset.shuffle()
            
        if state.epoch - self.last < self.every: return
        self.last = state.epoch
        self.do_eval(state, kwargs['model'])

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch != self.last: self.do_eval(state, kwargs['model'])

def get_ab_score(ds:QuickDataset):
    right = 0
    wrong = 0
    true_predicted = ds.columns('score','predicted_score')
    for i, a in enumerate(true_predicted):
        for b in true_predicted[i+1:]:
            if a[0]==b[0] or a[1]==b[1]: continue
            if (a[0]<b[0] and a[1]<b[1]) or (a[0]>b[0] and a[1]>b[1]): right += 1
            else: wrong += 1
    return right/(right+wrong) if (right+wrong) else 0
    
def get_rmse(eds:QuickDataset):
    loss_fn = torch.nn.MSELoss()
    rmse = loss_fn(torch.tensor(eds.column('score')), torch.tensor(eds.column('predicted_score')))
    return rmse

def combine_metadata(*args):
    metadata = {}
    for dic in args:
        for k in dic:
            metadata[k] = dic[k]
    return metadata

def train_predictor(clipper:FeatureExtractor, ds:QuickDataset, eds:QuickDataset, tds:QuickDataset):
    pretrained = args['load_model_path']

    with Timer('load models'):
        predictor = AestheticPredictor(pretrained=pretrained, clipper=clipper, input_size=clipper.number_of_features,
                                       dropouts=args['dropouts'], 
                                       hidden_layer_sizes=args['hidden_layers'])

    with Timer('Predict values') as logger:
        with torch.no_grad():
            ds.update_prediction(predictor)
        logger("==== Start (all images): rmse {:>6.3f} ab {:>5.2f}%".format(get_rmse(ds),100*get_ab_score(ds)))

    with Timer('train model'):
        train_args = TrainingArguments( remove_unused_columns=False, **training_args )
        callback = EvaluationCallback(every=args['eval_every_n_epochs'], datasets=[ds,eds], labels=["all images"," test set "], shuffles=[True,True])

        CustomTrainer.trainer(  loss = args['loss_model'], model = predictor, 
                                train_dataset = tds, eval_dataset = eds, 
                                args = train_args, callbacks = [callback,], 
                            ).train()
        ds.update_prediction(predictor)

        metadata = combine_metadata( ds.get_metadata(), clipper.get_metadata(), predictor.get_metadata() )
        save_file(predictor.state_dict(),args['save_model_path'],metadata=metadata)
   
    if args['mode']=='metasearch':
        if args['loss_model']=='ranking':
            return get_ab_score(eds), get_ab_score(tds), get_rmse(ds)
        else:
            return get_rmse(eds), get_rmse(tds), get_ab_score(ds)

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

def meta(mthd, label:str, rng:tuple|list):
    if isinstance(rng,(tuple,list)):
        return mthd(label, *rng)
    else:
        return rng
    
def meta_list(mthd, label:str, rnges:tuple|list):
    result = []
    for i, rng in enumerate(rnges):
        result.append(meta(mthd, f"{label}_{i}", rng))
    return result


if __name__=='__main__':
    get_args(aesthetic_training=True, aesthetic_model=True)

    with Timer('Extract features from images') as logger:
        clipper = FeatureExtractor.get_feature_extractor(pretrained=args['clip_model'], image_directory=args['top_level_image_directory'])
        data = DataHolder(top_level=args['top_level_image_directory'], save_model_folder=args['save_model'], use_score_file=args['use_score_file'])
        df = data.get_dataframe()
        ds = QuickDataset(df)
        with Timer('CLIP'):
            clipper.precache((f for f in df['image']))
            df['features'] = [clipper.prepare_from_file(f, device="cpu") for f in df['image']]
        df['score'] = [float(l) for l in df['label_str']]
        tds = QuickDataset(df, 'train')
        eds = QuickDataset(df, 'test')
        logger(f"{len(ds)} images ({len(tds)} training, {len(eds)} evaluation)")

    if args['mode']=='metasearch':
        best_keeper = BestKeeper(save_model_path=args['save_model_path'], minimise=args['loss_model']=='mse')
        with Timer("Metaparameter search"):
            import optuna
            def objective(trial:optuna.trial.Trial):
                training_args['num_train_epochs']            =             meta(trial.suggest_int,  'num_train_epochs',  metaparameter_args['num_train_epochs'])
                training_args['learning_rate']               = math.pow(10,meta(trial.suggest_float,'log_learning_rate', metaparameter_args['log_learning_rate']))
                training_args['per_device_train_batch_size'] =         2 * meta(trial.suggest_int,  'half_batch_size',   metaparameter_args['half_batch_size'])
                training_args['warmup_ratio']                =             meta(trial.suggest_float,'warmup_ratio',      metaparameter_args['warmup_ratio'])

                if 'dropouts' in metaparameter_args and metaparameter_args['dropouts']:
                    args['dropouts']      = meta_list(trial.suggest_float, 'dropout',    metaparameter_args['dropouts'] )

                if 'hidden_layers' in metaparameter_args and metaparameter_args['hidden_layers']:
                    args['hidden_layers'] = meta_list(trial.suggest_int,   'hidden_layer',metaparameter_args['hidden_layers'] )

                result = train_predictor(clipper, ds, eds, tds)
                score = result[0]
                if not score: raise optuna.TrialPruned()

                trial.set_user_attr('train-dataset', float(result[1]))
                trial.set_user_attr('eval_train_difference', float(result[1])-float(result[0]))
                trial.set_user_attr('rmse_loss' if args['loss_model']=='ranking' else 'ranking_score_loss', float(result[2]))
                
                best_keeper.keep_if_best(score)

                return score
            direction='minimize' if args['loss_model']=='mse' else 'maximize'
            study:optuna.study.Study = optuna.create_study(direction=direction, storage="sqlite:///db.sqlite3")
            print("optuna-dashboard sqlite:///db.sqlite3")
            for k in args: study.set_user_attr(k, args[k])
            study.optimize(objective, n_trials=metaparameter_args['meta_trials'])

            best_keeper.restore_best()
    else:
        train_predictor(clipper, ds, eds, tds)