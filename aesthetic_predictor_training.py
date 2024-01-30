from src.time_context import Timer

with Timer("Python imports"):
    import torch
    from safetensors.torch import save_file
    import os, math, shutil, random

    from src.data_holder import DataHolder
    from transformers import TrainerControl, TrainerState, TrainingArguments, TrainerCallback

    from src.ap.dataset import QuickDataset

    from arguments import training_args, args, get_args
    from src.ap.clip import CLIP
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

def train_predictor():
    pretrained = args['load_model_path']
    top_level_images = args['top_level_image_directory']

    with Timer('load models'):
        clipper = CLIP.get_clip(pretrained=args['clip_model'], image_directory=top_level_images)
        predictor = AestheticPredictor(pretrained=pretrained, clipper=clipper, input_size=clipper.top_level_size,
                                       dropouts=args['dropouts'], 
                                       hidden_layer_sizes=args['hidden_layers'])

    with Timer('Prepare images') as logger:
        data = DataHolder(top_level=top_level_images, save_model_folder=args['save_model'], use_score_file=args['use_score_file'])
        df = data.get_dataframe()
        ds = QuickDataset(df)
        with Timer('CLIP'):
            clipper.precache((f for f in df['image']))
            df['features'] = [clipper.prepare_from_file(f, device="cpu") for f in df['image']]
        df['score'] = [float(l) for l in df['label_str']]
        tds = QuickDataset(df, 'train')
        eds = QuickDataset(df, 'test')
        logger(f"{len(ds)} images ({len(tds)} training, {len(eds)} evaluation)")

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
            return get_ab_score(eds), get_ab_score(ds), get_rmse(eds)
        else:
            return get_rmse(eds), get_rmse(ds), get_ab_score(eds)

def create_name():
    ran = "".join( random.choices("0123456789", k=6) )
    return f"{args['hidden_layers']}_{args['dropouts']}_{ran}"

best_score = None
if __name__=='__main__':
    get_args(aesthetic_training=True, aesthetic_model=True)
    best_temp = os.path.splitext(args['save_model_path'])[0]+"-best.safetensors"

    if args['mode']=='metasearch':
  
        with Timer("Metaparameter search"):
            import optuna
            def objective(trial:optuna.trial.Trial):
                training_args['num_train_epochs'] = trial.suggest_int('num_train_epochs',2,100)
                training_args['learning_rate'] = math.pow(10,trial.suggest_float('log_learning_rate', -4, -1))
                training_args['per_device_train_batch_size'] = 2*trial.suggest_int('half_batch_size',1,64)
                training_args['warmup_ratio'] = trial.suggest_float('warmup_ratio',0.1,0.5)
                result = train_predictor()
                score = result[0]
                trial.set_user_attr('full-dataset', float(result[1]))
                trial.set_user_attr('other-attr', float(result[2]))
                global best_score
                if best_score is None or (score<best_score and args['loss_model']=='mse') or (score>best_score and args['loss_model']!='mse'):
                    shutil.copyfile(args['save_model_path'], best_temp)
                    best_score = score
                return score
            direction='minimize' if args['loss_model']=='mse' else 'maximize'
            study:optuna.study.Study = optuna.create_study(study_name=create_name(), direction=direction, storage="sqlite:///db.sqlite3")
            print("optuna-dashboard sqlite:///db.sqlite3")
            for k in args: study.set_user_attr(k, args[k])
            study.optimize(objective, n_trials=args['meta_trials'])
            print(f"Best model copied into {args['save_model_path']}")
            shutil.copyfile(best_temp,args['save_model_path'])

    else:
        train_predictor()