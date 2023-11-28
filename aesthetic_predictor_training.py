from src.time_context import Timer

with Timer("Python imports"):
    import torch
    from safetensors.torch import save_file
    import os, statistics,  shutil

    from src.data_holder import DataHolder
    from transformers import TrainerControl, TrainerState, TrainingArguments, TrainerCallback

    from src.ap.dataset import QuickDataset

    from arguments import training_args, args, get_args
    from src.ap.clip import CLIP
    from src.ap.aesthetic_predictor import AestheticPredictor
    from src.ap.ap_trainers import CustomTrainer
    from src.ap.aesthetic_metaparameter import AMP

    from src.metaparameter_searcher import MetaparameterSearcher, ParameterSet
    
class EvaluationCallback(TrainerCallback):
        def __init__(self, every, datasets, labels, shuffles):
            self.last = 0
            self.every = every or 0
            self.datasets = list(zip(datasets,labels,shuffles))

        def do_eval(self, state, predictor:AestheticPredictor):
            for dataset, label, _ in self.datasets:
                with torch.no_grad():
                    dataset.update_prediction(predictor)
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
    pretrained = os.path.join(args['load_model'],"model.safetensors") if ('load_model' in args and args['load_model']) else args['base_model']
    top_level_images = args['top_level_image_directory']

    with Timer('load models'):
        clipper = CLIP(pretrained=args['clip_model'], image_directory=top_level_images)
        predictor = AestheticPredictor(pretrained=pretrained, relu=args['aesthetic_model_relu'], clipper=clipper)

    with Timer('Prepare images') as logger:
        data = DataHolder(top_level=top_level_images, save_model_folder=args['save_model'], use_score_file=args['use_score_file'])
        df = data.get_dataframe()
        ds = QuickDataset(df)
        with Timer('CLIP'):
            df['features'] = [clipper.prepare_from_file(f, device="cpu") for f in df['image']]
            clipper.save_cache()
        df['score'] = [float(l) for l in df['label_str']]
        tds = QuickDataset(df, 'train')
        eds = QuickDataset(df, 'test')
        logger(f"{len(ds)} images ({len(tds)} training, {len(eds)} evaluation)")

    with Timer('Predict values') as logger:
        with torch.no_grad():
            ds.update_prediction(predictor)
        logger("==== Start (all images): rmse {:>6.3f} ab {:>5.2f}%".format(get_rmse(ds),100*get_ab_score(ds)))

    with Timer('train model'):
        train_args = TrainingArguments( remove_unused_columns=False, push_to_hub=False, output_dir=args['save_model'], **training_args )
        callback = EvaluationCallback(every=args['eval_every_n_epochs'], datasets=[ds,eds], labels=["all images"," test set "], shuffles=[True,True])

        CustomTrainer.trainer(  loss = args['loss_model'], model = predictor, 
                                train_dataset = tds, eval_dataset = eds, 
                                args = train_args, callbacks = [callback,], 
                            ).train()
        ds.update_prediction(predictor)

        metadata = combine_metadata( ds.get_metadata(), clipper.get_metadata(), predictor.get_metadata() )

        save_file(predictor.state_dict(),os.path.join(args['save_model'],"model.safetensors"),metadata=metadata)
   
    if args['mode']=='metasearch':
        return get_ab_score(eds)

if __name__=='__main__':
    get_args(aesthetic_training=True, aesthetic_model=True)

    if args['mode']=='metasearch':
        initial = ParameterSet.from_args( training_args )
        def evalfn(params:ParameterSet):
            params.to_args(training_args)
            return train_predictor()
        def callbk(params:ParameterSet, score, bad, tme, note):
            params.print()
            params.print(open("metasearch.txt",'+a'))
            txt = "Score {:>5.2f}% ({:>1}) {:>6.1f}s - {:<30}".format(100*score, bad, tme, note)
            print(txt, file=open("metasearch.txt",'+a'))
            print(txt)
        def best_so_far():
            shutil.copytree(args['save_model'], args['save_model']+"-best", dirs_exist_ok=True)
        
        params, score = MetaparameterSearcher(initial_parameters=initial, 
                                              evaluation_function=evalfn, 
                                              new_parameter_function=AMP(even_batch=True).update_mps, 
                                              callback=callbk, 
                                              best_so_far_callback=best_so_far,
                                              minimise=False).search()
        print(f"Best parameters {params} -> {score}")
        if os.path.exists(args['save_model']+"-best"):
            shutil.copytree(args['save_model']+"-best", args['save_model'], dirs_exist_ok=True)
    else:
        train_predictor()