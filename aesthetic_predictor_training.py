from src.time_context import Timer
with Timer("Python imports"):
    import torch
    from safetensors.torch import save_file
    import os, random, statistics, math, shutil

    from src.data_holder import DataHolder
    from transformers import TrainerControl, TrainerState, TrainingArguments, TrainerCallback
    from pandas import DataFrame
    try:
        from renumics import spotlight
        have_spotlight = True
    except:
        have_spotlight = False

    from arguments import training_args, args, get_args
    from src.ap.clip import CLIP
    from src.ap.aesthetic_predictor import AestheticPredictor
    from src.ap.ap_trainers import CustomTrainer

    from src.metaparameter_searcher import MetaparameterSearcher
    
class QuickDataset(torch.utils.data.Dataset):
    def __init__(self, df:DataFrame, split:str=None):
        self._df = df
        self.map = [i for i in range(len(df)) if (not split or df['split'].array[i]==split)]
        self.shuffle()

    def __getitem__(self, i):
        x = self._df['features'].array[self.map[i]]
        y = torch.tensor(self._df['score'].array[self.map[i]], dtype=torch.float)
        return {"x":x, "y":y}

    def __len__(self):
        return len(self.map)
    
    def shuffle(self):
        random.shuffle(self.map)

    def update_prediction(self, predictor:AestheticPredictor):
        self._df['predicted_score'] = (predictor.evaluate_files(self._df['image'], eval_mode=True))

    def statistical_metadata(self):
        scores = self.column('predicted_score')
        return {
            "n_images"              : str(len(self.map)),
            "mean_predicted_score"  : str(statistics.mean(scores)),
            "stdev_predicted_score" : str(statistics.stdev(scores)),
        }

    def column(self, col:str) -> list:
        return [self._df[col].array[x] for x in self.map]
    
    def column_where(self, col:str, match_col:str, match:str) -> list:
        return [self._df[col].array[x] for x in self.map if self._df[match_col].array[x]==match] 
    
    def columns(self, *args) -> list:
        cols = []
        for x in self.map:
            cols.append(tuple(self._df[col].array[x] for col in args))
        return cols

class EvaluationCallback(TrainerCallback):
        def __init__(self, every, datasets, labels, shuffles):
            self.last = 0
            self.every = every or 0
            self.datasets = list(zip(datasets,labels,shuffles))

        def do_eval(self, state, predictor:AestheticPredictor):
            for dataset, label, _ in self.datasets:
                with torch.no_grad():
                    dataset.update_prediction(predictor)
                print(f"==== Epoch {str(state.epoch)} ({label}): ")
                report(dataset, "folder {:>1} average score {:>5.3f} +/ {:>5.3f}")

        def on_epoch_end(self, arguments, state: TrainerState, control, **kwargs):
            for dataset, _, shuffle in self.datasets:
                if shuffle: dataset.shuffle()
            if state.epoch - self.last < self.every: return
            self.last = state.epoch
            self.do_eval(state, kwargs['model'])

        def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.epoch != self.last: self.do_eval(state, kwargs['model'])

def ab_report(ds:QuickDataset, prnt="AB: {:>5}/{:<5} correct ({:>5.2f}%)"):
    right = 0
    wrong = 0
    true_predicted = ds.columns('score','predicted_score')
    for i, a in enumerate(true_predicted):
        for b in true_predicted[i+1:]:
            if a[0]==b[0]: continue
            if (a[0]<b[0] and a[1]<b[1]) or (a[0]>b[0] and a[1]>b[1]): right += 1
            else: wrong += 1
    if prnt: print(prnt.format(right,right+wrong,100*right/(right+wrong)))
    return right/(right+wrong)
    
def report(eds:QuickDataset, prnt:str, prntrmse:str="rms error {:>6.3}"):
    loss_fn = torch.nn.MSELoss()
    if prnt and not args['loss_model']=='ranking':
        for x in sorted(set(eds.column('label_str'))):
            scores = eds.column_where('predicted_score','label_str',x)
            std = statistics.stdev(scores) if len(scores)>1 else 0
            print(prnt.format(x,statistics.mean(scores),std))
    rmse = loss_fn(torch.tensor(eds.column('score')), torch.tensor(eds.column('predicted_score')))
    if prntrmse and not args['loss_model']=='ranking': print(prntrmse.format(rmse))
    ab_report(eds)
    return rmse

def train_predictor():
    pretrained = os.path.join(args['load_model'],"model.safetensors") if ('load_model' in args and args['load_model']) else args['base_model']
    top_level_images = args['top_level_image_directory']

    with Timer('load models'):
        clipper = CLIP(pretrained=args['clip_model'], image_directory=top_level_images)
        predictor = AestheticPredictor(pretrained=pretrained, relu=args['aesthetic_model_relu'], clipper=clipper)

    with Timer('Prepare images'):
        data = DataHolder(top_level=top_level_images, save_model_folder=args['save_model'], use_score_file=args['use_score_file'])
        df = data.get_dataframe()
        ds = QuickDataset(df)
        with Timer('CLIP'):
            df['features'] = [clipper.prepare_from_file(f, device="cpu") for f in df['image']]
            clipper.save_cache()
        df['score'] = [float(l) for l in df['label_str']]
        tds = QuickDataset(df, 'train')
        eds = QuickDataset(df, 'test')

    with torch.no_grad():
        ds.update_prediction(predictor)
    report(ds, "Start (train+eval): folder {:>1} average score {:>5.3f} +/ {:>5.3f}")

    with Timer('train model'):
        train_args = TrainingArguments( remove_unused_columns=False, push_to_hub=False, output_dir=args['save_model'], **training_args )
        callback = EvaluationCallback(every=args['eval_every_n_epochs'], datasets=[ds,eds], labels=["all","test"], shuffles=[False,True])

        CustomTrainer.trainer(  loss = args['loss_model'], model = predictor, 
                                train_dataset = tds, eval_dataset = eds, 
                                args = train_args, callbacks = [callback,], 
                            ).train()
        ds.update_prediction(predictor)
        metadata = ds.statistical_metadata()
        metadata['clip_model'] = args['clip_model']
        save_file(predictor.state_dict(),os.path.join(args['save_model'],"model.safetensors"),metadata=ds.statistical_metadata())

    if have_spotlight and 'spotlight' in args['mode']: 
        try:
            spotlight.show(ds._df)
        except:
            pass

    if args['mode']=='meta':
        return report(eds,None,None), report(tds,None,None), ab_report(eds,None), ab_report(tds,None)
    
    if args['mode']=='metasearch':
        return ab_report(eds,None)

if __name__=='__main__':
    get_args(aesthetic_training=True, aesthetic_model=True)

    if args['mode']=='meta':
        with open("meta.csv",'w') as f:
            print("epochs,lr,batch,train_loss,eval_loss,train_ab,eval_ab,time", file=f)
            for lr in args['meta_lr'] if args['meta_lr'] else [training_args['learning_rate'],]:
                for epochs in args['meta_epochs'] if args['meta_epochs'] else [training_args['num_train_epochs'],]:
                    for batch in args['meta_batch'] if args['meta_batch'] else [training_args['per_device_train_batch_size'],]:
                        training_args['num_train_epochs'] = epochs
                        training_args['learning_rate']= lr
                        training_args['per_device_train_batch_size'] = batch
                        with Timer("meta-train") as m:
                            eval_loss, train_loss, eval_ab, train_ab = train_predictor()
                            time_taken = m(None)
                        print(args['meta_fmt'].format(epochs, lr, batch, train_loss, eval_loss, train_ab, eval_ab, time_taken),file=f,flush=True)
                        #print(f"{epochs},{lr},{batch},{train_loss},{eval_loss},{train_ab},{eval_ab}",file=f, flush=True)
    elif args['mode']=='metasearch':
        initial = [ training_args['num_train_epochs'],training_args['learning_rate'],training_args['per_device_train_batch_size'] ]
        def evalfn(params):
            training_args['num_train_epochs'],training_args['learning_rate'],training_args['per_device_train_batch_size'] = params
            return train_predictor()
        def newpf(params, scaleguide):
            efac = math.pow(2,scaleguide) * (0.8 + 0.4*random.random())
            p0 = int(params[0] * efac) if random.random()<0.5 else int(params[0] / efac)
            if p0<2: p0=2
            p1 = (params[1] * efac) if random.random()<0.5 else (params[1] / efac)
            p2 = int(params[2] * efac) if random.random()<0.5 else int(params[2] / efac)
            p2 = 2*(p2//2)
            if (p2==0): p2=2
            return [p0,p1,p2]
        def callbk(params, score, bad, scale, tme, note):
            print(f"{params} -> {score}  ({bad}, {scale}) {tme} s - {note}", file=open("metasearch.txt",'+a'))
            print(f"{params} -> {score}  ({bad}, {scale}) {tme} s - {note}")
        def best_so_far():
            shutil.copytree(args['save_model'], args['save_model']+"-best", dirs_exist_ok=True)
        
        params, score = MetaparameterSearcher(initial_parameters=initial, 
                                              evaluation_function=evalfn, 
                                              new_parameter_function=newpf, 
                                              callback=callbk, 
                                              best_so_far_callback=best_so_far,
                                              minimise=False).search()
        print(f"Best parameters {params} -> {score}")
        if os.path.exists(args['save_model']+"-best"):
            shutil.copytree(args['save_model']+"-best", args['save_model'], dirs_exist_ok=True)
    else:
        train_predictor()