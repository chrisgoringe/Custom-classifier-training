import os, argparse

metaparameter_args = {
    # A list, each element is either a tuple (min, max) or a value
    "dropouts"    : [ (0.0, 0.8), (0.0, 0.8), ],
    "layers"      : [ (10, 1000), (10, 1000), ],
}

# training_args are passed directly into the TrainingArguments object.
# Below are the most common of the 101 arguments available
# see https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
training_args = {
    "lr_scheduler_type"             : "cosine",
    "per_device_eval_batch_size"    : 2000,     
    "save_strategy"                 : "no",
    "evaluation_strategy"           : "no",
    "output_dir"                    : "out",
}

###
# Argument processing
###

def _parse_arguments(into:dict):
    to_int_list = lambda s : list( int(x.strip()) for x in s.split(',') if x )
    to_string_list = lambda s : list( x.strip() for x in s.split(',') if x )

    parser = argparse.ArgumentParser("Compare a series of scorefiles")
    main_group = parser.add_argument_group('Main arguments')
    meta_group = parser.add_argument_group('Metaparameter Search')
    model_group = parser.add_argument_group('Defining the model architecture')
    training_group = parser.add_argument_group('Training constants')
    metaparameter_group = parser.add_argument_group('Metaparameters')

    main_group.add_argument('-d', '--directory', help="Top level directory", required=True)
    main_group.add_argument('--model', default="model.safetensors", help="Filename to save model (default model.safetensors)")
    main_group.add_argument('--scores', default="scores.json", help="Filename of scores file (default scores.json)")
    main_group.add_argument('--errors', default="errors.json", help="Filename of errors file (default errors.json)")
    main_group.add_argument('--split', default="split.json", help="Filename of split file (default split.json)")

    meta_group.add_argument('--name', help="Name prefix for Optuna")
    meta_group.add_argument('--trials', type=int, default=200, help="Number of metaparameter trials" )
    meta_group.add_argument('--sampler', default="CmaEs", choices=['CmaEs', 'random', 'QMC'], help="Metaparameter search algorithm")

    model_group.add_argument('--feature_extractor_model', default="ChrisGoringe/vitH16", help="Model to use for feature extraction", type=to_string_list)
    model_group.add_argument('--weight_n_output_layers', default=0, type=int, help="Add a trainable projection of last n output layers to the start of the model")
    model_group.add_argument('--hidden_states', default="", help="Comma separated list of the hidden states to concatenate (0 is output layer, 1 is last hidden layer etc.)", type=to_int_list)
    model_group.add_argument('--final_layer_bias', action="store_true", help="Train with a bias in the final layer") 
    model_group.add_argument('--model_seed', type=int, default=0, help="Seed for initialising model")

    training_group.add_argument('--loss_model', default='mse', choices=['mse','ab','nll'], help="Loss model (mse=mean square error, ab=ab ranking, nll=negative log likelihood)")
    training_group.add_argument('--include_other_evaluations', action="store_true", help="Calculate scoring options other than the one being used")
    training_group.add_argument('--parameter_for_scoring', default='eval_mse', help="dataset_scorer to be used to evaluate trained model in format [full|train|eval]_[ab|mse|nll]")
    training_group.add_argument('--fraction_for_test', type=float, default=0.25, help="fraction of images to be reserved for test (eval)")
    training_group.add_argument('--test_pick_seed', type=int, default=42, help="Seed for random numbers when choosing test images") 

    metaparameter_group.add_argument('--min_train_epochs', type=int, default=5)
    metaparameter_group.add_argument('--max_train_epochs', type=int, default=50)
    metaparameter_group.add_argument('--min_warmup_ratio', type=float, default=0.0)
    metaparameter_group.add_argument('--max_warmup_ratio', type=float, default=0.2)
    metaparameter_group.add_argument('--min_log_lr', type=float, default=-4.5)
    metaparameter_group.add_argument('--max_log_lr', type=float, default=-2.5)
    metaparameter_group.add_argument('--min_batch_size', type=int, default=1)
    metaparameter_group.add_argument('--max_batch_size', type=int, default=100)    

    namespace = parser.parse_args()
    d = vars(namespace)
    for argument in d: into[argument] = d[argument]

    # Calculated arguments:
    for argument in ['train_epochs', 'warmup_ratio', 'log_lr', 'batch_size']:
        into[argument] = d[f"min_{argument}"] if d[f"min_{argument}"] == d[f"max_{argument}"] else (d[f"min_{argument}"], d[f"max_{argument}"])

    into['save_model_path'] = os.path.join(into['directory'], into['model'])
    into['direction']='maximize' if into['loss_model']=='ab' else 'minimize'
    into['best_minimize'] = not (into['parameter_for_scoring'].endswith("_ab"))
    into['output_channels'] = 2 if into['loss_model']=='nll' else 1
    m = {into['loss_model'],}
    if into['include_other_evaluations']:
        m.add('ab')
        m.add('mse')
    into['measures'] = list(m)

    training_args["metric_for_best_model"] = "ab" if into['loss_model']=="ab" else "loss"

class _Args(object):
    instance = None
    def __init__(self):
        self.args = {}
        self.more_args = {}
        self.specials = {}

    def __getattr__(self, attr):
        return self.get(attr)
    
    def parse_arguments(self, show=False):
        _parse_arguments(self.args)
        self.more_args = {  "metaparameter_args" : metaparameter_args,
                            "training_args" : training_args,
                            }
        self.specials = {   "feature_extractor_extras" : ['hidden_states','weight_n_output_layers',],
                            "aesthetic_model_extras" : ['final_layer_bias','hidden_states','weight_n_output_layers','model_seed','dropouts','layers','output_channels',],
                            "trainer_extras" : [],
                            }
        if show: self.show_args()

    def get(self, attr, default=Exception()):
        if attr in self.args: return self.args[attr]
        if attr in self.specials: return { x : self.args[x] for x in self.specials[attr] }
        for m in self.more_args:
            if m==attr: return self.more_args[m]
            if attr in self.more_args[m]: return self.more_args[m][attr]
        if isinstance(default, Exception): raise KeyError(attr)
        return default
    
    def set(self, key, value):
        self.args[key] = value
    
    def show_args(self):
        for a in self.keys: print("{:>30} : {:<40}".format(a, str(self.get(a))))

    @property
    def keys(self):
        for d in [self.args,] + [self.more_args[m] for m in self.more_args]:
            for a in d: yield a

    def meta(self, mthd, label:str, rng:tuple|list):
        if isinstance(rng,(tuple,list)):
            return mthd(label, *rng)
        else:
            return rng
        
    def meta_list(self, mthd, label:str, rnges:tuple|list):
        result = []
        for i, rng in enumerate(rnges):
            result.append(self.meta(mthd, f"{label}_{i}", rng))
        return result

_Args.instance = _Args.instance or _Args()
Args = _Args.instance