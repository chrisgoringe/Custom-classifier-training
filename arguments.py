import os, argparse

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

class ArgumentException(Exception):
    pass

class CommentArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith('#'): return [] 
        line = "=".join(a.strip() for a in arg_line.split('='))
        return [line,] if len(line) else []

class _Args(object):
    instance = None

    def _parse_arguments(self):
        to_int_list = lambda s : list( int(x.strip()) for x in s.split(',') if x ) if s is not None else None
        to_string_list = lambda s : list( x.strip() for x in s.split(',') if x ) if s is not None else None

        parser = CommentArgumentParser("Compare a series of scorefiles", fromfile_prefix_chars='@')
        main_group = parser.add_argument_group('Main arguments')
        model_group = parser.add_argument_group('Defining the model architecture')
        features_group = parser.add_argument_group('Feature extraction')
        training_group = parser.add_argument_group('Training constants')
        metaparameter_group = parser.add_argument_group('Metaparameters')

        main_group.add_argument('-d', '--directory', help="Top level directory", required=True)
        main_group.add_argument('-s', '--savefile', default="", help="Filename of (csv) scores file to save (default no save)")
        main_group.add_argument('--model', default="model.safetensors", help="Filename to save model (default model.safetensors)")
        main_group.add_argument('--scores', default="scores.json", help="Filename of scores file (default scores.json)")
        main_group.add_argument('--server', default='on', choices=['on','daemon','off'], help="Start an optuna dashboard server. Use daemon to start in daemon thread (terminates with training), or off for no server")
        main_group.add_argument('--database', default="sqlite:///db.sqlite", help="Storage for optuna. Set --database= for no database (implies --server=off)")

        model_group.add_argument('--final_layer_bias', action="store_true", help="Train with a bias in the final layer") 
        model_group.add_argument('--model_seed', type=int, default=0, help="Seed for initialising model (default none)")
        model_group.add_argument('--min_first_layer_size', type=int, default=10, help="Minimum number of features in first hidden layer (default 10)")
        model_group.add_argument('--max_first_layer_size', type=int, default=1000, help="Maximum number of features in first hidden layer (default 1000)")
        model_group.add_argument('--min_second_layer_size', type=int, default=10, help="Minimum number of features in second hidden layer (default 10)")
        model_group.add_argument('--max_second_layer_size', type=int, default=1000, help="Maximum number of features in second hidden layer (default 1000)")

        features_group.add_argument('--feature_extractor', type=to_string_list, default="ChrisGoringe/vit-large-p14-vision-fp16", help="Model to use for feature extraction")
        features_group.add_argument('--hidden_states_used', type=to_int_list, default=None, help="Comma separated list of the hidden states to extract features from (0 is output layer, 1 is last hidden layer etc.)")
        features_group.add_argument('--hidden_states_mode', default="default", choices=["default", "join", "average", "weight"], help="Combine multiple layers from feature extractor by join (default), average, or weight")
        features_group.add_argument('--fp16_features', action="store_true", help="Store features in fp16")

        training_group.add_argument('--loss_model', default='mse', choices=['mse','ab','nll', 'wmse'], help="Loss model (default mse) (mse=mean square error, ab=ab ranking, nll=negative log likelihood, wmse=weighted mse )")
        training_group.add_argument('--set_for_scoring', default='eval', choices=['eval', 'full', 'train'], help="Image set to be used for scoring a model when trained (default eval)")
        training_group.add_argument('--metric_for_scoring', choices=['mse', 'ab', 'nll', 'wmse', 'spearman', 'pearson', 'accuracy'], help="Metric to be used for scoring a model when trained (default is the loss_model)")
        training_group.add_argument('--calculate_ab', action="store_true", help="Calculate ab even if not being used for scoring")
        training_group.add_argument('--calculate_mse', action="store_true", help="Calculate mse even if not being used for scoring")
        training_group.add_argument('--calculate_spearman', action="store_true", help="Calculate spearman")
        training_group.add_argument('--calculate_pearson', action="store_true", help="Calculate pearson")
        training_group.add_argument('--calculate_accuracy', action="store_true", help="Calculate accuracy (fraction correct side of accuracy_divider)")
        training_group.add_argument('--accuracy_divider', type=float, help="Divider between 'high' and 'low' for accuracy. If not specified the median score is used.")
        training_group.add_argument('--normalise_weights', action="store_true", help="If using wmse, normalise the weights to mean of 1.0")
        
        training_group.add_argument('--fraction_for_eval', type=float, default=0.25, help="fraction of images to be reserved for eval (default 0.25)")
        training_group.add_argument('--eval_pick_seed', type=int, default=42, help="Seed for random numbers when choosing eval images (default 42)") 
        training_group.add_argument('--ignore_existing_split', action="store_true", help="Discard existing train/eval split")

        metaparameter_group.add_argument('--name', help="Name prefix for Optuna")
        metaparameter_group.add_argument('--trials', type=int, default=200, help="Number of metaparameter trials" )
        metaparameter_group.add_argument('--sampler', default="CmaEs", choices=['CmaEs', 'random', 'QMC'], help="Metaparameter search algorithm")
        metaparameter_group.add_argument('--min_train_epochs', type=int, default=5, help="(default 5)")
        metaparameter_group.add_argument('--max_train_epochs', type=int, default=50, help="(default 50)")
        metaparameter_group.add_argument('--min_warmup_ratio', type=float, default=0.0, help="(default 0.0)")
        metaparameter_group.add_argument('--max_warmup_ratio', type=float, default=0.2, help="(default 0.2)")
        metaparameter_group.add_argument('--min_log_lr', type=float, default=-4.5, help="(default -4.5)")
        metaparameter_group.add_argument('--max_log_lr', type=float, default=-2.5, help="(default -2.5)")
        metaparameter_group.add_argument('--min_log_weight_lr', type=float, default=-4.5, help="(default -4.5)")
        metaparameter_group.add_argument('--max_log_weight_lr', type=float, default=-2.5, help="(default -2.5)")
        metaparameter_group.add_argument('--min_batch_size', type=int, default=1, help="(default 1)")
        metaparameter_group.add_argument('--max_batch_size', type=int, default=100, help="(default 100)")    
        metaparameter_group.add_argument('--min_dropout', type=float, default=0.0, help="Minimum dropout between two hidden layers (default 0.0)")
        metaparameter_group.add_argument('--max_dropout', type=float, default=0.8, help="Maximum dropout between two hidden layers (default 0.8)")
        metaparameter_group.add_argument('--min_input_dropout', type=float, default=0.0, help="Minimum dropout between features and first hidden layer (default 0.0)")
        metaparameter_group.add_argument('--max_input_dropout', type=float, default=0.8, help="Maximum dropout between features and first hidden layer (default 0.8)")
        metaparameter_group.add_argument('--min_output_dropout', type=float, default=0.0, help="Minimum dropout before final projection (default 0.0)")
        metaparameter_group.add_argument('--max_output_dropout', type=float, default=0.0, help="Maximum dropout before final projection (default 0.0)")

        into={}
        namespace, unknowns = parser.parse_known_args()
        if unknowns: print(f"\nIgnoring unknown argument(s) {unknowns}")

        d = vars(namespace)
        into[":Arguments (specified or default)"]=None
        for argument in d: into[argument] = d[argument]

        into[":Derived arguments"]=None

        for argument in list(a[4:] for a in into if a.startswith('min_')):
            into[argument] = d[f"min_{argument}"] if d[f"min_{argument}"] == d[f"max_{argument}"] else (d[f"min_{argument}"], d[f"max_{argument}"])
        if into['hidden_states_mode']!='weight': into['log_weight_lr'] = 0

        into['metric_for_scoring'] = into.get('metric_for_scoring', None) or into['loss_model']
        into['parameter_for_scoring'] = f"{into['set_for_scoring']}_{into['metric_for_scoring']}"
        into['measures'] = list(o for o in ['ab', 'mse', 'wmse', 'spearman', 'pearson', 'accuracy'] if o==into['loss_model'] or o==into['metric_for_scoring'] or into.get(f"calculate_{o}",False))

        into['save_model_path'] = os.path.join(into['directory'], into['model'])
        into['score_direction'] = 'maximize' if into['metric_for_scoring'] in ['ab', 'spearman', 'pearson', 'accuracy', ] else 'minimize'
        into['output_channels'] = 2 if into['loss_model']=='nll' else 1

        into[":Set arguments"]=None
        into["training_args"] = training_args

        return into

    def __init__(self):
        self.args = self._parse_arguments()
        self.show_args()
        self.validate()
        self.arg_sets = {   "feature_extractor_extras" : ['hidden_states_used','hidden_states_mode','fp16_features',],
                            "aesthetic_model_extras" : ['hidden_states_used','hidden_states_mode','model_seed','dropouts','layers','output_channels',],
                            "trainer_extras" : ['weight_learning_rate'],
                        }

        for k in list(self.args): 
            if k.startswith(':'): self.args.pop(k)
        
    def __getattr__(self, attr):
        return self.get(attr)
    
    def get(self, attr, default=Exception()):
        if attr in self.args: return self.args[attr]
        if attr in self.arg_sets: return { x : self.args[x] for x in self.arg_sets[attr] }
        if isinstance(default, Exception): raise KeyError(attr)
        return default
    
    def set(self, key, value):
        self.args[key] = value
    
    def show_args(self):
        for a in self.args: 
            if a.startswith(':'):
                print(f"\n{a[1:]}\n")
            else:
                print("{:>30} : {:<40}".format(a, str(self.get(a))))

    def validate(self):
        if not os.path.isdir(self.directory): 
            raise ArgumentException( f"{self.directory} doesn't exist or isn't a directory" )
        if not os.path.exists(os.path.join(self.directory, self.scores)):
            raise ArgumentException(f"{os.path.join(self.directory, self.scores)} not found")

    @property
    def keys(self):
        for a in self.args: yield a

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