import os, re

common_args = {
    # if restarting a previous run, this is the folder to load from. 
    "load_model"                : r"",
    # folder to save the resulting model in. Required for training. 
    "save_model"                : r"training\mse.safetensors",
    # path to the top level image directory
    "top_level_image_directory" : r"training", 
    # the scores to train from
    "scorefile"                 : "image_scores.json",

    "model_scorefile"           : "model_scores.json",
    "error_scorefile"           : "error_scores.json",
    "splitfile"                 : "split.json",
}

# SDXL uses openai/clip-vit-large-patch14 and laion/CLIP-ViT-bigG-14-laion2B-39B-b160k 
# see https://github.com/huggingface/diffusers/blob/v0.26.2/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L149

# SD1.5 uses openai/clip-vit-large-patch14
# see https://github.com/huggingface/diffusers/blob/v0.26.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L312
aesthetic_model_args = {
    "clip_model" : [
        #"models/apple/aim-600M-half", 
        #"models/apple/aim-1B-half", 
        #"models/apple/aim-3B-half", 
        #"models/apple/aim-7B-half", 
        #"models/openai/clip-vit-large-patch14-half", 
        #"models/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k-half",
        #"models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K-half",
        #"laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
        "ChrisGoringe/vitH16"
    ],
}

aesthetic_model_extras = {
    "high_end_fix" : False,
    "variable_hef" : False,
}

aesthetic_training_args = {
    # loss model. 'nll', 'mse' or 'ranking'. nll is negative log likelihood, which returns prediction and a sigma
    "loss_model"                : 'mse',

    "parameter_for_scoring"     : 'eval_mse',   # We keep the best model, according to... [full|train|eval]_[ab|rmse|nll]  

    # set these next two to None by default
    "prune_bad_by"              : None,   # prune tests which are worse than the best so far by this much 
    "prune_bad_limit"           : None,   # prune tests worse than this as an absolute
    
    # what fraction of images to reserve as test images (when training), and a random seed for picking them
    "fraction_for_test"         : 0.25,
    "test_pick_seed"            : 42,        
}

trainer_extras = {
    "special_lr_parameters" : { re.compile("^parallel_blocks\.1\.*"): None },  # set by "delta_log_spec_lr"
} if aesthetic_training_args['loss_model']=='nll' else {}

metaparameter_args = {
    "name"              : "CmaEa200",     
    "meta_trials"       : 200,
    "sampler"           : "CmaEs",      # CmaEs, random, QMC.  CmaEs seems to work best

    # Each of these is a tuple (min, max) or a value.
    "num_train_epochs"   : (5, 50),
    "warmup_ratio"       : (0.0, 0.2),
    "log_learning_rate"  : (-3.5, -1.5),
    "half_batch_size"    : (1, 50),            

    # A list, each element is either a tuple (min, max) or a value
    "dropouts"           : [ (0.0, 0.8), (0.0, 0.8), 0 ],
    "hidden_layers"      : [ (100, 1000), (4, 1000) ],

    # again, this time for the error estimation - the delta is how different the lr is for the second network
    #"delta_log_spec_lr"  : (-3, 0),    
    #"dropouts_0"           : [ (0.0, 0.2), 0, 0 ],
    #"hidden_layers_0"      : [ (4,64), (4, 16) ],
}

aesthetic_analysis_args = {
    "ab_analysis_regexes"       : [  ],
}

# training_args are passed directly into the TrainingArguments object.
# Below are the most common of the 101 arguments available
# see https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
#
training_args = {
    "lr_scheduler_type"             : "cosine",
    "gradient_accumulation_steps"   : 1,  
    "per_device_eval_batch_size"    : 2000,     

    # save and evaluate during the run? 'epoch' (every epoch) or 'steps', and maximum number to keep
    "save_strategy"                 : "no",
    "save_steps"                    : 100,
    "save_total_limit"              : 4,
    "evaluation_strategy"           : "no",
    "eval_steps"                    : 100,
    "output_dir"                    : "out",

    # if save strategy and evaluation strategy are the same, can set this to True
    "load_best_model_at_end"        : False,  
}

# Calculated args
aesthetic_training_args['direction']='maximize' if aesthetic_training_args['loss_model']=='ranking' else 'minimize'

# Default values that get overwritten by any of the above - generally things that used to be options but really shouldn't be
class Args:
    args = { }

def get_args(aesthetic_training=False, aesthetic_analysis=False, aesthetic_model=False, show_training_args=True, show_args=True):
    for b, d in [(True, common_args),
                 (aesthetic_training, aesthetic_training_args),
                 (aesthetic_analysis, aesthetic_analysis_args),
                 (aesthetic_model,    aesthetic_model_args)]:
        if b:
            for k in d:
                Args.args[k] = d[k]    

    if show_args:
        for a in Args.args:
            print("{:>30} : {:<40}".format(a, str(args[a])))
    if show_training_args:
        for a in training_args:
            print("{:>30} : {:<40}".format(a, str(training_args[a])))

    for argument in ['load_model','save_model']:
        if argument in args and args[argument]:
            if os.path.isdir(args[argument]):
                if not os.path.exists(args[argument]): os.makedirs(args[argument])
                args[f"{argument}_path"]=os.path.join(args[argument],'model.safetensors')
            else:
                args[f"{argument}_path"]=args[argument]
        else:
            args[f"{argument}_path"]=None

class MetaRangeProcessor():
    def __init__(self):
        self.any_ranges = False

    def meta(self, mthd, label:str, rng:tuple|list):
        if isinstance(rng,(tuple,list)):
            self.any_ranges = True
            return mthd(label, *rng)
        else:
            return rng
        
    def meta_list(self, mthd, label:str, rnges:tuple|list):
        result = []
        for i, rng in enumerate(rnges):
            result.append(self.meta(mthd, f"{label}_{i}", rng))
        return result

args = Args.args
