import os, torch

common_args = {
    # if restarting a previous run, this is the folder to load from. 
    "load_model"                : "",#r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output\training\aim1B_vit14.safetensors",     
    # folder to save the resulting model in. Required for training. 
    "save_model"                : r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output\training\vitH14-half.safetensors",
    # path to the top level image directory
    "top_level_image_directory" : r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output\training", 
    # the scores to train from
    "scorefile"                 : "image_scores.json",
}

aesthetic_model_args = {
    "clip_model" : [
        #"models/apple/aim-600M-half", 
        #"models/apple/aim-1B-half", 
        #"models/apple/aim-3B-half", 
        #"models/apple/aim-7B-half", 
        #"models/openai/clip-vit-large-patch14-half", 
        #"models/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k-half",
        "models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K-half",
        #"laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    ],
}

aesthetic_training_args = {
    # loss model. 'mse' or 'ranking'. 
    "loss_model"                : 'ranking',
    
    # what fraction of images to reserve as test images (when training), and a random seed for picking them
    "fraction_for_test"         : 0.25,
    "test_pick_seed"            : 42,        
}

metaparameter_args = {
    "meta_trials"       : 200,
    "sampler"           : "CmaEs",

    # If none of the arguments after this are ranges, there will only by one run

    # Each of these is a tuple (min, max) or a value.
    "num_train_epochs"   : (5, 50),
    "warmup_ratio"       : (0.0, 0.2),
    "log_learning_rate"  : (-3.5, -1.5),          
    "half_batch_size"    : (1, 50),            

    # A list, each element is either a tuple (min, max) or a value
    "dropouts"           : [ (0.0, 0.8), (0.0, 0.8), 0 ],
    "hidden_layers"      : [ (100, 1000), (4, 1000) ],
}

aesthetic_analysis_args = {
    "ab_analysis_regexes"       : [ "jib", "envy", "dig", "albedo" ],
    "use_model_scores_for_stats": False,
}

# training_args are passed directly into the TrainingArguments object.
# Below are the most common of the 101 arguments available
# see https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
#
training_args = {
    "lr_scheduler_type"             : "cosine",
    "gradient_accumulation_steps"   : 1,  
    "per_device_eval_batch_size"    : 128,

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