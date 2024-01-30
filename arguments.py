import os 

common_args = {
    # "", "train", "evaluate", "spotlight", "metasearch"
    "mode"                      : "metasearch",

    # if restarting a previous run, this is the folder to load from. 
    "load_model"                : "",     

    # folder to save the resulting model in. Required for training. 
    "save_model"                : r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output\training",

    # path to the top level image directory
    "top_level_image_directory" : r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output\training", 
}

aesthetic_model_args = {
    "clip_model"                : "apple/aim-3B", # ["openai/clip-vit-large-patch14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"], 
    "hidden_layers"             : [ 256, 64 ],
}

aesthetic_training_args = {
    # aesthetic model dropouts - before each hidden layer (and after the last hidden layer). Padded at end wwith zeroes if needed
    "dropouts"                  : [ 0.6, 0.3, 0.0 ],

    # loss model. 'mse' or 'ranking'. 
    "loss_model"                : 'ranking',

    # if doing metaparameter (mode=metasearch), how many trials?
    "meta_trials"               : 50,

    # the scores to train to
    "use_score_file"            : "image_scores.json",

    # what fraction of images to reserve as test images (when training), and a random seed for picking them
    "fraction_for_test"         : 0.25,
    "test_pick_seed"            : 42,        
    
    # evaluate test set at end of each n epochs (0 for 'don't') - only has meaning during training
    "eval_every_n_epochs"       : 10,    
}

aesthetic_analysis_args = {
    "ab_analysis_regexes"       : [],
    "use_model_scores_for_stats": False,
}

# training_args are passed directly into the TrainingArguments object.
# Below are the most common of the 101 arguments available
# see https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
#
training_args = {
    # these are ignore if mode=metasearch
    "num_train_epochs"              : 50,
    "learning_rate"                 : 1e-4,
    "per_device_train_batch_size"   : 10,   
    "warmup_ratio"                  : 0.1,

    "lr_scheduler_type"             : "cosine",
    "gradient_accumulation_steps"   : 1,  
    "per_device_eval_batch_size"    : 128,

    # save during the run? 'epoch' (every epoch) or 'steps', and maximum number to keep
    "save_strategy"                 : "no",
    "save_steps"                    : 100,
    "save_total_limit"              : 4,

    # evaluate during the run? set to "no" to speed up slightly, 'epoch' to get updates
    "evaluation_strategy"           : "no",
    "eval_steps"                    : 100,

    # show logging messages during training? "steps" is default
    "logging_strategy"              : "no",
    "logging_dir"                   : "log",
    "output_dir"                    : "out",

    # Choose the best model, not the last model, to keep at the end. Requires save_strategy and evaluation_strategy to be the same (and not "no"). 
    "load_best_model_at_end"        : False,    
}

category_training_args = {
    # weight the loss by the inverse of the number of images in each category? Not applied in aesthetic trainer, so ignore it!
    "weight_category_loss"      : True,
    # the base model (automatically downloaded if required)   
    # google/vit-base-patch16-224  and google/efficientnet-b5 (or b0...b7) are good ones to try
    "base_model"                : "",
}

# Default values that get overwritten by any of the above - generally things that used to be options but really shouldn't be
class Args:
    args = {
        # If set to true, images from the score file with a score of zero are ignored
        "ignore_score_zero"         : False,
    }

def get_args(category_training=False, aesthetic_training=False, aesthetic_analysis=False, aesthetic_model=False, show_training_args=True, show_args=True):
    for b, d in [(True, common_args),
                 (category_training,category_training_args),
                 (aesthetic_training, aesthetic_training_args),
                 (aesthetic_analysis, aesthetic_analysis_args),
                 (aesthetic_model, aesthetic_model_args)]:
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


args = Args.args