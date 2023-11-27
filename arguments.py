common_args = {
    # "", "train", "evaluate", "spotlight", "metasearch"
    "mode"                      : "metasearch",

    # the base model (automatically downloaded if required)   
    # google/vit-base-patch16-224  and google/efficientnet-b5 (or b0...b7) are good ones to try
    # for aesthetic, models/sac+logos+ava1-l14-linearMSE.safetensors
    "base_model"                : "models/sac+logos+ava1-l14-linearMSE.safetensors",

    # if restarting a previous run, this is the folder to load from. If None or '', the base_model is used. 
    "load_model"                : "training/aesthetic",     

    # folder to save the resulting model in. Required for training. 
    "save_model"                : "training/aesthetic",

    # path to the top level image directory
    "top_level_image_directory" : r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output\bbp1",#"a:/aesthetic/training", 

    # if there is a score.json file use this instead of the folder names for the score?
    "use_score_file"            : True,

    # what fraction of images to reserve as test images (not used in training), and a random seed for picking them
    "fraction_for_test"         : 0.25,
    "test_pick_seed"            : 42,        
    
    # evaluate test set at end of each n epochs (0 for 'don't') - only has meaning during training
    "eval_every_n_epochs"       : 10,    
}

category_training_args = {
    # weight the loss by the inverse of the number of images in each category? Not applied in aesthetic trainer
    "weight_category_loss"      : True,
}

aesthetic_training_args = {
    # loss model. 'mse' or 'ranking'. 
    "loss_model"                : 'ranking',

    # ignore unscored images when training?
    "ignore_score_zero"         : True,

    # aesthetic model dropouts - default dropouts are [0.2,0.2,0.1]. 
    "aesthetic_model_dropouts"  : [0.2,0.2,0.1],
}

aesthetic_model_args = {
    # The aesthetic model has no activators - this seems wrong to me. This inserts them.
    "aesthetic_model_relu"      : True,
}

aesthetic_ab_args = {
    # Number of images to show (2-8)
    "ab_image_count"            : 2,

    # How strongly to prefer images with fewer comparisons (0 = no weighting). Probability weighted by (1-lcw)^(comparisons)
    "low_count_weight"          : 0.3,

    # The size (height) of the window used by the aesthetic_ab_scorer script
    "ab_scorer_size"            : 600,

    # The maximum width/height of images (1 for square)
    "ab_max_width_ratio"        : 1, 

    # Load a model and evaluate the images with that as well? Only relevance is in the generation of stats.
    "use_model_scores_for_stats": True,

    # How many comparison runs?
    "max_comparisons"           : 100,
}

aesthetic_analysis_args = {
    # in AB scorer; optionally provide a list of regex strings; instead of running it will give statistics for images matching
    "ab_analysis_regexes"       : ['^3','^4','^5','^6','^7','^batch1','^one_stdev','^one_point_two'],
    "use_model_scores_for_stats": True,
}

# training_args are passed directly into the TrainingArguments object.
# Below are the most common of the 101 arguments available
# see https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
training_args = {
    "num_train_epochs"              : 30,
    "learning_rate"                 : 1e-4,
    "per_device_train_batch_size"   : 4,   

    "lr_scheduler_type"             : "cosine",
    "warmup_ratio"                  : 0.1,
    "gradient_accumulation_steps"   : 1,  
    "per_device_eval_batch_size"    : 128,

    # save during the run? 'epoch' (every epoch) or 'steps', and maximum number to keep
    "save_strategy"                 : "no",
    "save_steps"                    : 200,
    "save_total_limit"              : 4,

    # evaluate during the run? set to "no" to speed up slightly, 'epoch' to get updates
    "evaluation_strategy"           : "no",
    "eval_steps"                    : 200,

    # Choose the best model, not the last model, to keep at the end. Requires save_strategy and evaluation_strategy to be the same (and not "no"). 
    "load_best_model_at_end"        : False,    
}

# Default values that get overwritten by any of the above
class Args:
    args = {
        # If set to true, images from the score file with a score of zero are ignored
        # most likely to be useful when 
        "ignore_score_zero"         : False,
        # clip model used by aesthetic scorer (default 'ViT-L/14' is the one used for the pretrained model included)
        "clip_model"                : "ViT-L/14",

        "aesthetic_model_dropouts"  : [0.2,0.2,0.1],
    }

def get_args(category_training=False, aesthetic_training=False, aesthetic_ab=False, aesthetic_analysis=False, aesthetic_model=False, show_training_args=True):
    for b, d in [(True, common_args),
                 (category_training,category_training_args),
                 (aesthetic_training, aesthetic_training_args),
                 (aesthetic_ab, aesthetic_ab_args),
                 (aesthetic_analysis, aesthetic_analysis_args),
                 (aesthetic_model, aesthetic_model_args)]:
        if b:
            for k in d:
                Args.args[k] = d[k]    

    print("args:")
    for a in Args.args:
        print("{:>30} : {:<40}".format(a, str(args[a])))
    if show_training_args:
        print("training_args")
        for a in training_args:
            print("{:>30} : {:<40}".format(a, str(training_args[a])))

args = Args.args