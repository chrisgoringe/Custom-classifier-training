common_args = {
    # "", "train", "evaluate", "meta", "spotlight", "analysis", "metasearch"
    "mode"                      : "metasearch",

    # the base model (automatically downloaded if required)   
    # google/vit-base-patch16-224  and google/efficientnet-b5 (or b0...b7) are good ones to try
    # for aesthetic, models/sac+logos+ava1-l14-linearMSE.safetensors
    "base_model"                : "models/sac+logos+ava1-l14-linearMSE.safetensors",    

    # if restarting a previous run, this is the folder to load from. If None or '', the base_model is used. Required for evaluate or spotlight
    "load_model"                : "",     

    # folder to save the resulting model in. Required for training. 
    "save_model"                : "training/aesthetic",

    # path to the top level image directory
    "top_level_image_directory" : "a:/aesthetic/training", 

    # if there is a score.json file use this instead of the folder names for the score?
    "use_score_file" : True,

    # what fraction of images to reserve as test images (not used in training), and a random seed for picking them
    "fraction_for_test"         : 0.25,
    "test_pick_seed"            : 42,        

    # evaluate test set at end of each n epochs (0 for 'don't')
    "eval_every_n_epochs"       : 10,            
}

category_training_args = {
    # weight the loss by the inverse of the number of images in each category? Not applied in aesthetic trainer
    "weight_category_loss"      : True,
}

aesthetic_training_args = {
    # with mode=='meta', use these metaparameters (list of values to permute through)
    "meta_epochs" : [125,150,175],
    "meta_lr"     : [1e-4],
    "meta_batch"  : [128,256],

    # format string for meta.csv (epochs,lr,batch,train_loss,eval_loss,train_ab,eval_ab,time)
    "meta_fmt"    : "{:>4},{:>8.2},{:>3},{:>8.4f},{:>8.4f},{:>6.4f},{:>6.4f},{:>6.1f}",

    # loss model. 'mse' or 'ranking'. 
    "loss_model"                : 'ranking',

    # ignore unscored images when training?
    "ignore_score_zero"         : True,
}

aesthetic_model_args = {
    # aesthetic model dropouts - default dropouts are [0.2,0.2,0.1]. 
    "aesthetic_model_dropouts"  : [0.2,0.2,0.1],

    # The aesthetic model has no activators - this seems wrong to me. This inserts them.
    "aesthetic_model_relu"      : True,
}

aesthetic_ab_args = {
    # The size (height) of the window used by the aesthetic_ab_scorer script
    "ab_scorer_size"            : 600,
    "ignore_score_zero"         : False,
    "load_model"                : "training/aesthetic", 
    "use_model_scores_for_stats": True,
    "max_comparisons"           : 100,
}

aesthetic_analysis_args = {
    # in AB scorer; optionally provide a list of regex strings; instead of running it will give statistics for images matching
    "ab_analysis_regexes"       : ['^3','^4','^5','^6','^7','^batch2','^batch3','^batch4'],
    "ignore_score_zero"         : True,
    "use_model_scores_for_stats": True,
}

# The most common training arguments. There are 101 arguments available
# see https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
training_args = {
    # number of steps = images * epochs / batch
    # rule of thumb: steps * learning_rate = 10   (images ~ 1000)
    # 4e-4, batch 2, 50 epochs

    "num_train_epochs"              : 20,
    "learning_rate"                 : 2e-4,

    #
    "lr_scheduler_type" : "cosine",
    "warmup_ratio" : 0.1,
    
    # these ones will depend on architecture and memory
    "per_device_train_batch_size"   : 12,   
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

class Args:
    args = {}

def get_args(category_training=False, aesthetic_training=False, aesthetic_ab=False, aesthetic_analysis=False, aesthetic_model=False):
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
    print("trainging_args")
    for a in training_args:
        print("{:>30} : {:<40}".format(a, str(training_args[a])))

args = Args.args