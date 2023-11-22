args = {
    # "train", "evaluate", or "spotlight"  (spotlight requires 'pip install spotlight')
    # or 'train,evaluate' which does a train, saves, and then evaluates the newly saved model
    # (for aesthetic, always does train and evaluate, spotlight can be added here)
    "mode"                      : "train,evaluate",

    # with mode=='meta', use these metaparameters (list of values to permute through)
    "meta_epochs" : [50,100,200],
    "meta_lr"     : [1e-3,3e-3,1e-2],
    "meta_batch"  : [4,8,16],

    # the base model (automatically downloaded if required)   
    # google/vit-base-patch16-224  and google/efficientnet-b5 (or b0...b7) are good ones to try
    # for aesthetic, models/sac+logos+ava1-l14-linearMSE.safetensors
    "base_model"                : "models/sac+logos+ava1-l14-linearMSE.safetensors",    

    # if restarting a previous run, this is the folder to load from. If None or '', the base_model is used. Required for evaluate or spotlight
    "load_model"                : "training/aesthetic",     

    # folder to save the resulting model in. Required for training. 
    "save_model"                : "training/aesthetic",

    # path to the top level image directory
    "top_level_image_directory" : "C:\\Users\\chris\\Documents\\GitHub\\ComfyUI_windows_portable\\ComfyUI\\output\\compare", 

    # if there is a score.json file use this instead of the folder names for the score?
    "use_score_file" : True,

    # what fraction of images to reserve as test images (not used in training), and a random seed for picking them
    "fraction_for_test"         : 0.25,
    "test_pick_seed"            : 42,        

    # evaluate test set at end of each n epochs (0 for 'don't')
    "eval_every_n_epochs"       : 10,            

    # weight the loss by the inverse of the number of images in each category? Not applied in aesthetic trainer
    "weight_category_loss"      : True,

    # loss model. Currently only 'mse' or 'ranking'. Only used for aesthetic trainer
    "loss_model" : 'ranking',

    # aesthetic model dropouts - default dropouts are [0.2,0.2,0.1]. 
    "aesthetic_model_dropouts"  : [0.2,0.2,0.1],

    # The aesthetic model has no activators - this seems wrong to me. This inserts them.
    "aesthetic_model_relu"      : True,

    # The size (height) of the window used by the aesthetic_ab_scorer script
    "ab_scorer_size"            : 600,
}

# The most common training arguments. There are 101 arguments available
# see https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
training_args = {
    # number of steps = images * epochs / batch
    # rule of thumb: steps * learning_rate = 10   (images ~ 1000)
    # 4e-4, batch 2, 50 epochs

    "num_train_epochs"              : 100,
    "learning_rate"                 : 4e-3,

    #
    "lr_scheduler_type" : "cosine",
    "warmup_ratio" : 0.1,
    
    # these ones will depend on architecture and memory
    "per_device_train_batch_size"   : 8,   
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

evaluation_args = {
}
