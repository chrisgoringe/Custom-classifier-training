args = {
    # "train", "evaluate", or "spotlight"  (spotlight requires 'pip install spotlight')
    # or 'train,evaluate' which does a train, saves, and then evaluates the newly saved model
    "mode"                      : "train,evaluate",

    # the base model (automatically downloaded if required)   patch16-384
    # google/vit-base-patch16-224  and google/efficientnet-b5 (or b0...b7) are good ones to try
    "base_model"                : "google/vit-base-patch16-224",    

    # if restarting a previous run, this is the folder to load from. If None or '', the base_model is used. Required for evaluate or spotlight
    "load_model"                : "",     

    # folder to save the resulting model in. Required for training. 
    "save_model"                : "model_out",

    # path to the top level image directory, which should contain one subdirectory per image class, named for the image class
    "top_level_image_directory" : "images", 

    # what fraction of images to reserve as test images (not used in training), and a random seed for picking them
    "fraction_for_test"         : 0.25,
    "test_pick_seed"            : 42,                    

    # when doing "evaluate", just evaluate the test images? This is always the case when doing evals during training.
    "evaluate_test_only"        : True, 

    # weight the loss by the inverse of the number of images in each category?
    "weight_category_loss"      : True,

    # experimental - leave False!
    "transfer_learning": False,
    "restart_layers": 0,
    "thaw_layers": 7,
}

# The most common training arguments. There are 101 arguments available
# see https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
training_args = {
    "num_train_epochs"              : 10,
    "learning_rate"                 : 1e-4,
    
    # these ones will depend on architecture and memory
    "per_device_train_batch_size"   : 32,   
    "gradient_accumulation_steps"   : 1,  
    "per_device_eval_batch_size"    : 128,

    # save during the run? 'epoch' (every epoch) or 'steps', and maximum number to keep
    "save_strategy"                 : "no",
    "save_steps"                    : 200,
    "save_total_limit"              : 4,

    # evaluate during the run? set to "no" to speed up slightly, 'epoch' to get updates
    "evaluation_strategy"           : "epoch",
    "eval_steps"                    : 200,

    # Choose the best model, not the last model, to keep at the end. Requires save_strategy and evaluation_strategy to be the same (and not "no"). 
    "load_best_model_at_end"        : False,    
}

evaluation_args = {
}
