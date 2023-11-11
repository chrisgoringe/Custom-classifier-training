args = {
    # "train", "evaluate", or "spotlight"  (spotlight requires 'pip install spotlight')
    "mode"                      : "train",

    # the base model (automatically downloaded if required) 
    # google/vit-base-patch16-224-in21k and google/efficientnet-b5 (or b0...b7) are good ones to try
    "base_model"                : "google/vit-base-patch16-224-in21k",    

    # if restarting a previous run, this is the folder to load from. If None or '', the base_model is used. Required for evaluate or spotlight
    "load_model"                : "",     

    # folder to save the resulting model in. Required for training.
    "save_model"                : "my_model",

    # path to the top level image directory, which should contain one subdirectory per image class, named for the image class
    "top_level_image_directory" : "path/to/my/images", 

    # what fraction of images to reserve as test images (not used in training), and a random seed for picking them
    "fraction_for_test"         : 0.25,
    "test_pick_seed"            : 42,                    

    # when doing "evaluate", just evaluate the test images?
    "evaluate_test_only"        : True, 
}

# The most common training arguments. There are 101 arguments available
# see https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
training_args = {
    "num_train_epochs"              : 5,
    "learning_rate"                 : 1e-5,
    
    # these ones will depend on architecture and memory
    "per_device_train_batch_size"   : 8,   
    "gradient_accumulation_steps"   : 4,  
    "per_device_eval_batch_size"    : 64,

    # save during the run? 'epoch' (every epoch) or 'steps', and maximum number to keep
    "save_strategy"                 : "no",
    "save_steps"                    : 200,
    "save_total_limit"              : 4,

    # evaluate during the run?
    "evaluation_strategy"           : "epoch",
    "eval_steps"                    : 200,

    # Choose the best model, not the last model, to keep at the end. Requires save_strategy and evaluation_strategy to be the same (and not "no"). 
    "load_best_model_at_end"        : False,    
}

evaluation_args = {
}

def check_arguments():
    if 'load_model' not in args or not args['load_model']:
        assert args['mode']=='train', "If not training, need to specify a model to reload!"
        args['load_model']=args['base_model']

    training_args['output_dir'] = args['save_model']

    if 'per_device_eval_batch_size' not in evaluation_args or not evaluation_args['per_device_eval_batch_size']:
        if 'per_device_eval_batch_size' in training_args and training_args['per_device_eval_batch_size']:
            evaluation_args['per_device_eval_batch_size'] = training_args['per_device_eval_batch_size']

    if args['mode']=='train':
        assert 'save_model' in args and args['save_model'], "Training needs a save_model location!"
