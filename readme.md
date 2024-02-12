# Custom Aesthetic Training

(latest update 10 Feb 2024)

Script to train (and evaluate) a custom aesthetic model for evaluating images according to personal taste (or another criteria).

# Install

In some suitable directory,
```
git clone https://github.com/chrisgoringe/custom-classifier-training
```

## Virtual environment

Any virtual environment that you use for torch stuff will probably work (for instance, if you run ComfyUI, use that one).

If creating a new one, first [install torch](https://pytorch.org/get-started/locally/).

To make sure you have everything
```              
pip install -r requirements.txt                                      
```

If you want to use the Apple feature extraction models (optional), then
```
pip install git+https://git@github.com/apple/ml-aim.git
```

# Training

## Setup your images

Create a directory, say `images`, as your `top_level_image_directory`. It can directly contain your images, or they can be placed in subdirectories.

There must also be a `score.json` file. This is best generated using the image_ab_scorer from https://github.com/chrisgoringe/image_comparison.

## Configuration and running

Edit `arguments.py`. You only really need to change `save_model` and `top_level_image_directory`.

Basic settings:

```python
    # if restarting a previous run (or using other tools). Normally "" for training. 
    "load_model"                : r"",
    # where to save the model. 
    "save_model"                : r"training4\model_trained_training4.safetensors",
    # path to the top level image directory
    "top_level_image_directory" : r"training4", 
    # the scores to train from
    "scorefile"                 : "scores.json",
    # three additional (optional) output files: the scores as predicted by the model, the errors (scores - model_scores), and the split (train/test)
    "model_scorefile"           : "model_scores.json",
    "error_scorefile"           : "error_scores.json",
    "splitfile"                 : "split.json",
}
```

Then just run `python aesthetic_predictor_training.py`.

Some other things you might want to change (see the notes in the arguments file)
- `clip_model` - used as a feature extractor
- `loss_model` - `mse` (mean square error) is default, `ab` evaluates loss using MarginRankingLoss
- `parameter_for_scoring` - choosing the best model based on `mse` or `ab`, and using the `full`, `train`, or `test` portions.

# Metaparameters

Model training is so fast that we do a metaparameter search instead of choosing. Below are the things you could tweak
```python
metaparameter_args = {
    "name"              : None,     # name used by Optuna Dashboard
    "meta_trials"       : 200,
    "sampler"           : "CmaEs",      # CmaEs, random, QMC.  CmaEs seems to work best

    # Each of these is a tuple (min, max) or a value.
    "num_train_epochs"   : (5, 50),
    "warmup_ratio"       : (0.0, 0.2),
    "log_learning_rate"  : (-3.5, -1.5),
    "half_batch_size"    : (1, 50),            

    # A list, where each element is either a tuple (min, max) or a value
    # dropouts apply before each hidden layer so should be one longer than hidden layers, but will be automatically padded with zeroes
    # number of hidden layers is fixed (by the length of hidden_layers below, so you can change it, but metaparameter search wont)
    "dropouts"           : [ (0.0, 0.8), (0.0, 0.8), 0 ],
    "hidden_layers"      : [ (10, 1000), (10, 1000), ],
}
```

# Monitoring training

In a new command line, enter `optuna-dashboard sqlite:///db.sqlite` to launch the optuna dashboard.

# Spotlight

[Spotlight](https://github.com/Renumics/spotlight) is a gui for analysing datasets. After a training run, run `aesthetic_spotlight.py` to see its magic!

# Data analysis

Script `aesthetic_data_analysis.py` can do spearman and ab analysis on scorefiles. Edit the Args class at the top:

```python
class Args:
    top_level_image_directory = r"training4"
    scorefile = "scores.json"
    splitfile = "split.json"
    test_split_only = True        # Only include files that were in the 'test' (or eval) split of the training

    load_and_run_model = True     # If True, specify the model in the next line
    model = r"training4\model.safetensors" # Relative to the script; all other files are relative to the top_level_image_directory
    save_model_score_and_errors = False  # If True, save model scores and errors in the next two files
    save_model_scorefile = "model_scores.json"
    save_error_scorefile = "error_scores.json"

    load_model_scorefile = False  # If not loading and running a model, can just load a scorefile
    model_scorefile = "model_scores.json"

    regexes = []    # Zero or more regexes (as strings to be compiled). The analysis will run on (subject to the test_split constraint)
                    # - all files 
                    # - for each subfolder, just the files in it
                    # - for each regex, just the files whose path matches the regex
```

For each test, the output is:
```
All :   201 images, db score  0.057 +/- 0.67, model score  0.178 +/- 0.54, spearman 0.6102 (p= 6.8e-22), pearson 0.5997 (p= 5.2e-21), AB  71.37%
```


# Acknowledgements

Idea inspired by bmf (Đỗ Khang), who also provided lots of excellent discussion, feedback, and datasets!
