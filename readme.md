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

`python aesthetic_predictor_training.py -d=DIRECTORY`

There are lots of other options. Here's the output of `python aesthetic_predictor_training.py --help`:

```
options:
  -h, --help            show this help message and exit

Main arguments:
  -d DIRECTORY, --directory DIRECTORY
                        Top level directory
  --model MODEL         Filename to save model (default model.safetensors)
  --scores SCORES       Filename of scores file (default scores.json)
  --errors ERRORS       Filename of errors file (default errors.json)
  --split SPLIT         Filename of split file (default split.json)

Defining the model architecture:
  --final_layer_bias    Train with a bias in the final layer
  --model_seed MODEL_SEED
                        Seed for initialising model (default none)
  --min_layer_size MIN_LAYER_SIZE
                        Minimum number of features in each hidden layer (default 10)
  --max_layer_size MAX_LAYER_SIZE
                        Maximum number of features in each hidden layer (default 1000)

Feature extraction:
  --feature_extractor_model FEATURE_EXTRACTOR_MODEL
                        Model to use for feature extraction
  --hidden_states HIDDEN_STATES
                        Comma separated list of the hidden states to concatenate (0 is output layer, 1 is last hidden layer etc.)
  --weight_n_output_layers WEIGHT_N_OUTPUT_LAYERS
                        Add a trainable projection of last n output layers to the start of the model
  
Training constants:
  --loss_model {mse,ab,nll}
                        Loss model (default mse) (mse=mean square error, ab=ab ranking, nll=negative log likelihood)
  --set_for_scoring {eval,full,train}
                        Image set to be used for scoring a model when trained (default eval)
  --metric_for_scoring {mse,ab,nll,spearman}
                        Metric to be used for scoring a model when trained (default is the loss_model)
  --calculate_ab        Calculate ab even if not being used for scoring
  --calculate_mse       Calculate mse even if not being used for scoring
  --calculate_spearman  Calculate spearman even if not being used for scoring
  --fraction_for_eval FRACTION_FOR_EVAL
                        fraction of images to be reserved for eval (aka eval) (default 0.25)
  --eval_pick_seed EVAL_PICK_SEED
                        Seed for random numbers when choosing eval images (default 42)

Metaparameters:
  --name NAME           Name prefix for Optuna
  --trials TRIALS       Number of metaparameter trials
  --sampler {CmaEs,random,QMC}
                        Metaparameter search algorithm
  --min_train_epochs MIN_TRAIN_EPOCHS
                        (default 5)
  --max_train_epochs MAX_TRAIN_EPOCHS
                        (default 50)
  --min_warmup_ratio MIN_WARMUP_RATIO
                        (default 0.0)
  --max_warmup_ratio MAX_WARMUP_RATIO
                        (default 0.2)
  --min_log_lr MIN_LOG_LR
                        (default -4.5)
  --max_log_lr MAX_LOG_LR
                        (default -2.5)
  --min_batch_size MIN_BATCH_SIZE
                        (default 1)
  --max_batch_size MAX_BATCH_SIZE
                        (default 100)
  --min_dropout MIN_DROPOUT
                        (default 0.0)
  --max_dropout MAX_DROPOUT
                        (default 0.8)
  ```

### Defining the model architecture parameters

The base model takes the features from the feature extractor as a vector (typically 1024 or 1280 long). It then has two hidden layers, each consisting of (`nn.Dropout`, `nn.Linear`, `nn.RELU`) and then a final `nn.Linear` to project the last hidden layer to a single value. By default this last step has no bias, this can be set with `--final_layer_bias`. The size of the hidden layers is part of the metaparameter search, defined by the limits `--min_layer_size` and `--max_layer_size`.

### Feature extraction

The predictor can currently use any CLIP model, or the AIM feature extractor. The default is `ChrisGoringe/vitH16`, which is a 16 bit version of `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`.

Other important options are `openai/clip-vit-large-patch14` (used by SD1.5) and `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` (added by SDXL).

Other models tested include `apple/aim-600M,` `apple/aim-1B`, `apple/aim-3B`, `apple/aim-7B`, `laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`.

It is possible to use more than one model and have the features concatenated together. This currently isn't support at the command line.

Most image feature extractors provide access to multiple hidden layers, not just the last (using this is analagous to the `CLIP skip` in CLIP text processing). These can be accessed in one of two (mututally exclusive) ways:

- `--hidden_states` takes a comma-separated list of integers, will concatenate the features from the specified layers (0 = usual output, 1 = previous layer, etc). This can be used to pick a hidden layer (eg `--hidden_states=1`) or to train using more than one (eg `--hidden_states=0,1`).

- `--weight_n_output_layers=n` adds a `nn.Linear` to the model which combines the features from the last `n` layers using a single set of `n` trainable weights

### Training constants

- `--loss_model` - `mse` (mean square error) is default, `ab` evaluates loss using MarginRankingLoss, `nll` (experimental) produces a value and estimated error and evaluates loss using negative log likelihood (`nn.GaussianNLLLoss`)

- `--set_for_scoring` and `--metric_for_scoring` are used by the metaparameter search to evaluate a model at the end of training. Set can be `full`, `train`, or `eval` portions, metric can be `mse` or `nll` (the loss values), `ab` (the percentage of pairs of images that are correctly ordered by the model) or `spearman` (spearman rank comparison). Default is to use the specified loss model applied to the eval images only.

- `calculate_ab (mse, spearman)` can be specified to calculate these metrics even if they are not being used for scoring. They are saved in the trial arguments (see Optuna Dashboard)

### Metaparameters

Model training is so fast that we do a metaparameter search...

- `--name` to give the run a name
- `--trials` to set the number of trials in the run
- `--sampler` choice of sampler for Optuna metaparameter search
- `--[min|max]_[train_epochs|warmup_ratio|log_lr|batch_size|dropout]` along with `layer_size` these are the metaparameters varied between trials. 

# Monitoring training

In a new command line, enter `optuna-dashboard sqlite:///db.sqlite` to launch the optuna dashboard.

# Spotlight

[Spotlight](https://github.com/Renumics/spotlight) is a gui for analysing datasets. After a training run, run `aesthetic_spotlight.py` to see its magic!

# Data analysis

Script `aesthetic_data_analysis.py` can do spearman and ab analysis on scorefiles. It will ignore any command line parameters it doesn't recognise (meaning you can generally use the same set of command line parameters as used for training...). Specify the image directory and either a model that you have trained, or a scorefile produced by a model. The model results are compared with the 'true' scores.

```
  -d DIRECTORY, --directory DIRECTORY
                        Top level directory
  --scores SCORES       Filename of scores file (default scores.json)
  -m MODEL, --model MODEL
                        Model (if any) to load and run on images (default is not to load a model)
  --model_scores MODEL_SCORES
                        Filename of model scores file (ignored if model is specified)
  --split SPLIT         Filename of split file (default split.json)
  --include_train_split
                        Include training split in analysis
  --save_scores_and_errors
                        Save score and error files from running model
  --save_model_scorefile SAVE_MODEL_SCOREFILE
  --save_error_scorefile SAVE_ERROR_SCOREFILE
```

For each test, the output is:
```
All :   201 images, db score  0.057 +/- 0.67, model score  0.178 +/- 0.54, spearman 0.6102 (p= 6.8e-22), pearson 0.5997 (p= 5.2e-21), AB  71.37%
```

# Acknowledgements

Idea inspired by bmf (Đỗ Khang), who also provided lots of excellent discussion, feedback, and datasets!
