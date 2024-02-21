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
python -m pip install -r requirements.txt                                      
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

The command line arguments are kept in a text file. Copy the example `arguments-example.txt` to `arguments.txt` and edit it appropriately, then

`python aesthetic_predictor_training.py "@arguments.txt"`

You can see all the command line arguments available by running `python aesthetic_predictor_training.py --help`; they are all also described in the `arguments-example.txt` file.

More details of some of them are below.

### Defining the model architecture parameters

The base model takes the features from the feature extractor as a vector (typically 1024 or 1280 long). It then has two hidden layers, each consisting of (`nn.Dropout`, `nn.Linear`, `nn.RELU`) and then a final `nn.Linear` to project the last hidden layer to a single value. The size of the hidden layers is part of the metaparameter search, defined by the limits `--min_first_layer_size`, `--max_first_layer_size`, `--min_second_layer_size`, `--max_second_layer_size`.

### Feature extraction

The predictor can currently use any CLIP model, or the AIM feature extractor. The default is `ChrisGoringe/vitH16`, which is a 16 bit version of `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`.

Other important options are `openai/clip-vit-large-patch14` (used by SD1.5) and `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` (added by SDXL).

Other models tested include `apple/aim-600M,` `apple/aim-1B`, `apple/aim-3B`, `apple/aim-7B`, `laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`.

It is possible to use more than one model and have the features concatenated together. This currently isn't support at the command line.

Most image feature extractors provide access to multiple hidden layers, not just the last (using this is analagous to the `CLIP skip` in CLIP text processing). To access these, set `--hidden_states_used` to a comma separated list of integers specifying the layers to use. 0 represents the final output, 1 is the layer before it, 2 the layer before that, etc.. The layers do not need to be consecutive, so `--hidden_states_used=0,2,5` is fine.

By default these outputs are concatenated to produce a larger input to the model (so instead of 1024 features, with `--hidden_states_used=0,1` the model would receive 2048 features). Alternatively, with `--weight_hidden_states` the layers are be merged using a `nn.Linear` which gives each layer a single weight which is included in the training.

### Training constants

- `--loss_model` - `mse` (mean square error) is default, `ab` evaluates loss using MarginRankingLoss, `nll` (experimental) produces a value and estimated error and evaluates loss using negative log likelihood (`nn.GaussianNLLLoss`)

- `--set_for_scoring` and `--metric_for_scoring` are used by the metaparameter search to evaluate a model at the end of training. Set can be `full`, `train`, or `eval` portions, metric can be `mse` or `nll` (the loss values), `ab` (the percentage of pairs of images that are correctly ordered by the model), `spearman` (spearman rank correlation), `pearson` (pearson correlation), or `accuracy` (see below). Default is to use the specified loss model applied to the eval images only.

- `calculate_ab (mse, spearman, pearson, accuracy)` can be specified to calculate these metrics even if they are not being used for scoring. They are saved in the trial arguments (see Optuna Dashboard)

`accuracy` is the percentage of images that are placed on the correct side of a threshold ('good' v. 'bad'). By default the threshold is the median iamge score; you can choose a different value with `--accuracy_divider` 

### Metaparameters

Optuna is used to search through the metaparameter space; 

- `--name` to give the run a name
- `--trials` to set the number of trials in the run
- `--sampler` choice of sampler for Optuna metaparameter search

The metaparameter space consists of the model layer sizes (`first_layer_size` and `second_layer_size`) and training metaparameters ('`train_epochs`', '`warmup_ratio`', '`log_lr`, `batch_size`, `dropout`, `input_dropout`, `output_dropout`). Each is specified with `--min_xxx` and `--max_xxx` (which can be equal to specify a fixed value).

`input_dropout` applies between the feature extractor and the model; `dropout` applies between the two hidden layers of the model, `output_dropout` (default 0) applies after the second hidden layer of the model, before the final projection to a single value. 


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
