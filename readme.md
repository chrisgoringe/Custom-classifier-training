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

There must also be a scorefile. A `.json` file can be generated using the image_ab_scorer from https://github.com/chrisgoringe/image_comparison to capture your personal preferences. 

For other sources of scoring, a `.csv` file is probably more convenient. The `.csv` file should have column names in the first row, and no excess whitespace. The columns must include `relative_path` and `score` (relative path being the location of the image relative to the `.csv` file). Optional columns are `split` (valid values are `eval` and `train`), `weight` (the weight to be given to this image when the loss model is `wmse` (weighted mean square error)). An additional column, `model_score` will be added when the `.csv` file is save (assuming `--savefile=filename.csv` is specified). Any other columns will be loaded and preserved. So a typical file might begin:

```csv
relative_path,score,split
images\241705.jpg,9.999043,train
images\118978.jpg,8.301219,eval
images\27695.jpg,4.544847,train
images\21913.jpg,5.989449,eval
images\34683.jpg,9.631905,train
```

## Configuration and running

The command line arguments are kept in a text file. Copy the example `arguments-example.txt` to `arguments.txt` and edit it appropriately, then

`python aesthetic_predictor_training.py @arguments.txt`

You can override the values in `arguments.txt` by adding arguments after it on the command line:

`python aesthetic_predictor_training.py @arguments.txt --server=daemon`

You can see all the command line arguments available by running `python aesthetic_predictor_training.py --help`; they are all also described in the `arguments-example.txt` file.

More details of some of them are below.

## Meaning of some of the available arguments

### Defining the model architecture parameters

The base model takes the features from the feature extractor as a vector (typically 1024 or 1280 long). It then has two hidden layers (each consisting of `nn.Dropout`, `nn.Linear`, `nn.RELU`), and then a final `nn.Dropout`, `nn.Linear` to project the last hidden layer to a single value. The size of the hidden layers is part of the metaparameter search, defined by the limits `--min_first_layer_size`, `--max_first_layer_size`, `--min_second_layer_size`, `--max_second_layer_size`.

### Feature extraction

The predictor can currently use any CLIP model, or any of the apple AIM feature extractors. A number of models have been reduced in size by converting them to fp16 and, in the case of CLIP, removing the text part. The following are available:

|name|original name|architecture|notes|
|-|-|-|-|
|ChrisGoringe/vit-large-p14-vision-fp16|openai/clip-vit-large-patch14|CLIPVisionModelWithProjection|SD1.5, 0.6GB, default|
|ChrisGoringe/bigG-vision-fp16|laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|CLIPVisionModelWithProjection|SDXL and Stable Cascade, 3.4GB|
|ChrisGoringe/aim-600M-fp16|apple/AIM-600M|AIM|1.3GB|
|ChrisGoringe/aim-1B-fp16|apple/AIM-1B|AIM|2.4GB|
|ChrisGoringe/aim-3B-fp16|apple/AIM-3B|AIM|5.3GB|
|ChrisGoringe/aim-7B-fp16|apple/AIM-7B|AIM|12.6GB|

The script `model_shrink.py` can be used (maybe) to reduce a CLIP or AIM model (but this is not maintained, documented, or recommended!)

It is possible to use more than one model and have the features concatenated together. However, this currently isn't support at the command line.

Most image feature extractors provide access to multiple hidden layers, not just the last (using this is analagous to the `CLIP skip` in CLIP text processing). To access these, set `--hidden_states_used` to a comma separated list of integers specifying the layers to use. 0 represents the final output, 1 is the layer before it, 2 the layer before that, etc.. The layers do not need to be consecutive, so `--hidden_states_used=0,2,5` is fine. If you don't specify `--hidden_states_used` the default for the model is used (0 for most models, multiple layers in AIM models).

The way multiple layers are combined is governed by `--hidden_states_mode` which can be:

- `default` - the default, which is `join` for CLIP models and `average` for AIM models
- `join` - outputs are concatenated to produce a larger input to the model - the number of features is multiplied by the number of layers used
- `average` - the layer outputs are averaged on a per-feature basis
- `weight` - the layers weighted-averaged using a weighting pre-processor; a `nn.Linear`, with the same weights used for each feature 


### Training constants

- `--loss_model` - `mse` (mean square error) is default, `ab` evaluates loss using MarginRankingLoss, `nll` (experimental) produces a value and estimated error and evaluates loss using negative log likelihood (`nn.GaussianNLLLoss`), `wmse` is mean square error, but with each image weighted according to the `weight` value from the scorefile.

- `--set_for_scoring` and `--metric_for_scoring` are used by the metaparameter search to evaluate a model at the end of training. Set determines the images used to score: `full`, `train`, or `eval`. Metric can be `mse`, `wmse`, or `nll` (the loss values); `ab` (the percentage of pairs of images that are correctly ordered by the model), `spearman` (spearman rank correlation), `pearson` (pearson correlation), or `accuracy` (see below). Default is to use the specified loss model with `set=eval`.

- `calculate_ab (mse, wmse, spearman, pearson, accuracy)` can be specified to calculate these metrics even if they are not being used for scoring. They are saved in the trial arguments (see Optuna Dashboard)

`accuracy` is the percentage of images that are placed on the correct side of a threshold ('good' v. 'bad'). By default the threshold is the median image score; you can choose a different value with `--accuracy_divider` 

### Metaparameters

Optuna is used to search through the metaparameter space. Use

- `--name` to give the run a name
- `--trials` to set the number of trials in the run
- `--sampler` choice of sampler for Optuna metaparameter search

The metaparameter space consists of the model layer sizes (`first_layer_size` and `second_layer_size`) and training metaparameters (`train_epochs`, `warmup_ratio`, `log_lr`, `batch_size`, `dropout`, `input_dropout`, `output_dropout`). Each is specified with `--min_xxx` and `--max_xxx` (which can be equal to specify a fixed value).

When `--hidden_states_mode==weight`, an additional metaparameter, `log_weight_lr` is available; it controls the learning rate for the weighting pre-processor.

`input_dropout` applies between the feature extractor and the model; `dropout` applies between the two hidden layers of the model, `output_dropout` (default 0) applies after the second hidden layer of the model, before the final projection to a single value. 

# Other Tools

## Monitoring training

Metaparameter results are saved in a local sqlite database, specified by `--database` (default `sqlite:///db.sqlite`). Set `--database=` (ie blank) to turn this off.

The trainer also launches an Optuna Dashboard that you can access on `http://127.0.0.1:8080/`. Control this with `--server={on,daemon,off}` (daemon will terminate with training).

## Spotlight

[Spotlight](https://github.com/Renumics/spotlight) is a gui for analysing datasets. After a training run, run `aesthetic_spotlight.py` to see its magic!

```
  -d DIRECTORY, --directory DIRECTORY
                        Top level directory
  --scores SCORES       Filename of scores file (default scores.csv)
```

## Data analysis

Script `aesthetic_data_analysis.py` can do statistical analysis on scorefiles. Specify the scorefile produced by a training run. The model results are compared with the 'true' scores.

```
  -d DIRECTORY, --directory DIRECTORY
                        Top level directory
  --scores SCORES       Filename of scores file (default scores.json)
  --include_train_split
                        Include training split in analysis (default is eval images only)
  --regex REGEX         Only include images matching this regex
  --directories         Perform separate analysis for each subdirectory
```

## Scorer

Script `aesthetic_scorer.py` scores the images in a directory into a scorefile (or stdout if `--outfile` not specified).

```
  -d DIRECTORY, --directory DIRECTORY
                        Top level directory
  -m MODEL, --model MODEL
                        Path to model
  -o OUTFILE, --outfile OUTFILE
                        Save to file (.csv) in directory
```

# Acknowledgements

Idea inspired by bmf (Đỗ Khang), who also provided lots of excellent discussion, feedback, and datasets!
