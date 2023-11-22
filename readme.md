# Custom Classifier and Custom Aesthetic Training

Script to train (and evaluate) a custom image classifier in a way that is suitable for use with the Custom Classifier ComfyUI nodes (coming soon!).

They are generic models, so should work for any other use for that matter.

## Install

In some suitable directory,
```
git clone https://github.com/chrisgoringe/custom-classifier-training
pip install git+https://github.com/openai/CLIP.git for aesthetic models
```

## Virtual environment

Any virtual environment that you use for torch stuff will probably work (for instance, if you run ComfyUI, use that one).

If creating a new one, first [install torch](https://pytorch.org/get-started/locally/), then 
```
pip install -r requirements.txt
```

# Custom Classifier

## Setup your images

Create a subdirectory, say `images`, as your `top_level_image_directory`. In it create one subdirectory per category, named for the category. So, for instance, `images/good` and `images/bad`.

Folders and files that start with `.` are ignored; folders which contain no images are ignored. Images are assumed to have extension `.png`, `.jpg` or `.jpeg`. Edit `src/data_holder.py` line 8 if you have other image extensions.

## Configuration and running

Edit `arguments.py`. The comments should explain everything. Then just run `python custom_classifier_training.py`.

Basic settings:

```python
    "mode"                      : "train,evaluate",
    "base_model"                : "google/vit-base-patch16-224-in21k",  # or see file for options
    "load_model"                : "",          # unless restarting, in which case 'my_model'
    "save_model"                : "my_model",  # the path to a folder used to save
    "top_level_image_directory" : "path/to/images", 
```

Read through arguments.py to see other things you can change, it isn't very long.

## Pretrained models 

This script is for customising pretrained models (`base_model` in the configuration). It's been tested with `google/vit-base-patch16-224-in21k` and `google/efficientnet-b5` and should work with the other efficientnets (b0...b7 in increasing size and complexity) and ViT architectures. 

It might well work with any transformers based model. 

---

# Custom Aesthetic

## Setup your images

Create a subdirectory, say `images`, as your `top_level_image_directory`. In it you can:

*Either* create a set of subdirectories with integer value names ('1', '2' etc) each of which contains images rated with that score, 

*Or* create one or more arbitrary subdirectories of images, and provide a `score.json` file (which can be created by the human_aesthetic_score.py script).

## Configuration and running

Edit `arguments.py`. The comments should explain everything. Then just run `python aesthetic_predictor_training.py`.

Basic settings:

```python
    "mode"                      : "",  # "meta" for metaparameter search, "spotlight" to use that tool after training
    "base_model"                : "models/sac+logos+ava1-l14-linearMSE.safetensors",  # or "" to start from scratch
    "load_model"                : "",                # unless restarting, in which case 'my_model'
    "save_model"                : "my_model",        # the path to a folder used to save
    "top_level_image_directory" : "path/to/images", 
    "loss_model"                : 'ranking',   # evaluate model based on how well it ranks pairs (relative values), 
                                               # or 'mse' for absolute mean square error (absolute values)
```

Read through arguments.py to see other things you can change, it isn't very long.

## Your aesthetic ratings: score.json

The `score.json` file takes the form `filename:score`, plus an (optional) `#meta#` like this:
```json
{
  "3\\betterbeyondneko_v10_00003_.png": -0.4118409741404654,
  "3\\betterbeyondneko_v10_00026_.png": 0.25,
  ...
  "#meta#": {
    "evaluations": 1007,
    "model_evals": 123
  }
}
```
The filename is relative to the directory where the score.json file is located, the score is roughly based on a normal distribution about zero, width one.

## Manual AB comparisons

In order to provide your aesthetic preferences, you need to rate images to create the `score.json` file.

You can do this using the `aesthetic_ab_scorer.py` script. This will randomly choose two images for you to choose which you prefer. Press '1' for the left image or '2' for the right image, or 'q' to quit. When you quit some statistics about the dataset will be shown in the console (press 'r' anytime to see these stats). 

- `x images in   yy.y s`  (how many have you just rated how quickly)
- `123/1208 are rated zero`  (how many images have a rating of zero - probably never compared)
- `x choices matched prediction,  y contradicted prediction [  z not predicted] = (aa.bb%)` how often was your choice consistent with the rankings in the database so far. As you continue to train, this value should increase (but in my experience tends to cap out around 75% due to inconsistency and/or non-linearity of preference)
- `A total of   x comparisons have been made for  y images ( z per image)` you probably want at least a few comparisons per image (note z=2x/y because each comparison rates two image)

The score.json file is reloaded each time you run the script, so you can evaluate your images in short bursts - you can also add new images between runs, and they will be added (with a score of 0).

In arguments.py set
```python
    "top_level_image_directory" : "C:\\Users\\chris\\Documents\\GitHub\\ComfyUI_windows_portable\\ComfyUI\\output\\compare", 
    "ab_scorer_size"            : 600,  # the height of the app on the screen
```

then `python aesthetic_ab_scorer.py`

## Initialise new images from model

You can initialise the `score.json` file using `aesthetic_score_from_model.py`. The script will look in the specified directory and load the score.json file, add any new images, and then calculate a score for any image which doesn't have one. You can do this with a model you have trained (probably best) or use the default `models/sac+logos+ava1-l14-linearMSE.safetensors` model (to initialise the rating of images by a generic aesthetic value).

Because the model is trained by ranking, the scores produced by the model are processed as follows:

- the standard deviation of the existing image data is calculated (or taken as 0.5 if there is no existing manual data)
- *all* images are evaluated
- values returned by model are shifted so that the mean is zero
- the values for new images are passed through a tanh() function (squash extremes)
- the values for new images are scaled to have the same standard deviation as the existing image data

After doing this, you will want to use the AB comparison to give your own opinions (otherwise you end up training a model on data it produced, which makes no sense)

In arguments.py set
```python
    "load_model"                : "mymodeldirectory",  # or 'models/sac+logos+ava1-l14-linearMSE.safetensors'
    "top_level_image_directory" : "C:\\Users\\chris\\Documents\\GitHub\\ComfyUI_windows_portable\\ComfyUI\\output\\compare", 
```

then `python aesthetic_score_from_model.py`

---

# Spotlight

[Spotlight](https://github.com/Renumics/spotlight) is a gui for analysing datasets. If you install it (probably just `pip install spotlight`) you can use `"mode":"spotlight"`, which works like evaluate, but launches spotlight with the resulting data.

## Use or share your model - custom classifier

To use the model you will need the four files saved in your save_model folder:

```
categories.json
config.json
model.safetensors
preprocessor_config.json
```

If you have set a save_strategy during training there will also be a number of subdirectories (which are *big*) that you don't need.

---

# Acknowledgements

Idea inspired by bmf (Đỗ Khang), who also provided lots of excellent discussion, feedback, and datasets!

Training code loosely based on [renumics](https://github.com/Renumics).