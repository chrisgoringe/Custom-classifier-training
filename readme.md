# Custom Classifier Training

Script to train (and evaluate) a custom image classifier in a way that is suitable for use with the Custom Classifier ComfyUI nodes (coming soon!).

They are generic models, so should work for any other use for that matter.

## Install

In some suitable directory,
```
git clone https://github.com/chrisgoringe/custom-classifier-training
```

## Virtual environment

Any virtual environment that you use for torch stuff will probably work (for instance, if you run ComfyUI, use that one).

If creating a new one, first [install torch](https://pytorch.org/get-started/locally/), then 
```
pip install -r requirements.txt
```

## Setup your images

Create a subdirectory, say `images`, as your `top_level_image_directory`. In it create one subdirectory per category, named for the category. So, for instance, `images/good` and `images/bad`.

## Configuration

Edit `arguments.py`. The comments should explain everything. Then just run `python custom_classifier_training.py`.

Basic settings for training:

```python
    "mode"                      : "train",
    "base_model"                : "google/vit-base-patch16-224-in21k",    
    "load_model"                : "",          # unless restarting, in which case 'my_model'
    "save_model"                : "my_model",  # the path to a folder used to save
    "top_level_image_directory" : "path/to/images", 
```

Then for evaluation:
```python
    "mode"                      : "evaluate",
    "base_model"                : "google/vit-base-patch16-224-in21k",    
    "load_model"                : "my_model",   
    "save_model"                : "my_model",
    "top_level_image_directory" : "path/to/my-images", 
```

Use Task Manager to see if you are using shared GPU memory instead of just dedicated - if so, reduce batch size etc.

## Pretrained models

This script is for customising pretrained models (`base_model` in the configuration). It's been tested with `google/vit-base-patch16-224-in21k` and `google/efficientnet-b5` and should work with the other efficientnets (b0...b7 in increasing size and complexity) and ViT architectures. 

It might well work with any transformers based model. If it fails, the most likely issue will be that you need to add a new ImageProcessor class in `_get_feature_extractor` in `src/prediction.py`. Please report successes or failures in the GitHub issues!

## Use or share your model

To use the model you will need the four files saved in your save_model folder:

```
categories.json
config.json
model.safetensors
preprocessor_config.json
```

If you have set a save_strategy during training there will also be a number of subdirectories (which are *big*) that you don't need.

## Acknowledgements

Idea inspired by bmf, who also provided lots of excellent discussion, feedback, and datasets!

Training code loosely based on [renumics](https://github.com/Renumics).