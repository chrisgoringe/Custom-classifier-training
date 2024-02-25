import torch
from transformers import CLIPModel, CLIPVisionConfig, CLIPVisionModelWithProjection, AutoModel, AutoProcessor, PretrainedConfig, CLIPProcessor
import os, shutil, json
from safetensors.torch import save_file, load_file
from aim.torch.models import AIMForImageClassification

VISION_CONFIG_DEFAULTS = {
    "attention_dropout": 0.0,
    "dropout": 0.0,
    "hidden_act": "gelu",
    "hidden_size": 1664,
    "image_size": 224,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "layer_norm_eps": 1e-05,
    "model_type": "clip_vision_model",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 48,
    "patch_size": 14,
    "projection_dim": 1280,
    "transformers_version": "4.24.0"
  }

def convert(pretrained, directory):
    if not os.path.exists(directory): os.makedirs(directory)
    c = CLIPModel.from_pretrained(pretrained, cache_dir="models")
    c.text_model = None
    c.text_projection = None
    c.to(torch.half)
    sd = c.state_dict()
    sd.pop('logit_scale',None)
    save_file(sd, f"{directory}/model.safetensors", {"format":"pt"})

    d = PretrainedConfig.get_config_dict(pretrained)
    while isinstance(d,list) or isinstance(d,tuple): 
        d = d[0]
    d.pop("_commit_hash",None)

    config = {"architectures": ["CLIPVisionModelWithProjection"], "torch_dtype":"float16"}
    for key in VISION_CONFIG_DEFAULTS: config[key] = d['vision_config'].get(key, VISION_CONFIG_DEFAULTS[key])

    with open(os.path.join(directory,"config.json"),'w') as f: print(json.dumps(config, indent=2), file=f)

    cached_directory = os.path.join("models", pretrained)
    cached_directory = cached_directory.replace('/','--')
    cached_directory = cached_directory.replace('\\','--')
    cached_directory = os.path.join("models", cached_directory)
    with open(os.path.join(cached_directory,'refs','main'), 'r') as f: 
        main_snapshot_directory = os.path.join(cached_directory,"snapshots",f.readline())
    for other_file in ["preprocessor_config.json",]:
        shutil.copy(os.path.join(main_snapshot_directory,other_file), os.path.join(directory,other_file))

def convert_aim(pretrained, directory):
    if not os.path.exists(directory): os.makedirs(directory)
    c = AIMForImageClassification.from_pretrained(pretrained, cache_dir="models")
    c.to(torch.half)
    c.save_pretrained(directory)
    d = PretrainedConfig.get_config_dict(pretrained)
    while isinstance(d,list) or isinstance(d,tuple): 
        d = d[0]
    d.pop("_commit_hash",None)
    with open(os.path.join(directory,"config.json"),'w') as f: print(json.dumps(d, indent=2), file=f)

def load(directory):
    c = CLIPVisionModelWithProjection.from_pretrained(directory)
    p = AutoProcessor.from_pretrained(directory)

def load_aim(directory):
    c = AIMForImageClassification.from_pretrained(directory)

directory = "models/vit-large-p14-vision-fp16"
model = "openai/clip-vit-large-patch14"
is_aim = False

if not is_aim:
    convert(model, directory)
    load(directory)
else:
    convert_aim(model, directory)
    load_aim(directory)
