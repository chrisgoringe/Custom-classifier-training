import torch
from transformers import CLIPModel, CLIPVisionConfig, CLIPVisionModelWithProjection, AutoModel, AutoProcessor, PretrainedConfig
import os, shutil, json
from safetensors.torch import save_file, load_file
from aim.torch.models import AIMForImageClassification

def convert(pretrained, directory):
    if not os.path.exists(directory): os.makedirs(directory)
    c = CLIPModel.from_pretrained(pretrained, cache_dir="models")
    c.text_model = None
    c.text_projection = None
    c.to(torch.half)
    sd = c.state_dict()
    sd.pop('logit_scale',None)
    save_file(sd, f"{directory}/model.safetensors", {"format":"pt"})

    vision_config = c.config.vision_config.get_config_dict()
    with open(os.path.join(directory,"config.json"),'w') as f: print(json.dumps(vision_config, indent=2), file=f)

    processor = AutoProcessor.from_pretrained(pretrained, cache_dir="models")
    processor_config = processor.get_config_dict()
    with open(os.path.join(directory,"preprocessor_config.json"),'w') as f: print(json.dumps(processor_config, indent=2), file=f)

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

#convert("openai/clip-vit-large-patch14", "models/vit-vision-fp16")
#load("models/vit-vision-fp16")
for s in ["7B",]:
    directory = os.path.join("models",f"aim-{s}-fp16")
    convert_aim(f"apple/aim-{s}", directory)
    load_aim(directory)


pass


