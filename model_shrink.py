import torch
from transformers import CLIPModel, CLIPVisionConfig, CLIPVisionModelWithProjection, AutoModel, AutoProcessor
import os, shutil, json
from safetensors.torch import save_file, load_file

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
    with open(os.path.join(directory,"config.json")) as f: print(json.dumps(vision_config, indent=2), file=f)

    processor = AutoProcessor.from_pretrained(pretrained, cache_dir="models")
    processor_config = processor.get_config_dict()
    with open(os.path.join(directory,"preprocessor_config.json")) as f: print(json.dumps(processor_config, indent=2), file=f)

def load(directory):
    c = CLIPVisionModelWithProjection.from_pretrained(directory)
    p = AutoProcessor.from_pretrained(directory)


convert("openai/clip-vit-large-patch14", "models/vit-vision-fp16")

load("models/vit-vision-fp16")


pass


