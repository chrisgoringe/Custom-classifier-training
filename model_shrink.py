import torch
from aim.torch.models import AIMForImageClassification
from transformers import CLIPModel
import os, shutil

def halve(clazz, pretrained, cache_dir, save_directory):
    model = clazz.from_pretrained(pretrained, cache_dir=cache_dir)
    model.to(torch.half)
    model.save_pretrained(save_directory=save_directory)
    find_and_copy_config(os.path.join(cache_dir, f"models--{'--'.join(pretrained.split('/'))}"), save_directory)

def find_and_copy_config(from_directory, to_directory):
    for root, dirs, files in os.walk(from_directory):
        for dir in dirs:
            if os.path.exists(os.path.join(root,dir,"config.json")):
                shutil.copy(os.path.join(root,dir,"config.json"), os.path.join(to_directory,"config.json"))
                return

#for sz in ["600M","1B","3B","7B"]:
#    halve(AIMForImageClassification, f"apple/aim-{sz}", "models/apple", f"models/apple/aim-{sz}-half")

for x in ["openai/clip-vit-large-patch14","laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"]:
    halve(CLIPModel, x, "models/clip", f"models/clip/{x}-half")
