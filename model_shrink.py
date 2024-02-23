import torch
from transformers import CLIPModel, CLIPVisionConfig, CLIPVisionModelWithProjection, AutoModel, AutoProcessor
import os, shutil, json
from safetensors.torch import save_file, load_file

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
                if os.path.exists(os.path.join(root,dir,"preprocessor_config.json")):
                    shutil.copy(os.path.join(root,dir,"preprocessor_config.json"), os.path.join(to_directory,"preprocessor_config.json"))
                return

#for sz in ["600M","1B","3B","7B"]:
#    halve(AIMForImageClassification, f"apple/aim-{sz}", "models/apple", f"models/apple/aim-{sz}-half")

sd = load_file('models/bigG-vision-half/model.safetensors')
sd.pop('logit_scale')
save_file(sd, 'models/bigG-vision-fp16/model.safetensors', {"format":"pt"})

#with open("models/bigG-vision-half/config.json",'r') as f: j = json.load(f)
#con = CLIPVisionConfig(**j)
#vision_model = CLIPVisionModelWithProjection(con)
#sd = load_file("models/bigG-vision-half/model.safetensors")
#missing, unexpected = vision_model.load_state_dict(sd, strict=False)

#CLIPVisionModelWithProjection.from_pretrained("models/bigG-vision-half")
#c = AutoModel.from_pretrained("ChrisGoringe/bigG-vision-half")
c = CLIPVisionModelWithProjection.from_pretrained("models/bigG-vision-fp16")
p = AutoProcessor.from_pretrained("models/bigG-vision-fp16")



pass


