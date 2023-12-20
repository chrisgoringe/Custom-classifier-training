import torch, clip
from PIL import Image
from safetensors.torch import save_file, load_file
import os, sys
from torch._tensor import Tensor
from transformers import CLIPVisionModel, AutoProcessor, CLIPModel

class CLIP:
    last_clip = None
    last_file = None
    model:torch.nn.Module = None

    @classmethod
    def get_clip(cls, pretrained="ViT-L/14", device="cuda", image_directory=".", use_cache=True):
        id = "_".join(pretrained) if isinstance(pretrained,list) else pretrained
        if not (cls.last_file and cls.last_file==id):
            if isinstance(pretrained,list):
                cls.last_clip = MultiCLIP(pretrained, device, image_directory, use_cache)
            elif pretrained in clip.available_models():
                cls.last_clip = OpenAICLIP(pretrained, device, image_directory, use_cache)
            else:
                cls.last_clip = TransformersCLIP(pretrained, device, image_directory, use_cache)
            cls.last_file = id
        return cls.last_clip

    def __init__(self, pretrained, device, image_directory, use_cache=True):
        self.metadata = {"clip_model":pretrained}
        self.device = device
        self.image_directory = image_directory
        if use_cache:
            self.cached = {}
            self.cachefile = os.path.join(image_directory,f"clipcache.{pretrained.replace('/','_')}.safetensors")
            try:
                self.cached = load_file(self.cachefile, device=self.device)
                print(f"Reloaded CLIPs from {self.cachefile} - delete this file if you don't want to do that")
            except:
                print(f"Didn't reload CLIPs from {self.cachefile}")

    def prepare_from_file(self, filepath, device="cuda"):
        rel = os.path.relpath(filepath, self.image_directory)
        if rel not in self.cached:
            self.cached[rel] = self.get_image_features_tensor(Image.open(filepath))
        return self.cached[rel].to(device)
    
    def save_cache(self):
        save_file(self.cached, self.cachefile)

    def get_metadata(self):
        return self.metadata
    
    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        raise NotImplementedError()
    
    def to(self, device:str):
        self.model.to(device)
        self.device = device

class OpenAICLIP(CLIP):  
    def __init__(self, pretrained="ViT-L/14", device="cuda", image_directory=".", use_cache=True):
        super().__init__(pretrained, device, image_directory, use_cache)
        self.model, self.preprocess = clip.load(pretrained, device=device, download_root="models/clip")

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        with torch.no_grad():
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.to(torch.float)
   
class TransformersCLIP(CLIP):
    def __init__(self, pretrained="", device="cuda", image_directory=".", use_cache=True):
        super().__init__(pretrained, device, image_directory, use_cache)
        self.model = CLIPModel.from_pretrained(pretrained, cache_dir="models/clip")
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(pretrained, cache_dir="models/clip")

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            for k in inputs:
                if isinstance(inputs[k],torch.Tensor): inputs[k] = inputs[k].to(self.device)
            outputs = self.model.get_image_features(output_hidden_states=True, **inputs)
            return outputs.flatten()
        
class MultiCLIP(CLIP):
    def __init__(self, pretrained:list, device, image_directory, use_cache):
        super().__init__("_".join(pretrained), device, image_directory, use_cache)
        self.models = [ CLIP.get_clip(p, "cpu", image_directory, use_cache=False) for p in pretrained ]
        
    def get_image_features_tensor(self, image: Image) -> Tensor:
        ift = None
        for m in self.models:
            m.to(self.device)
            features = m.get_image_features_tensor(image)
            ift = features if ift is None else torch.cat([ift,features])
            m.model.to('cpu')
        return ift

if __name__=='__main__':
    c = CLIP.get_clip()