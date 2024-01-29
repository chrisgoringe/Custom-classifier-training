import torch, clip
from PIL import Image
from safetensors.torch import save_file, load_file
import os, sys
from torch._tensor import Tensor
from transformers import CLIPVisionModel, AutoProcessor, CLIPModel
from tqdm import tqdm

try:
    from aim.torch.models import AIMForImageClassification
    from aim.torch.data import val_transforms
except:
    print("AIM not available - pip install git+https://git@github.com/apple/ml-aim.git if you want to use it")

OpenAIModels = ["RN50","RN101","RN50x4","RN50x16","RN50x64","ViT-B/32","ViT-B/16","ViT-L/14","ViT-L/14@336px"]
AppleAIMModels = ["apple/aim-600M","apple/aim-1B","apple/aim-3B","apple/aim-7B"]

class CLIP:
    last_clip = None
    last_file = None

    @classmethod
    def get_clip(cls, pretrained="ViT-L/14", device="cuda", image_directory="."):
        id = "_".join(pretrained) if isinstance(pretrained,list) else pretrained
        if not (cls.last_file and cls.last_file==id):
            if isinstance(pretrained,list):
                cls.last_clip = MultiCLIP(pretrained, device, image_directory)
            elif pretrained in OpenAIModels:
                cls.last_clip = OpenAICLIP(pretrained, device, image_directory)
            elif pretrained in AppleAIMModels:
                cls.last_clip = AppleNotCLIP(pretrained, device, image_directory)
            else:
                cls.last_clip = TransformersCLIP(pretrained, device, image_directory)
            cls.last_file = id
        return cls.last_clip

    def __init__(self, pretrained, device, image_directory):
        self.metadata = {"clip_model":pretrained}
        self.device = device
        self.image_directory = image_directory
        self.pretrained = pretrained
        self.model = None

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
    
    def cache_from_files(self, filepaths, device="cuda"):
        for filepath in tqdm(filepaths, desc=f"Caching {self.pretrained}"):
            rel = os.path.relpath(filepath, self.image_directory)
            self.cached[rel] = self.prepare_from_file(filepath, device)
    
    def precache(self, filepaths):
        newfiles = { f for f in filepaths if os.path.relpath(f, self.image_directory) not in self.cached }
        if newfiles:
            self.cache_from_files(newfiles)
            self.save_cache()

    def save_cache(self):
        save_file(self.cached, self.cachefile)

    def get_metadata(self):
        return self.metadata
    
    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        raise NotImplementedError()
    
    def load(self):
        raise NotImplementedError()
    
    def to(self, device:str):
        if self.model==None: self.load()
        self.model.to(device)
        self.device = device

    def simplify(self):
        pass

class OpenAICLIP(CLIP):  
    def __init__(self, pretrained="ViT-L/14", device="cuda", image_directory="."):
        super().__init__(pretrained, device, image_directory)

    def load(self):
        self.model, self.preprocess = clip.load(self.pretrained, device=self.device, download_root="models/clip")

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        if self.model==None: self.load()
        with torch.no_grad():
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.to(torch.float)
   
class TransformersCLIP(CLIP):
    def __init__(self, pretrained="", device="cuda", image_directory="."):
        super().__init__(pretrained, device, image_directory)

    def load(self):
        self.model = CLIPModel.from_pretrained(self.pretrained, cache_dir="models/clip")
        self.simplify()
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.pretrained, cache_dir="models/clip")

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        if self.model==None: self.load()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            for k in inputs:
                if isinstance(inputs[k],torch.Tensor): inputs[k] = inputs[k].to(self.device)
            outputs = self.model.get_image_features(output_hidden_states=True, **inputs)
            return outputs.flatten()
        
    def simplify(self):
        del self.model.text_model

class AppleNotCLIP(CLIP):
    def __init__(self, pretrained="", device="cuda", image_directory="."):
        super().__init__(pretrained, device, image_directory)
        
    def load(self):
        self.model = AIMForImageClassification.from_pretrained(self.pretrained, cache_dir="models/apple").to(self.device)
        self.processor = val_transforms()

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        if self.model==None: self.load()
        with torch.no_grad():
            inp = self.processor(image).unsqueeze(0).to(self.device)
            image_features = self.model.extract_features(inp, max_block_id=-1)
            return image_features.to(torch.float)
        
class MultiCLIP(CLIP):
    def __init__(self, pretrained:list, device, image_directory):
        super().__init__("_".join(pretrained), device, image_directory)
        self.models = [ CLIP.get_clip(p, "cpu", image_directory) for p in pretrained ]
        
    def get_image_features_tensor(self, image: Image) -> Tensor:
        ift = None
        for m in self.models:
            m.to(self.device)
            features = m.get_image_features_tensor(image)
            ift = features if ift is None else torch.cat([ift,features])
            #m.model.to('cpu')
        return ift
    
    def precache(self, filepaths):
        newfiles = { f for f in filepaths if os.path.relpath(f, self.image_directory) not in self.cached }
        if not newfiles: return
        for m in self.models:
            m.to('cuda')
            m.precache(newfiles)
            for f in newfiles:
                rel = os.path.relpath(f, self.image_directory)
                features = m.prepare_from_file(f)
                self.cached[rel] = features if rel not in self.cached else torch.cat([self.cached[rel],features])
            m.model.to('cpu')
        self.save_cache()

if __name__=='__main__':
    c = CLIP.get_clip()