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

NUMBER_OF_FEATURES = {  "openai/clip-vit-large-patch14"            : 768,
                        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" : 1280,
                        "apple/aim-600M"                           : 1536,
                        "apple/aim-1B"                             : 2048,
                        "apple/aim-3B"                             : 3072,
                        "apple/aim-7B"                             : 0,
}

class FeatureExtractor:
    last_fe = None
    last_file = None

    @classmethod
    def get_feature_extractor(cls, pretrained="ViT-L/14", device="cuda", image_directory="."):
        id = "_".join(pretrained) if isinstance(pretrained,list) else pretrained
        if not (cls.last_file and cls.last_file==id):
            if isinstance(pretrained,list):
                cls.last_fe = Multi_FeatureExtractor(pretrained, device, image_directory)
            elif pretrained in OpenAIModels:
                cls.last_fe = OpenAI_FeatureExtractor(pretrained, device, image_directory)
            elif pretrained.startswith("apple"):
                cls.last_fe = Apple_FeatureExtractor(pretrained, device, image_directory)
            else:
                cls.last_fe = Transformers_FeatureExtractor(pretrained, device, image_directory)
            cls.last_file = id
        return cls.last_fe

    def __init__(self, pretrained, device, image_directory):
        self.metadata = {"feature_extractor_model":pretrained}
        self.device = device
        self.image_directory = image_directory
        self.pretrained = pretrained
        self.model = None

        self.cached = {}
        self.cachefile = os.path.join(image_directory,f"featurecache.{pretrained.replace('/','_').replace(':','_')}.safetensors")
        try:
            self.cached = load_file(self.cachefile, device=self.device)
            print(f"Reloaded features from {self.cachefile} - delete this file if you don't want to do that")
        except:
            print(f"Didn't reload features from {self.cachefile}")
    
    def prepare_from_file(self, filepath, device="cuda"):
        rel = os.path.relpath(filepath, self.image_directory)
        if rel not in self.cached:
            self.cached[rel] = self.get_image_features_tensor(Image.open(filepath))
        return self.cached[rel].to(device).squeeze()
    
    def cache_from_files(self, filepaths, device="cuda"):
        for filepath in tqdm(filepaths, desc=f"Caching {self.pretrained}"):
            rel = os.path.relpath(filepath, self.image_directory)
            self.cached[rel] = self.prepare_from_file(filepath, device)
    
    def precache(self, filepaths, delete_model=True):
        newfiles = { f for f in filepaths if os.path.relpath(f, self.image_directory) not in self.cached }
        if newfiles:
            self.cache_from_files(newfiles)
            self.save_cache()
        if delete_model: self.delete_model()

    def save_cache(self):
        save_file(self.cached, self.cachefile)

    def get_metadata(self):
        return self.metadata
    
    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        raise NotImplementedError()
    
    def load(self):
        raise NotImplementedError()
    
    def delete_model(self):
        raise NotImplementedError()
    
    def to(self, device:str):
        if self.model==None: self.load()
        self.model.to(device)
        self.device = device

    def simplify(self):
        pass

    @property
    def number_of_features(self):
        if self.pretrained in NUMBER_OF_FEATURES: return NUMBER_OF_FEATURES[self.pretrained]
        raise NotImplementedError()

class OpenAI_FeatureExtractor(FeatureExtractor):  
    def __init__(self, pretrained="ViT-L/14", device="cuda", image_directory="."):
        super().__init__(pretrained, device, image_directory)

    def load(self):
        self.model, self.preprocess = clip.load(self.pretrained, device=self.device, download_root="models/clip")

    def delete_model(self):
        del self.model, self.preprocess

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        if self.model==None: self.load()
        with torch.no_grad():
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.to(torch.float)
   
class Transformers_FeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained="", device="cuda", image_directory="."):
        super().__init__(pretrained, device, image_directory)

    def load(self):
        self.model = CLIPModel.from_pretrained(self.pretrained, cache_dir="models/clip")
        self.simplify()
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.pretrained, cache_dir="models/clip")

    def delete_model(self):
        self.model.to('cpu')
        del self.model, self.processor

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

class Apple_FeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained="", device="cuda", image_directory="."):
        super().__init__(pretrained, device, image_directory)
        if ":" in self.pretrained:
            self.pretrained, self.block = self.pretrained.split(':')
            self.block = int(self.block)
        else:
            self.block = None
        
    def load(self):
        self.model = AIMForImageClassification.from_pretrained(self.pretrained, cache_dir="models/apple").to(self.device)
        self.processor = val_transforms()

    def delete_model(self):
        if self.model:
            self.model.to('cpu')
            del self.model, self.processor

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        if self.model==None: self.load()
        with torch.no_grad():
            inp = self.processor(image).unsqueeze(0).to(self.device)
            if self.block:
                image_features = self.model.extract_features(inp, max_block_id=0)
                image_features = image_features[0][:,:self.block,:]
            else:
                _, image_features = self.model(inp)
            return image_features.to(torch.float).flatten()
        
class Multi_FeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained:list, device, image_directory):
        super().__init__("_".join(pretrained), device, image_directory)
        self.feature_extractors = [ FeatureExtractor.get_feature_extractor(p, "cpu", image_directory) for p in pretrained ]

    def delete_model(self):
        for fe in self.feature_extractors: fe.delete_model()
        
    def get_image_features_tensor(self, image: Image) -> Tensor:
        ift = None
        for fe in self.feature_extractors:
            fe.to(self.device)
            features = fe.get_image_features_tensor(image)
            ift = features if ift is None else torch.cat([ift,features])
        return ift
    
    def precache(self, filepaths, delete_model=True):
        newfiles = { f for f in filepaths if os.path.relpath(f, self.image_directory) not in self.cached }
        if not newfiles: return
        for fe in self.feature_extractors:
            fe.to('cuda')
            fe.precache(newfiles, delete_model=False)
            for f in newfiles:
                rel = os.path.relpath(f, self.image_directory)
                features = fe.prepare_from_file(f)
                self.cached[rel] = features if rel not in self.cached else torch.cat([self.cached[rel],features])
            fe.model.to('cpu')
        if delete_model: self.delete_model()
        self.save_cache()

    @property
    def number_of_features(self):
        return sum(fe.number_of_features for fe in self.feature_extractors)
