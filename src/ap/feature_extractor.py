import torch, clip
from PIL import Image
from safetensors.torch import save_file, load_file
import os
from torch._tensor import Tensor
from transformers import AutoProcessor, CLIPModel
from tqdm import tqdm

try:
    from aim.torch.models import AIMForImageClassification
    from aim.torch.data import val_transforms
except:
    print("AIM not available - pip install git+https://git@github.com/apple/ml-aim.git if you want to use it")

#OpenAIModels = ["RN50","RN101","RN50x4","RN50x16","RN50x64","ViT-B/32","ViT-B/16","ViT-L/14","ViT-L/14@336px"]

NUMBER_OF_FEATURES = {  "openai/clip-vit-large-patch14"            : 768,
                        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" : 1280,
                        "apple/aim-600M"                           : 1536,
                        "apple/aim-1B"                             : 2048,
                        "apple/aim-3B"                             : 3072,
                        "apple/aim-7B"                             : 4096,
}

class FeatureExtractor:
    @classmethod
    def realname(cls, pretrained):
        x = pretrained
        if x.startswith(r"models/"): x = x[7:]
        if x.endswith("-half"): x = x[:-5]
        return x
    
    @classmethod
    def get_feature_extractor(cls, pretrained="ViT-L/14", device="cuda", image_directory=".", use_cache=True):
        if isinstance(pretrained,list):
            return Multi_FeatureExtractor(pretrained=pretrained, device=device, image_directory=image_directory, use_cache=use_cache)
        elif "___" in pretrained:
            return Multi_FeatureExtractor(pretrained=pretrained.split("___"), device=device, image_directory=image_directory, use_cache=use_cache)
        #elif pretrained in OpenAIModels:
        #    return OpenAI_FeatureExtractor(pretrained=pretrained, device=device, image_directory=image_directory, use_cache=use_cache)
        elif "apple" in pretrained:
            return Apple_FeatureExtractor(pretrained=pretrained, device=device, image_directory=image_directory, use_cache=use_cache)
        else:
            return Transformers_FeatureExtractor(pretrained=pretrained, device=device, image_directory=image_directory, use_cache=use_cache)

    def __init__(self, pretrained, device, image_directory, use_cache):
        self.metadata = {"feature_extractor_model":pretrained if isinstance(pretrained,str) else "___".join(pretrained)}
        self.device = device
        self.image_directory = image_directory
        self.pretrained = pretrained
        self.model = None
        self.have_warned = False
        self.use_cache = use_cache

        self.cached = {}
        unique_name = pretrained if isinstance(pretrained, str) else "__".join(pretrained)
        self.cachefile = os.path.join(image_directory,f"featurecache.{unique_name.replace('/','_').replace(':','_')}.safetensors")
        if self.use_cache:
            if os.path.exists(self.cachefile):
                self.cached = load_file(self.cachefile, device=self.device)
                print(f"Reloaded features from {self.cachefile} - delete this file if you don't want to do that")
            else:
                print(f"No feature cachefile found at {self.cachefile}")
    
    def get_features_from_file(self, filepath, device="cuda", caching=False):
        rel = os.path.relpath(filepath, self.image_directory)
        if not self.use_cache:
            return self._get_image_features_tensor(Image.open(filepath))
        if rel not in self.cached:
            if not self.have_warned and not caching:
                print("Getting features from file not in feature cache - precaching is likely to be faster!")
                self.have_warned = True
            self.cached[rel] = self._get_image_features_tensor(Image.open(filepath))
        return self.cached[rel].to(device).squeeze()
    
    def _cache_from_files(self, filepaths, device="cuda"):
        if not self.use_cache: return
        for filepath in tqdm(filepaths, desc=f"Caching {self.pretrained}"):
            rel = os.path.relpath(filepath, self.image_directory)
            self.cached[rel] = self.get_features_from_file(filepath, device, caching=True)
    
    def precache(self, filepaths, delete_model=True):
        if not self.use_cache: return
        newfiles = { f for f in filepaths if os.path.relpath(f, self.image_directory) not in self.cached }
        if newfiles:
            self._cache_from_files(newfiles)
            self._save_cache()
        if delete_model: self._delete_model()

    def _save_cache(self):
        if not self.use_cache: return
        save_file(self.cached, self.cachefile)

    def get_metadata(self):
        return self.metadata
    
    def _get_image_features_tensor(self, image:Image) -> torch.Tensor:
        raise NotImplementedError()
    
    def _load(self):
        raise NotImplementedError()
    
    def _delete_model(self):
        raise NotImplementedError()
    
    def _to(self, device:str, load_if_needed=True):
        self.device = device
        if not self.model or load_if_needed: return
        if self.model==None: self._load()

    def simplify(self):
        pass

    @property
    def number_of_features(self):
        x = self.realname(self.pretrained)
        if x in NUMBER_OF_FEATURES: return NUMBER_OF_FEATURES[x]
        raise NotImplementedError()
'''
class OpenAI_FeatureExtractor(FeatureExtractor):  
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load(self):
        self.model, self.preprocess = clip.load(self.pretrained, device=self.device, download_root="models/clip")

    def _delete_model(self):
        del self.model, self.preprocess

    def _get_image_features_tensor(self, image:Image) -> torch.Tensor:
        if self.model==None: self._load()
        with torch.no_grad():
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features._to(torch.float)
'''
   
class Transformers_FeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load(self):
        self.model = CLIPModel.from_pretrained(self.pretrained, cache_dir="models/clip")
        self.simplify()
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.realname(self.pretrained), cache_dir="models/clip")

    def _delete_model(self):
        if not self.model: return
        self.model.to('cpu')
        del self.model, self.processor

    def _get_image_features_tensor(self, image:Image) -> torch.Tensor:
        if self.model==None: self._load()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            for k in inputs:
                if isinstance(inputs[k],torch.Tensor): inputs[k] = inputs[k].to(self.device)
            outputs = self.model.get_image_features(output_hidden_states=True, **inputs)
            return outputs.flatten()
        
    def simplify(self):
        del self.model.text_model

class Apple_FeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if ":" in self.pretrained:
            self.pretrained, self.block = self.pretrained.split(':')
            self.block = int(self.block)
        else:
            self.block = None
        
    def _load(self):
        self.model = AIMForImageClassification.from_pretrained(self.pretrained, cache_dir="models/apple").to(self.device)
        self.processor = val_transforms()

    def _delete_model(self):
        if self.model:
            self.model.to('cpu')
            del self.model, self.processor

    def _get_image_features_tensor(self, image:Image) -> torch.Tensor:
        if self.model==None: self._load()
        self.model.to(self.device)
        with torch.no_grad():
            inp = self.processor(image).unsqueeze(0).to(self.device)
            if self.block:
                image_features = self.model.extract_features(inp, max_block_id=0)
                image_features = image_features[0][:,:self.block,:]
            else:
                _, image_features = self.model(inp)
            return image_features.to(torch.float).flatten()
        
class Multi_FeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained:list, **kwargs):
        super().__init__(pretrained, **kwargs)
        self.feature_extractors = [ FeatureExtractor.get_feature_extractor(p, **kwargs) for p in pretrained ]

    def _delete_model(self):
        for fe in self.feature_extractors: fe._delete_model()
        
    def _get_image_features_tensor(self, image: Image) -> Tensor:
        ift = None
        for fe in self.feature_extractors:
            fe._to(self.device)
            features = fe._get_image_features_tensor(image)
            ift = features if ift is None else torch.cat([ift,features])
        return ift
    
    def precache(self, filepaths, delete_model=True):
        if not self.use_cache: return
        newfiles = { f for f in filepaths if os.path.relpath(f, self.image_directory) not in self.cached }
        if not newfiles: return
        for fe in self.feature_extractors:
            fe._to('cuda')
            fe.precache(newfiles, delete_model=False)
            for f in newfiles:
                rel = os.path.relpath(f, self.image_directory)
                features = fe.get_features_from_file(f)
                self.cached[rel] = features if rel not in self.cached else torch.cat([self.cached[rel],features])
            fe._to('cpu', load_if_needed=False)
        if delete_model: self._delete_model()
        self._save_cache()

    @property
    def number_of_features(self):
        return sum(fe.number_of_features for fe in self.feature_extractors)
