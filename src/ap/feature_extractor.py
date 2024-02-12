import torch
from PIL import Image
from safetensors.torch import save_file, load_file
import os
from torch._tensor import Tensor
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from tqdm import tqdm

try:
    from aim.torch.models import AIMForImageClassification
    from aim.torch.data import val_transforms
except:
    print("AIM not available - pip install git+https://git@github.com/apple/ml-aim.git if you want to use it")

# REALNAMES is used for downloading the (small) preprocessor file
REALNAMES = {
    "ChrisGoringe/vitH16" : "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}

class FeatureExtractor:
    @classmethod
    def realname(cls, pretrained):
        return REALNAMES.get(pretrained, pretrained)
    
    @classmethod
    def get_feature_extractor(cls, pretrained=None, **kwargs):
        pretrained = pretrained[0] if isinstance(pretrained,list) and len(pretrained)==1 else pretrained
        if isinstance(pretrained,list):
            return Multi_FeatureExtractor(pretrained=pretrained, **kwargs)
        elif "___" in pretrained:
            return Multi_FeatureExtractor(pretrained=pretrained.split("___"), **kwargs)
        elif "apple" in pretrained:
            return Apple_FeatureExtractor(pretrained=pretrained, **kwargs)
        else:
            return Transformers_FeatureExtractor(pretrained=pretrained, **kwargs)

    def __init__(self, pretrained, device="cuda", image_directory=".", use_cache=True, base_directory=None):
        self.metadata = {"feature_extractor_model":pretrained if isinstance(pretrained,str) else "___".join(pretrained)}
        self.device = device
        self.image_directory = image_directory
        self.pretrained = pretrained
        self.model = None
        self.have_warned = False
        self.use_cache = use_cache
        self.base_directory = base_directory
        self.dtype = torch.float

        self.cached = {}
        unique_name = pretrained if isinstance(pretrained, str) else "__".join(pretrained)
        self.cachefile = os.path.join(image_directory,f"featurecache.{unique_name.replace('/','_').replace(':','_')}.safetensors")
        if self.use_cache:
            if os.path.exists(self.cachefile):
                self.cached = load_file(self.cachefile, device=self.device)
                if len(self.cached):
                    print(f"Reloaded features from {self.cachefile} - delete this file if you don't want to do that")
                    self.number_of_features = len(next(iter(self.cached.values())))
                    return
                else:
                    self._load()
            print(f"No feature cachefile found at {self.cachefile}")
        self._load()

    @property
    def model_path(self):
        return os.path.join(self.base_directory, self.pretrained) if self.base_directory else self.pretrained
    
    def get_features_from_file(self, filepath, device="cuda", caching=False):
        rel = os.path.relpath(filepath, self.image_directory)
        if not self.use_cache:
            return self._get_image_features_tensor(Image.open(filepath))
        if rel not in self.cached:
            if caching and not self.have_warned:
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
        self.have_warned = True
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
    
    def _to(self, device:str, load_if_needed=True):
        self.device = device
        if not self.model or load_if_needed: return
        if self.model==None: self._load()

    def _delete_model(self):
        if not self.model: return
        self.model.to('cpu')
        self.model = None
        self.processor = None
    
class TextFeatureExtractor:
    def __init__(self, pretrained, device="cuda"):
        if isinstance(pretrained,list):
            assert len(pretrained)==1
            pretrained = pretrained[0]
        self.model = CLIPModel.from_pretrained(pretrained, cache_dir="models")
        self.tokenizer = AutoTokenizer.from_pretrained(FeatureExtractor.realname(pretrained), cache_dir="models")
        self.model.to(device)
        self.device = device

    def get_text_features_tensor(self, text:str, clip_skip:int=None):
        text_inputs = self.tokenizer( text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt" )
        text_input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        return self.model.get_text_features(text_input_ids, attention_mask)
    
    @property
    def number_of_features(self):
        return self.model.projection_dim
   
class Transformers_FeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load(self):
        self.model = CLIPModel.from_pretrained(self.pretrained, cache_dir="models")
        self.number_of_features = self.model.projection_dim
        self.model.text_model = None
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.realname(self.pretrained), cache_dir="models")

    def _get_image_features_tensor(self, image:Image) -> torch.Tensor:
        if self.model==None: self._load()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            for k in inputs:
                if isinstance(inputs[k],torch.Tensor): inputs[k] = inputs[k].to(self.device)
            outputs = self.model.get_image_features(output_hidden_states=True, **inputs)
            return outputs.flatten()
        
class Apple_FeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _load(self):
        self.model = AIMForImageClassification.from_pretrained(self.model_path, cache_dir="models").to(self.device)
        self.number_of_features = self.model.head.bn.num_features
        self.processor = val_transforms()

    def _get_image_features_tensor(self, image:Image) -> torch.Tensor:
        if self.model==None: self._load()
        self.model.to(self.device)
        with torch.no_grad():
            inp = self.processor(image).unsqueeze(0).to(self.device)
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

    def _to(self, device:str, load_if_needed=True):
        for fe in self.feature_extractors: fe._to(device, load_if_needed)

    @property
    def number_of_features(self):
        return sum(fe.number_of_features for fe in self.feature_extractors)

