import torch
from PIL import Image
from safetensors.torch import save_file, load_file
import os
from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPVisionModelWithProjection
from tqdm import tqdm

# REALNAMES is used for downloading the (small) preprocessor file
REALNAMES = {
    "ChrisGoringe/vitH16" : "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}

VISION_MODELS = ["ChrisGoringe/bigG-vision-fp16",]

class FeatureExtractorException(Exception):
    pass

class FeatureExtractor:
    @classmethod
    def realname(cls, pretrained):
        return REALNAMES.get(pretrained, pretrained)
    
    @classmethod
    def get_feature_extractor(cls, pretrained=None, **kwargs):
        pretrained = pretrained[0] if isinstance(pretrained,list) and len(pretrained)==1 else pretrained
        #if isinstance(pretrained,list):
        #    return Multi_FeatureExtractor(pretrained=pretrained, **kwargs)
        #elif "___" in pretrained:
        #    return Multi_FeatureExtractor(pretrained=pretrained.split("___"), **kwargs)
        #elif "apple" in pretrained:
        #    return Apple_FeatureExtractor(pretrained=pretrained, **kwargs)
        if pretrained in VISION_MODELS: return VisionModel_FeatureExtractor(pretrained=pretrained, **kwargs)
        return Transformers_FeatureExtractor(pretrained=pretrained, **kwargs)

    def __init__(self, pretrained, device="cuda", image_directory=".", use_cache=True, base_directory=".", hidden_states_used=[0,], stack_hidden_states=False):
        self.metadata = {"feature_extractor_model":pretrained if isinstance(pretrained,str) else "___".join(pretrained)}
        self.device = device
        self.image_directory = image_directory
        self.pretrained = pretrained
        self.models = {}
        self.use_cache = use_cache
        self.base_directory = base_directory

        self.stack_hidden_states = (stack_hidden_states=="True") if isinstance(stack_hidden_states, str) else stack_hidden_states
        self.hidden_states_used = list(int(x) for x in hidden_states_used[1:-1].split(',') if x) if isinstance(hidden_states_used,str) else hidden_states_used

        self.caches = { l:{} for l in self.hidden_states_used }
        self.cache_needs_saving = { l:False for l in self.hidden_states_used }

        for layer in self.hidden_states_used:
            if os.path.exists(cf := self.cachefile(layer)):
                self.caches[layer] = load_file(cf, device=self.device)
                print(f"Reloaded cache from {cf}")
                self.number_of_features = next(iter(self.caches[layer].values())).shape[-1]
            else:
                self._load()
                print(f"No feature cachefile found at {cf}, will generate features as required")

        self.number_of_features = self.number_of_features * (1 if self.stack_hidden_states else len(self.hidden_states_used))

    def check_model(self, model, hidden_states_used, stack_hidden_states):
        if model and self.metadata["feature_extractor_model"]!=model:
            raise FeatureExtractorException(f"Feature extractor has model {self.metadata['feature_extractor_model']} not {model}")
        if self.stack_hidden_states!=stack_hidden_states:
            raise FeatureExtractorException("Inconsistency in feature extractor stack_hidden_states")
        if self.hidden_states_used!=hidden_states_used:
            raise FeatureExtractorException("Inconsistency in hidden states used")

    def cachefile(self, layer):
        unique_name = self.pretrained + f"_{layer}"
        return os.path.join(self.image_directory,f"featurecache.{unique_name.replace('/','_').replace(':','_')}.safetensors")       

    @property
    def model_path(self):
        return os.path.join(self.base_directory, self.pretrained)
    
    def ensure_in_cache(self, filepath):
        rel = os.path.relpath(filepath, self.image_directory)
        layers_needed = []
        for layer in self.hidden_states_used:
            #if layer not in self.caches: self.caches[layer] = {}
            if rel not in self.caches[layer]:
                layers_needed.append(layer)
        if layers_needed:
            all_layers = self._get_image_features_tensor(Image.open(filepath), layers=layers_needed)
            for layer in all_layers:
                self.caches[layer][rel] = all_layers[layer]
                self.cache_needs_saving[layer] = True
    
    def get_features_from_file(self, filepath):
        self.ensure_in_cache(filepath)
        rel = os.path.relpath(filepath, self.image_directory)
        if self.stack_hidden_states:
            return torch.stack(list(self.caches[layer][rel] for layer in self.hidden_states_used))
        else:
            return torch.cat(list(self.caches[layer][rel] for layer in self.hidden_states_used))
    
    def precache(self, filepaths, delete_model=True):
        try:
            for f in tqdm(filepaths): self.ensure_in_cache(f)
        finally:
            self._save_cache()
        if delete_model: self._delete_model()

    def _save_cache(self):
        for layer in self.caches:
            if self.cache_needs_saving[layer]:
                save_file(self.caches[layer], self.cachefile(layer))
                self.cache_needs_saving[layer] = False

    def get_metadata(self):
        return self.metadata
    
    def _get_image_features_tensor(self, image:Image, layers:list) -> torch.Tensor:
        raise NotImplementedError()
    
    def _load(self):
        raise NotImplementedError()
    
    def _delete_model(self):
        self._to('cpu', load_if_needed=False)
        self.models = {}

    def _to(self, device, load_if_needed=True):
        if load_if_needed: self._load()
        if self.models.get('model', None) is not None: self.models['model'].to(device)
        self.device = device
    
class TextFeatureExtractor:
    def __init__(self, pretrained, device="cuda"):
        self.pretrained = pretrained[0] if isinstance(pretrained,list) else pretrained
        self.device = device
        self.models = {}

    def _load(self):
        if self.models.get('model', None) is None:
            self.models['model'] = CLIPModel.from_pretrained(self.pretrained, cache_dir="models")
            self.models['model'].to(self.device)
        if self.models.get('tokenizer', None) is None:
            self.models['tokenizer'] = AutoTokenizer.from_pretrained(FeatureExtractor.realname(self.pretrained), cache_dir="models")
        self.device = self.device

    def get_text_features_tensor(self, text:str, clip_skip:int=None):
        self._load()
        text_inputs = self.models['tokenizer']( text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt" )
        text_input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        return self.models['model'].get_text_features(text_input_ids, attention_mask)
    
    @property
    def number_of_features(self):
        return self.models['model'].projection_dim
   
class Transformers_FeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        ap_metadata = kwargs.pop('ap_metadata', {})
        kwargs['hidden_states_used'] = kwargs.pop('hidden_states_used', None) or ap_metadata.get('hidden_states_used', None)
        kwargs['stack_hidden_states'] = kwargs.pop('stack_hidden_states') if 'stack_hidden_states' in kwargs else ap_metadata.get('stack_hidden_states', None)

        super().__init__(**kwargs)
        self.metadata['hidden_states_used'] = "_".join(str(x) for x in self.hidden_states_used)

    def _load(self, model_clazz=CLIPModel):
        if self.models.get('model',None) is None: 
            self.models['model'] = model_clazz.from_pretrained(self.pretrained, cache_dir="models")
            self.number_of_features = self.models['model'].visual_projection.out_features
            self.metadata['number_of_features'] = str(self.number_of_features)
            self.models['model'].text_model = None
            self.models['model'].to(self.device)
        if self.models.get('processor',None) is None: 
            self.models['processor'] = AutoProcessor.from_pretrained(self.realname(self.pretrained), cache_dir="models")

    def _get_image_features_tensor(self, image:Image, layers:list) -> torch.Tensor:
        self._load()
        with torch.no_grad():
            inputs = self.models['processor'](images=image, return_tensors="pt")
            vision_outputs = self.models['model'].vision_model(
                pixel_values=inputs['pixel_values'].to(self.device),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            results = {}
            for this_layer in layers:
                poolable = vision_outputs.hidden_states[-1-this_layer][:,0,:]
                pooled_output = self.models['model'].vision_model.post_layernorm(poolable)
                results[this_layer] = self.models['model'].visual_projection(pooled_output)
            return results
        
class VisionModel_FeatureExtractor(Transformers_FeatureExtractor):
    def _load(self):
        super()._load(model_clazz=CLIPVisionModelWithProjection)
        self.models['model'].to(self.models['model'].config.torch_dtype)

'''     
class Apple_FeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        if (hs := kwargs.pop('hidden_states', None)): print(f"hidden_states not implemented for Apple_FeatureExtractor - ignoring {hs}")
        kwargs.pop('ap_metadata', None)
        super().__init__(**kwargs)
        
    def _load(self):
        try:
            from aim.torch.models import AIMForImageClassification
            from aim.torch.data import val_transforms
        except:
            print("AIM not available - pip install git+https://git@github.com/apple/ml-aim.git if you want to use it")
            raise NotImplementedError("AIM not available - pip install git+https://git@github.com/apple/ml-aim.git if you want to use it")
        
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

'''