import torch
from PIL import Image
from safetensors.torch import save_file, load_file
import os
from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPVisionModelWithProjection, PretrainedConfig
from tqdm import tqdm

# REALNAMES is used for downloading the (small) preprocessor file
REALNAMES = {
    "ChrisGoringe/vitH16" : "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}

# VISION_MODELS are CLIP models that have been reduced by removal of the text part (and often conversion to fp16)
VISION_MODELS = ["ChrisGoringe/bigG-vision-fp16",]

# Models that load using the apple/ml-aim framework (not yet very well integrated into transformers)
APPLE_MODELS = ["apple/aim-600M", "ChrisGoringe/aim-600M-fp16", 
                "apple/aim-1B",   "ChrisGoringe/aim-1B-fp16", 
                "apple/aim-3B",   "ChrisGoringe/aim-3B-fp16", 
                "apple/aim-7B",   "ChrisGoringe/aim-7B-fp16", 
                ]

class FeatureExtractorException(Exception):
    pass

class FeatureExtractor:
    @classmethod
    def realname(cls, pretrained):
        #return pretrained
        return REALNAMES.get(pretrained, pretrained)
    
    @classmethod
    def get_feature_extractor(cls, pretrained=None, **kwargs):
        if isinstance(pretrained,str) and len(pretrained.split('__'))>1: pretrained = pretrained.split('__')
        pretrained = pretrained[0] if isinstance(pretrained,list) and len(pretrained)==1 else pretrained

        if isinstance(pretrained,list): return Multi_FeatureExtractor(pretrained=pretrained, **kwargs)
        if pretrained in APPLE_MODELS: return Apple_FeatureExtractor(pretrained=pretrained, **kwargs)
        if pretrained in VISION_MODELS: return VisionModel_FeatureExtractor(pretrained=pretrained, **kwargs)

        return Transformers_FeatureExtractor(pretrained=pretrained, **kwargs)

    def __init__(self, pretrained, device="cuda", image_directory=".", use_cache=True, base_directory=".", hidden_states_used=None, hidden_states_mode="join", fp16_features=False):
        self.device = device
        self.image_directory = image_directory
        self.pretrained = pretrained
        self.models = {}
        self.use_cache = use_cache
        self.base_directory = base_directory

        self.hidden_states_mode = hidden_states_mode
        self.hidden_states_used = list(int(x) for x in hidden_states_used[1:-1].split(',') if x) if isinstance(hidden_states_used,str) else hidden_states_used
        self.hidden_states_used = self.hidden_states_used or self.default_hidden_states
        self.dtype = torch.half if fp16_features else torch.float

        if self.hidden_states_mode=='default': self.hidden_states_mode = self.default_hidden_states_mode

        # input to post_processing is a list length n of tensors shape [1,_number_of_features]  (n = len(hidden_states_used)
        if   self.hidden_states_mode=='join':    self.post_processing, self.states_per_state = lambda a : torch.cat(a,dim=-1), len(self.hidden_states_used)
        elif self.hidden_states_mode=='weight':  self.post_processing, self.states_per_state = lambda a : torch.cat(a, dim=0), 1
        elif self.hidden_states_mode=='average': self.post_processing, self.states_per_state = lambda a : torch.stack(a, dim=-1).mean(dim=-1) , 1

        self.caches = { l:{} for l in self.hidden_states_used }
        self.cache_needs_saving = { l:False for l in self.hidden_states_used }

        for layer in self.hidden_states_used:
            if os.path.exists(cf := self._cachefile(layer)):
                self.caches[layer] = load_file(cf, device=self.device)
                print(f"Reloaded cache from {cf}")
                self._number_of_features = next(iter(self.caches[layer].values())).shape[-1]
            else:
                self._load()
                print(f"No feature cachefile found at {cf}, will generate features as required")

        self.metadata = {"feature_extractor_model":pretrained, 'hidden_states_used':",".join(str(x) for x in self.hidden_states_used), 'hidden_states_mode':self.hidden_states_mode}

    @property
    def default_hidden_states(self):
        return [0,]
    
    @property
    def default_hidden_states_mode(self):
        return "join"
    
    @property
    def number_of_features(self):
        return self._number_of_features * self.states_per_state

    def check_model(self, ap_metadata:dict):
        def check(label, none_ok=False):
            if none_ok and ap_metadata.get(label,None) is None: return
            if (a:=self.metadata[label]) != (b:=ap_metadata[label]): raise FeatureExtractorException(f"Inconsistency in {label}: feature extractor has {a}, model expects {b}")

        check("feature_extractor_model", none_ok=True)
        check("hidden_states_mode")
        check("hidden_states_used")

    def _cachefile(self, layer):
        unique_name = self.pretrained + f"_{layer}"
        return os.path.join(self.image_directory,f"featurecache.{unique_name.replace('/','_').replace(':','_')}.safetensors")       
    
    def _ensure_in_cache(self, filepath):
        rel = os.path.relpath(filepath, self.image_directory)
        layers_needed = []
        for layer in self.hidden_states_used:
            if rel not in self.caches[layer]:
                layers_needed.append(layer)
        if layers_needed:
            all_layers = self._get_image_features_tensor(Image.open(filepath), layers=layers_needed)
            for layer in all_layers:
                self.caches[layer][rel] = all_layers[layer].to(self.dtype)
                self.cache_needs_saving[layer] = True
    
    def get_features_from_file(self, filepath):
        self._ensure_in_cache(filepath)
        rel = os.path.relpath(filepath, self.image_directory)
        return self.post_processing(list(self.caches[layer][rel] for layer in self.hidden_states_used)).to(self.dtype)
    
    def precache(self, filepaths, delete_model=True):
        try:
            for f in tqdm(filepaths): self._ensure_in_cache(f)
        finally:
            self._save_cache()
        if delete_model: self._delete_model()

    def clear_cache(self):
        self.caches = {}

    def _save_cache(self):
        for layer in self.caches:
            if self.cache_needs_saving[layer]:
                save_file(self.caches[layer], self._cachefile(layer))
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
    def __init__(self, ap_metadata={}, **kwargs):
        kwargs['hidden_states_used'] = kwargs.pop('hidden_states_used', None) or ap_metadata.get('hidden_states_used', None)
        kwargs['hidden_states_mode'] = kwargs.pop('hidden_states_mode', None) or ap_metadata.get('hidden_states_mode', None)
        super().__init__(**kwargs)

    def _load(self, model_clazz=CLIPModel):
        if self.models.get('model',None) is None: 
            self.models['model'] = model_clazz.from_pretrained(self.pretrained, cache_dir="models")           
            self.models['model'].text_model = None
            self.models['model'].to(self.device)
        if self.models.get('processor',None) is None: 
            self.models['processor'] = AutoProcessor.from_pretrained(self.realname(self.pretrained), cache_dir="models")
        self._number_of_features = self.models['model'].visual_projection.out_features
        self.metadata['number_of_features'] = str(self.number_of_features)

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


class Apple_FeatureExtractor(FeatureExtractor):
    @property
    def default_hidden_states(self):
        config = PretrainedConfig.from_pretrained(self.pretrained, cache_dir="models")
        return list( config.num_blocks-i-1 for i in config.probe_layers )
    
    @property
    def default_hidden_states_mode(self):
        return "average"
    
    def _load(self):
        try:
            from aim.torch.models import AIMForImageClassification
            from aim.torch.data import val_transforms
        except:
            print("AIM not available - pip install git+https://git@github.com/apple/ml-aim.git if you want to use it")
            raise NotImplementedError("AIM not available - pip install git+https://git@github.com/apple/ml-aim.git if you want to use it")
        
        if self.models.get('model', None) is None:
            self.models['model'] = AIMForImageClassification.from_pretrained(self.pretrained, cache_dir="models").to(self.dtype).to(self.device)
            self.models['model'].trunk.post_transformer_layer = None
        if self.models.get('processor', None) is None:
            self.models['processor'] = val_transforms()

        self._number_of_features = self.models['model'].head.bn.num_features

    def _get_image_features_tensor(self, image:Image, layers:list) -> torch.Tensor:
        self._load()
        self.models['model'].to(self.device)
        results = {}
        image = image.convert('RGB') if image.mode!='RGB' else image
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                inp = self.models['processor'](image).unsqueeze(0).to(self.device)    
                x = self.models['model'].preprocessor(inp, mask=None)
                x, features = self.models['model'].trunk(x, mask=None, max_block_id=-1)
                for this_layer in layers:
                    tokens = self.models['model'].trunk.post_trunk_norm(features[-1-this_layer])
                    _, image_features = self.models['model'].head(tokens, mask=None)
                    results[this_layer] = image_features.to(self.dtype).flatten()
        return results
        

class Multi_FeatureExtractor():
    def __init__(self, pretrained:list, hidden_states_used=[0,], stack_hidden_states=False, **kwargs):
        self.feature_extractors = [ FeatureExtractor.get_feature_extractor(p, hidden_states_used=hidden_states_used, stack_hidden_states=stack_hidden_states, **kwargs) for p in pretrained ]
        self.metadata = {"feature_extractor_model":"__".join(pretrained)}
        self.stack_hidden_states = (stack_hidden_states=="True") if isinstance(stack_hidden_states, str) else stack_hidden_states
        self.hidden_states_used = list(int(x) for x in hidden_states_used[1:-1].split(',') if x) if isinstance(hidden_states_used,str) else hidden_states_used
        self.device = kwargs.get('device','cuda')

    def check_model(self, **kwargs):
        FeatureExtractor.check_model(self, **kwargs)
        
    def get_features_from_file(self, **kwargs):
        ift = None
        for fe in self.feature_extractors:
            fe._to(self.device)
            features = fe.get_features_from_file(**kwargs)
            ift = features if ift is None else torch.cat([ift,features])
        return ift

    def precache(self, **kwargs):
        for fe in self.feature_extractors: fe.precache(**kwargs)

    def get_metadata(self):
        return self.metadata
        
    
    @property
    def number_of_features(self):
        return sum(fe.number_of_features for fe in self.feature_extractors)

