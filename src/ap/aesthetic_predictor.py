import torch
import torch.nn as nn
from safetensors.torch import load_file
from .feature_extractor import FeatureExtractor
import json

def to_bool(s): 
    if isinstance(s,str): return (s=="True")
    if isinstance(s,bool): return s
    raise NotImplementedError()

def to_int_list(s):
    if isinstance(s,str): 
        s = s[1:-1] if s.startswith('[') else s
        return list(int(x) for x in s.split(',') if x)
    if isinstance(s,list): return s
    raise NotImplementedError()

def to_float_list(s):
    if isinstance(s,str): return list(float(x) for x in s[1:-1].split(',') if x)
    if isinstance(s,list): return s
    raise NotImplementedError()

class FakeFE():
    number_of_features = None
    check_model = lambda a,b,c : True

class AestheticPredictor(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained:str, feature_extractor=None, **feargs):
        metadata, _ = cls.load_metadata_and_sd(pretrained=pretrained, return_sd=False)
        feature_extractor = feature_extractor or FeatureExtractor.get_feature_extractor(pretrained=metadata["feature_extractor_model"], ap_metadata=metadata, **feargs)
        return AestheticPredictor(feature_extractor=feature_extractor, pretrained=pretrained)
    
    @classmethod
    def no_feature_extractor(cls, pretrained:str):
        return AestheticPredictor(pretrained=pretrained)
    
    def precache(self, image_filepaths:list):
        self.feature_extractor.precache(image_filepaths)

    def to(self, device):
        super().to(device)
        if self.feature_extractor is not None: self.feature_extractor._to(device, load_if_needed=False)
        self.device = device
        return self
    
    def _get_argument(self, p:str, default, cast:callable):
        value = self.metadata[p] if p in self.metadata else (self.kwargs[p] if p in self.kwargs else default)
        if value is None: raise Exception(f"No value found for {p}")
        if isinstance(value, list): self.metadata[p] = ",".join(str(x) for x in value)
        else: self.metadata[p] = str(value)
        return cast(value)

    def __init__(self, feature_extractor:FeatureExtractor=FakeFE(), pretrained="", device="cuda", model_seed=None, **kwargs):  
        super().__init__()
        
        self.metadata, sd = self.load_metadata_and_sd(pretrained)
        self.kwargs = kwargs
        self.feature_extractor = feature_extractor

        self.hidden_states_used     = self._get_argument('hidden_states_used',      feature_extractor.hidden_states_used, to_int_list)
        self.hidden_states_mode     = self._get_argument('hidden_states_mode',      "join", str)      
        self.number_of_features     = self._get_argument('number_of_features',      feature_extractor.number_of_features, int)
        
        self.hidden_layer_sizes     = self._get_argument('layers',                  None,   to_int_list)
        self.output_channels        = self._get_argument('output_channels',         1,      int)
        self.dropouts               = self._get_argument('dropouts',                [],     to_int_list)
        self.metadata.pop('dropouts')

        if self.hidden_states_mode == "default": 
            self.hidden_states_mode = self.feature_extractor.default_hidden_states_mode
            self.metadata['hidden_states_mode'] = self.hidden_states_mode
        
        self.feature_extractor.check_model(self.metadata)

        if model_seed: torch.manual_seed(model_seed)

        if self.hidden_states_mode=="weight":
            self.weight = nn.Sequential(
                nn.Linear(len(self.hidden_states_used), 1),
                nn.ReLU()
            )            
            self.preprocess = lambda a : self.weight(torch.transpose(a,-2,-1)).reshape((a.shape[0],-1))
            if (weights := kwargs.get('fixed_hidden_state_weights',None)):
                old = self.weight[0]
                details = {"dtype":old.weight.dtype, "device":old.weight.device}
                self.weight[0].weight = torch.nn.parameter.Parameter(torch.tensor(data=weights[:-1], **details).reshape_as(old.weight), requires_grad=False)
                self.weight[0].bias = torch.nn.parameter.Parameter(torch.tensor(data=weights[-1], **details).reshape_as(old.bias), requires_grad=False)
        else:
            self.preprocess = lambda a : a

        self.main_process = nn.Sequential()
        current_size = self.number_of_features
        for i, hidden_layer_size in enumerate(self.hidden_layer_sizes):
            self.main_process.append(nn.Dropout(self.dropouts[i] if i<len(self.dropouts) else 0.0))
            self.main_process.append(nn.Linear(current_size, hidden_layer_size))
            self.main_process.append(nn.ReLU())
            current_size = hidden_layer_size
        self.main_process.append(nn.Dropout(self.dropouts[-1] if self.dropouts else 0.0))
        self.main_process.append(nn.Linear(current_size, self.output_channels))

        if sd: self.load_state_dict(sd)
        self.to(device)

    def info(self):
        if self.preprocess:
            return {"preprocess_weights" : ",".join("{:>9.5f}".format(x.item()) for x in self.weight[0].weight.squeeze()),
                    "preprocess_bias" :  "{:>9.5f}".format(self.weight[0].bias.item())}
        return {}
    
    def is_weight_parameter(self, parameter_name):
        return parameter_name.startswith('weight')

    @classmethod
    def load_metadata_and_sd(cls, pretrained, return_sd=True):
        if pretrained:
            with open(pretrained, "rb") as f:
                data = f.read()
                n_header = data[:8]
            n = int.from_bytes(n_header, "little")
            metadata_bytes = data[8 : 8 + n]
            header = json.loads(metadata_bytes)
            return header.get("__metadata__", {}), load_file(pretrained) if return_sd else {}
        else:
            return {}, {}
        
    def get_metadata(self): return self.metadata

    def forward(self, x, **kwargs) -> torch.Tensor:
        return self.main_process(self.preprocess(x))
        
    def evaluate_image(self, img):
        return self(self.feature_extractor._get_image_features_tensor(img).to(self.device))
    
    def evaluate_files(self, files, as_sorted_tuple=False, output_value=0):
        def score_files(fs):
            data = torch.stack(list(self.feature_extractor.get_features_from_file(f) for f in fs))
            return self(data)[:,output_value] if output_value is not None else self(data)
        
        scores = score_files(files).cpu()

        if as_sorted_tuple: 
            scores = [(scores[i][0], f) for i,f in enumerate(files)]
            scores.sort()

        return scores

    def evaluate_file(self, file, output_value=0):
        return self.evaluate_files([file], output_value=output_value)
            
#import os
#   def evaluate_directory(self, directory, as_sorted_tuple=False, eval_mode=False, output_value=0):
#       return self.evaluate_files([os.path.join(directory,f) for f in os.listdir(directory)], as_sorted_tuple, eval_mode, output_value=output_value)
    