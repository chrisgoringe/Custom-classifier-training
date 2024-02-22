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
    if isinstance(s,str): return list(int(x) for x in s[1:-1].split(',') if x)
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
        return AestheticPredictor(feature_extractor=None, pretrained=pretrained)
    
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
        self.metadata[p] = str(value)
        return cast(value)

    def __init__(self, feature_extractor:FeatureExtractor=FakeFE(), pretrained="", device="cuda", model_seed=None, **kwargs):  
        super().__init__()
        
        self.metadata, sd = self.load_metadata_and_sd(pretrained)
        self.kwargs = kwargs
        self.feature_extractor = feature_extractor

        self.output_channels        = self._get_argument('output_channels',         1,      int)
        self.stack_hidden_states    = self._get_argument('stack_hidden_states',     0,      to_bool)
        self.hidden_states_used     = self._get_argument('hidden_states_used',      [0,],   to_int_list)
        self.hidden_layer_sizes     = self._get_argument('layers',                  None,   to_int_list)
        self.number_of_features     = self._get_argument('number_of_features',      feature_extractor.number_of_features,    int)
        self.dropouts               = self._get_argument('dropouts',                [],     to_int_list)
        self.metadata.pop('dropouts')        
        
        self.feature_extractor.check_model(self.metadata.get("feature_extractor_model", None), self.hidden_states_used, self.stack_hidden_states)

        if model_seed: torch.manual_seed(model_seed)

        self.preprocess = nn.Sequential(
            nn.Linear(len(self.hidden_states_used), 1),
            nn.ReLU()
         ) if self.stack_hidden_states else None

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
            ws = self.preprocess[0].weight.squeeze()
            b = self.preprocess[0].bias
            last = float(ws[-1].item())
            return { "normalised_hidden_layer_projection" : ",".join("{:>8.4f}".format(x.item()/last) for x in ws) + " (bias {:>8.4f}".format(b.item()) }
        return {}

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
        if self.preprocess:
            xp = self.preprocess(x.permute((0,2,1)))
            xp = xp.reshape((x.shape[0],-1))
        else:
            xp = x
        return self.main_process(xp)
        
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
    