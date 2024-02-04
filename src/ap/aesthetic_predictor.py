import torch
import torch.nn as nn
from safetensors.torch import load_file
from .feature_extractor import FeatureExtractor
import os, json

class AestheticPredictor(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained:str, use_cache=True, base_directory=None):
        metadata, _ = cls.load_metadata_and_sd(pretrained=pretrained, return_sd=False)
        fe_model = metadata["feature_extractor_model"]
        return AestheticPredictor(feature_extractor=FeatureExtractor.get_feature_extractor(pretrained=fe_model, use_cache=use_cache, base_directory=base_directory), pretrained=pretrained)

    def set_dtype(self, dtype):
        super().to(dtype)
        self.feature_extractor.set_dtype(dtype)
        self.dtype = dtype

    def to(self, device):
        super().to(device)
        self.feature_extractor._to(device, load_if_needed=False)
        self.device = device

    def __init__(self, feature_extractor:FeatureExtractor, pretrained, device="cuda", dropouts:list=[], hidden_layer_sizes=None, seed=42, **kwargs):  
        super().__init__()
        torch.manual_seed(seed)
        self.dtype = torch.float
        self.metadata, sd = self.load_metadata_and_sd(pretrained)

        hidden_layer_sizes = [int(x) for x in self.metadata.get('layers','[0]')[1:-1].split(',')] if 'layers' in self.metadata else hidden_layer_sizes
        while len(dropouts) < len(hidden_layer_sizes)+1: dropouts.append(0)

        if "feature_extractor_model" in self.metadata: 
            assert self.metadata["feature_extractor_model"] == feature_extractor.metadata["feature_extractor_model"], \
                "Mismatched feature extractors : saved file has " + \
                self.metadata["feature_extractor_model"] + " arguments specify " + \
                feature_extractor.metadata["feature_extractor_model"]

        self.metadata['input_size'] = str(feature_extractor.number_of_features)
        self.metadata['layers'] = str(hidden_layer_sizes)

        self.layers = nn.Sequential( )
        current_size = feature_extractor.number_of_features
        for i, hidden_layer_size in enumerate(hidden_layer_sizes):
            self.layers.append(nn.Dropout(dropouts[i]))
            self.layers.append(nn.Linear(current_size, hidden_layer_size))
            current_size = hidden_layer_size
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropouts[-1]))
        self.layers.append(nn.Linear(current_size, 1))

        if sd: self.load_state_dict(sd)
        
        self.layers.to(device)
        self.device = device
        self.feature_extractor = feature_extractor

        if 'mean_predicted_score' in self.metadata:
            mean = float(self.metadata['mean_predicted_score'])
            std = float(self.metadata['stdev_predicted_score'])
            self.scale = lambda a : float((a-mean)/std)
        else:
            self.scale = lambda a : float(a)

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
        
    def get_metadata(self):
        return self.metadata

    def forward(self, x, **kwargs):
        return self.layers(x)
        
    def evaluate_image(self, img):
        return self(self.feature_extractor._get_image_features_tensor(img).to(self.device))
    
    def evaluate_files(self, files, as_sorted_tuple=False, eval_mode=False):
        def score_files(fs):
            data = torch.stack(list(self.feature_extractor.get_features_from_file(f) for f in fs))
            return self(data)
        
        def score_file(f):
            return self(self.feature_extractor.get_features_from_file(f)).item()
        
        if eval_mode:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                scores = score_files(files).cpu().flatten()
            if was_training: self.train()
        else:
            scores = [score_file(f) for f in files]

        if as_sorted_tuple: 
            scores = [(scores[i], f) for i,f in enumerate(files)]
            scores.sort()

        return scores

    def evaluate_file(self, file):
        return self.evaluate_files([file], eval_mode=True)[0]
            
    def evaluate_directory(self, directory, as_sorted_tuple=False, eval_mode=False):
        return self.evaluate_files([os.path.join(directory,f) for f in os.listdir(directory)], as_sorted_tuple, eval_mode)
    