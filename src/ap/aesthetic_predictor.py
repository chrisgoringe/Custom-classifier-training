import torch
import torch.nn as nn
from safetensors.torch import load_file
from .feature_extractor import FeatureExtractor
import os, json

class AestheticPredictor(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained:str, use_cache=True, base_directory=None, image_directory=None, explicit_nof=None):
        metadata, _ = cls.load_metadata_and_sd(pretrained=pretrained, return_sd=False)
        fe_model = metadata["feature_extractor_model"]
        if base_directory and os.path.exists(os.path.join(base_directory, fe_model)): fe_model = os.path.join(base_directory, fe_model)
        feature_extractor = FeatureExtractor.get_feature_extractor(pretrained=fe_model, 
                                                                   use_cache=use_cache, 
                                                                   base_directory=base_directory, 
                                                                   image_directory=image_directory) if explicit_nof is None else explicit_nof
        return AestheticPredictor(feature_extractor=feature_extractor, pretrained=pretrained)
    
    def precache(self, image_filepaths:list):
        self.feature_extractor.precache(image_filepaths)

    def to(self, device):
        super().to(device)
        if self.feature_extractor is not None: self.feature_extractor._to(device, load_if_needed=False)
        self.device = device
        return self

    def __init__(self, feature_extractor:FeatureExtractor|int, pretrained, device="cuda", dropouts:list=[], hidden_layer_sizes=None, seed=None, **kwargs):  
        super().__init__()
        if seed: torch.manual_seed(seed)
        self.metadata, sd = self.load_metadata_and_sd(pretrained)
        self.feature_extractor = feature_extractor
        self.scale = lambda a:a # self._set_scale_from_metadata()

        self.output_channels = int(self.metadata['output_channels']) if 'output_channels' in self.metadata else kwargs.get('output_channels',1)
        self.final_layer_bias = self.metadata['final_layer_bias']=="True" if 'final_layer_bias' in self.metadata else kwargs.get('final_layer_bias', True)

        hidden_layer_sizes = [int(x) for x in self.metadata.get('layers','[0]')[1:-1].split(',')] if 'layers' in self.metadata else hidden_layer_sizes
        while len(dropouts) < len(hidden_layer_sizes)+1: dropouts.append(0)

        hidden_layer_sizes = [hidden_layer_sizes,]
        dropouts = [dropouts,]

        for i in range(self.output_channels-1):
            extra_hls = [int(x) for x in self.metadata.get(f"layers_{i}",'[0]')[1:-1].split(',')] if 'layers_0' in self.metadata else kwargs.get(f"hidden_layer_sizes_{i}",None)
            extra_drp = kwargs.get(f"dropouts_{i}",[])
            while len(extra_drp)<len(extra_hls)+1: extra_drp.append(0)
            hidden_layer_sizes.append( extra_hls )
            dropouts.append(extra_drp)

        if "feature_extractor_model" in self.metadata and not isinstance(feature_extractor,int): 
            assert feature_extractor.metadata["feature_extractor_model"].endswith(self.metadata["feature_extractor_model"]), \
                "Mismatched feature extractors : saved file has " + \
                self.metadata["feature_extractor_model"] + " arguments specify " + \
                feature_extractor.metadata["feature_extractor_model"]

        #self.metadata['input_size'] = str(feature_extractor.number_of_features)
        self.metadata['layers'] = str(hidden_layer_sizes[0])
        for i in range(self.output_channels-1): self.metadata[f"layers_{i}"] = str(hidden_layer_sizes[i+1])
        self.metadata['feature_extractor_model'] = self.metadata.get('feature_extractor_model',None) or feature_extractor.metadata["feature_extractor_model"]
        self.metadata['output_channels'] = str(self.output_channels)
        self.metadata['final_layer_bias'] = str(self.final_layer_bias)

        if isinstance(self.feature_extractor,int): 
            nof = feature_extractor
            self.feature_extractor = None
        else:
            nof = feature_extractor.number_of_features

        self.parallel_blocks = torch.nn.ModuleList(
            (self.build_block(nof, hls, dropouts[i]) for i, hls in enumerate(hidden_layer_sizes))
        )

        if sd: self.load_state_dict(sd)
        self.to(device)

    def info(self):
        return {}

    def _set_scale_from_metadata(self):
        raise Exception("Really?")

    def build_block(self, number_of_features, hidden_layer_sizes, dropouts):
        assert not 0 in hidden_layer_sizes, "Zero hidden layer size!"
        b = nn.Sequential()
        current_size = number_of_features
        for i, hidden_layer_size in enumerate(hidden_layer_sizes):
            b.append(nn.Dropout(dropouts[i]))
            b.append(nn.Linear(current_size, hidden_layer_size))
            b.append(nn.ReLU())
            current_size = hidden_layer_size
        b.append(nn.Dropout(dropouts[-1]))
        b.append(nn.Linear(current_size, 1, bias=self.final_layer_bias))
        return b

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
        primary = torch.cat(tuple(block(x) for block in self.parallel_blocks), dim=1)
        return primary
        
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
            
    def evaluate_directory(self, directory, as_sorted_tuple=False, eval_mode=False, output_value=0):
        return self.evaluate_files([os.path.join(directory,f) for f in os.listdir(directory)], as_sorted_tuple, eval_mode, output_value=output_value)
    