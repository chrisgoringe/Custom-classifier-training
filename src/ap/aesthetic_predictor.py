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

    def set_dtype(self, dtype):
        super().to(dtype)
        self.feature_extractor.set_dtype(dtype)

    def to(self, device):
        super().to(device)
        if self.feature_extractor is not None: self.feature_extractor._to(device, load_if_needed=False)
        self.device = device

    def __init__(self, feature_extractor:FeatureExtractor|int, pretrained, device="cuda", dropouts:list=[], hidden_layer_sizes=None, 
                 dropouts_0=[], hidden_layer_sizes_0=None, seed=None, **kwargs):  
        super().__init__()
        if seed: torch.manual_seed(seed)
        self.metadata, sd = self.load_metadata_and_sd(pretrained)
        self.feature_extractor = feature_extractor
        self.scale = lambda a:a # self._set_scale_from_metadata()

        self.high_end_fix = self.metadata['high_end_fix']=="True" if 'high_end_fix' in self.metadata else kwargs['high_end_fix'] if 'high_end_fix' in kwargs else False
        self.output_channels = int(self.metadata['output_channels']) if 'output_channels' in self.metadata else kwargs.get('output_channels',1)

        hidden_layer_sizes = [int(x) for x in self.metadata.get('layers','[0]')[1:-1].split(',')] if 'layers' in self.metadata else hidden_layer_sizes
        while len(dropouts) < len(hidden_layer_sizes)+1: dropouts.append(0)

        hidden_layer_sizes_0 = [int(x) for x in self.metadata.get('layers_0','[0]')[1:-1].split(',')] if 'layers_0' in self.metadata else hidden_layer_sizes_0
        if hidden_layer_sizes_0: 
            while len(dropouts_0) < len(hidden_layer_sizes_0)+1: dropouts_0.append(0)


        if "feature_extractor_model" in self.metadata and not isinstance(feature_extractor,int): 
            assert feature_extractor.metadata["feature_extractor_model"].endswith(self.metadata["feature_extractor_model"]), \
                "Mismatched feature extractors : saved file has " + \
                self.metadata["feature_extractor_model"] + " arguments specify " + \
                feature_extractor.metadata["feature_extractor_model"]

        #self.metadata['input_size'] = str(feature_extractor.number_of_features)
        self.metadata['layers'] = str(hidden_layer_sizes)
        if hidden_layer_sizes_0: self.metadata['layers_0'] = str(hidden_layer_sizes_0)
        self.metadata['high_end_fix'] = str(self.high_end_fix)
        self.metadata['feature_extractor_model'] = self.metadata.get('feature_extractor_model',None) or feature_extractor.metadata["feature_extractor_model"]
        self.metadata['output_channels'] = str(self.output_channels)

        #self.parallel_blocks = torch.nn.ModuleList(
        #    self.build_block(feature_extractor.number_of_features, hidden_layer_sizes, dropouts) for _ in range(self.output_channels)
        #)
        nof = feature_extractor if isinstance(feature_extractor,int) else feature_extractor.number_of_features
        if isinstance(self.feature_extractor,int): self.feature_extractor = None
        self.parallel_blocks = torch.nn.ModuleList(
            (self.build_block(nof, hidden_layer_sizes, dropouts),
            self.build_block(nof, hidden_layer_sizes_0, dropouts_0),)
        ) if self.output_channels==2 else torch.nn.ModuleList(
            (self.build_block(nof, hidden_layer_sizes, dropouts),)
        )

        if self.high_end_fix:
            raise NotImplementedError("High End fix diabled")
            self.high_end = self.build_block(feature_extractor.number_of_features, hidden_layer_sizes, dropouts)
            if ('variable_hef' in kwargs and kwargs['variable_hef']) or 'high_end_weighter.0.bias' in sd:
                self.high_end_weighter = nn.Sequential( nn.Linear(1,1), nn.Tanh(), nn.Linear(1,1) )
                self.set_initial_hef(self.high_end_weighter, kwargs.get("variable_hef_init", [5, -3, 0.5, 0.5]))
            else:
                self.high_end_weighter = lambda a : (1+torch.tanh((a - 0.6)*5))*0.5
        else:
            self.high_end_weighter = None

        if sd: self.load_state_dict(sd)
        self.to(device)
        print(f"{self.info()}")

        

    def set_initial_hef(self, hef, v):
        hef.requires_grad_(False)
        hef[0].weight[0] = v[0]
        hef[0].bias[0] = v[1]
        hef[2].weight[0] = v[2]
        hef[2].bias[0] = v[3]
        hef.requires_grad_(True)   

    def info(self):
        if isinstance(self.high_end_weighter,nn.Module):
            return self.high_end_weighter.state_dict()
        return {}

    def _set_scale_from_metadata(self):
        raise Exception("Really?")
        if 'mean_predicted_score' in self.metadata:
            mean = float(self.metadata['mean_predicted_score'])
            std = float(self.metadata['stdev_predicted_score'])
            return lambda a : float((a-mean)/std)
        else:
            return lambda a : float(a)

    def build_block(self, number_of_features, hidden_layer_sizes, dropouts):
        b = nn.Sequential()
        current_size = number_of_features
        for i, hidden_layer_size in enumerate(hidden_layer_sizes):
            b.append(nn.Dropout(dropouts[i]))
            b.append(nn.Linear(current_size, hidden_layer_size))
            current_size = hidden_layer_size
            b.append(nn.ReLU())
        b.append(nn.Dropout(dropouts[-1]))
        b.append(nn.Linear(current_size, 1))
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
        if self.high_end_fix:
            raise NotImplementedError()
            high_end = self.high_end(x)
            high_weight = self.high_end_weighter(primary) 
            primary = primary + high_end * high_weight 
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
    
    def evaluate_file_sigma(self, file):
        return torch.abs(self.evaluate_file(file, output_value=1)) if self.output_channels>1 else 1.0

    def evaluate_file(self, file, output_value=0):
        return self.evaluate_files([file], output_value=output_value)
            
    def evaluate_directory(self, directory, as_sorted_tuple=False, eval_mode=False, output_value=0):
        return self.evaluate_files([os.path.join(directory,f) for f in os.listdir(directory)], as_sorted_tuple, eval_mode, output_value=output_value)
    