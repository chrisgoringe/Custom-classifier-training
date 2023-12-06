import torch
import torch.nn as nn
from safetensors.torch import load_file
from .clip import CLIP
import os, json

class AestheticPredictor(nn.Module):
    def __init__(self, clipper:CLIP, input_size=768, pretrained="", device="cuda", dropouts:list=None, hidden_layer_sizes=None, seed=42):  
        super().__init__()
        torch.manual_seed(seed)
        self.metadata, sd = self.load_metadata_and_sd(pretrained)

        hidden_layer_sizes = hidden_layer_sizes or [int(x) for x in self.metadata.get('layers','[0]')[1:-1].split(',')]
        dropouts = dropouts or [0]*len(hidden_layer_sizes)
        while len(dropouts) < len(hidden_layer_sizes): dropouts.append(0)

        self.metadata['input_size'] = str(input_size)
        self.metadata['layers'] = str(hidden_layer_sizes)

        if dropouts[-1]!=0: print("Last dropout non-zero - that's probably not a great idea...")
        
        self.layers = nn.Sequential( )
        current_size = input_size
        for i, hidden_layer_size in enumerate(hidden_layer_sizes):
            self.layers.append(nn.Linear(current_size, hidden_layer_size))
            current_size = hidden_layer_size
            self.layers.append(nn.Dropout(dropouts[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(current_size, 1))

        if sd: self.load_state_dict(sd)
        
        self.layers.to(device)
        self.device = device
        self.clipper = clipper

        if 'mean_predicted_score' in self.metadata:
            mean = float(self.metadata['mean_predicted_score'])
            std = float(self.metadata['stdev_predicted_score'])
            self.scale = lambda a : float((a-mean)/std)
        else:
            self.scale = lambda a : float(a)

    def load_metadata_and_sd(self, pretrained):
        if pretrained:
            with open(pretrained, "rb") as f:
                data = f.read()
                n_header = data[:8]
            n = int.from_bytes(n_header, "little")
            metadata_bytes = data[8 : 8 + n]
            header = json.loads(metadata_bytes)
            return header.get("__metadata__", {}), load_file(pretrained)
        else:
            return {}, {}
        
    def get_metadata(self):
        return self.metadata

    def forward(self, x, **kwargs):
        return self.layers(x)
    
    def evaluate_files(self, files, as_sorted_tuple=False, eval_mode=False):
        def score_files(fs):
            data = torch.stack(list(self.clipper.prepare_from_file(f) for f in fs))
            return self(data)
        
        def score_file(f):
            return self(self.clipper.prepare_from_file(f)).item()
        
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
    
    def evaluate_image(self, image):
        with torch.no_grad():
            return self(self.clipper.get_image_features_tensor(image))
    
    def evaluate_file(self, file):
        return self.evaluate_files([file], eval_mode=True)[0]
            
    def evaluate_directory(self, directory, as_sorted_tuple=False, eval_mode=False):
        return self.evaluate_files([os.path.join(directory,f) for f in os.listdir(directory)], as_sorted_tuple, eval_mode)
    