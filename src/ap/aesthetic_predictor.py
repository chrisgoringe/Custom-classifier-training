import torch
import torch.nn as nn
from safetensors.torch import load_file
from .clip import CLIP
import os, json

class AestheticPredictor(nn.Module):
    def __init__(self, clipper:CLIP, input_size=768, pretrained="", device="cuda", dropouts=[0.2,0.2,0.1], relu=True, layers=None):
        super().__init__()
        self.metadata, sd = self.load_metadata_and_sd(pretrained)
        layers = layers or [1024,128,64,16]

        assert self.metadata.get('input_size', str(input_size)) == str(input_size) , "Inconsistency in input_size"
        self.metadata['input_size'] = str(input_size)
        assert self.metadata.get('relu', str(relu)) == str(relu) , "Inconsistency in relu"
        self.metadata['relu'] = str(relu)
        assert self.metadata.get('layers', str(layers)) == str(layers), "Inconsistency in layers"
        self.metadata['layers'] = str(layers)
        assert len(layers)==len(dropouts)+1, f"must have layers (have {len(layers)}) must equal dropouts (have {len(dropouts)})+1"
        
        self.input_size = input_size
        
        if len(layers)==4:
            self.layers = nn.Sequential(
                nn.Linear(self.input_size, layers[0]), 
                nn.Dropout(dropouts[0]),          
                nn.Linear(layers[0], layers[1]),
                nn.Dropout(dropouts[1]),
                nn.Linear(layers[1], layers[2]),
                nn.Dropout(dropouts[2]),
                nn.Linear(layers[2], layers[3]),
                nn.Linear(layers[3], 1)
            )

            pretrained_has_relu = sd and not 'layers.2.bias' in sd
            if pretrained_has_relu: self.add_relu()
            if sd: self.load_state_dict(load_file(pretrained))
            if relu and not pretrained_has_relu: self.add_relu()
        else:
            self.layers = nn.Sequential( 
                nn.Linear(self.input_size, layers[0]),
                nn.ReLU() 
            )
            for i in range(1,len(layers)):
                self.layers.append(nn.Dropout(dropouts[i-1]))
                self.layers.append(nn.Linear(layers[i-1], layers[i]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(layers[-1], 1))
        
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

    def add_relu(self):
        self.layers = nn.Sequential(
            self.layers[0], nn.ReLU(),
            self.layers[1], self.layers[2], nn.ReLU(),
            self.layers[3], self.layers[4], nn.ReLU(),
            self.layers[5], self.layers[6], nn.ReLU(),
            self.layers[7]
        )

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
                #scores = [(score_file(f), f) for f in files] if as_sorted_tuple else [score_file(f) for f in files]
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
    