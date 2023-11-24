import torch
import torch.nn as nn
from safetensors.torch import load_file
from .clip import CLIP
import os

class AestheticPredictor(nn.Module):
    def __init__(self, clipper:CLIP, input_size=768, pretrained="", device="cuda", dropouts=[0.2,0.2,0.1], relu=True):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024), 
            nn.Dropout(dropouts[0]),          
            nn.Linear(1024, 128),
            nn.Dropout(dropouts[1]),
            nn.Linear(128, 64),
            nn.Dropout(dropouts[2]),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
        if pretrained: 
            sd = load_file(pretrained)
            load_has_relu = not 'layers.2.bias' in sd
            if load_has_relu:
                self.add_relu()
                self.load_state_dict(load_file(pretrained))
                if not relu: print("Reloaded file had ReLU - args['aesthetic_model_relu'] = False being ignored.")
            else:
                self.load_state_dict(load_file(pretrained))
                if relu: self.add_relu()
        else:
            if relu: self.add_relu()
        self.layers.to(device)
        self.device = device
        self.clipper = clipper

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
        def score_file(f):
            return self(self.clipper.prepare_from_file(f)).item()
        
        if eval_mode:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                scores = [(score_file(f), f) for f in files] if as_sorted_tuple else [score_file(f) for f in files]
            if was_training: self.train()
        else:
            scores = [(score_file(f), f) for f in files] if as_sorted_tuple else [score_file(f) for f in files]

        if as_sorted_tuple: scores.sort()

        return scores
    
    def evaluate_image(self, image):
        with torch.no_grad():
            return self(self.clipper.get_image_features_tensor(image))
    
    def evaluate_file(self, file):
        return self.evaluate_files([file], eval_mode=True)[0]
            
    def evaluate_directory(self, directory, as_sorted_tuple=False, eval_mode=False):
        return self.evaluate_files([os.path.join(directory,f) for f in os.listdir(directory)], as_sorted_tuple, eval_mode)
    