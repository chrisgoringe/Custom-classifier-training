import torch, clip
from PIL import Image
from safetensors.torch import save_file, load_file
import os

class CLIP:
    def __init__(self, pretrained="ViT-L/14", device="cuda", image_directory="."):
        self.metadata = {"clip_model":pretrained}
        self.model, self.preprocess = clip.load(pretrained, device=device)
        self.device = device
        self.cached = {}
        self.cachefile = os.path.join(image_directory,f"clipcache.{pretrained.replace('/','_')}.safetensors")
        self.image_directory = image_directory
        try:
            self.cached = load_file(self.cachefile, device=self.device)
            print(f"Reloaded CLIPs from {self.cachefile} - delete this file if you don't want to do that")
        except:
            print(f"Didn't reload CLIPs from {self.cachefile}")

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.to(torch.float)
   
    def prepare_from_file(self, filename, device="cuda"):
        rel = os.path.relpath(filename, self.image_directory)
        if rel not in self.cached:
            self.cached[rel] = self.get_image_features_tensor(Image.open(filename))
        return self.cached[rel].to(device)
    
    def save_cache(self):
        save_file(self.cached, self.cachefile)

    def get_metadata(self):
        return self.metadata
