import torch, clip
from PIL import Image
from safetensors.torch import save_file, load_file
import os
from transformers import CLIPVisionModel, AutoProcessor

class CLIP:
    @classmethod
    def get_clip(cls, pretrained="ViT-L/14", device="cuda", image_directory="."):
        try:
            return OpenAICLIP(pretrained, device, image_directory)
        except:
            pass
        try:
            return TransformersCLIP(pretrained, device, image_directory)
        except:
            pass

    def setup(self, pretrained, image_directory):
        self.cached = {}
        self.cachefile = os.path.join(image_directory,f"clipcache.{pretrained.replace('/','_')}.safetensors")
        self.image_directory = image_directory
        try:
            self.cached = load_file(self.cachefile, device=self.device)
            print(f"Reloaded CLIPs from {self.cachefile} - delete this file if you don't want to do that")
        except:
            print(f"Didn't reload CLIPs from {self.cachefile}")

    def prepare_from_file(self, filename, device="cuda"):
        rel = os.path.relpath(filename, self.image_directory)
        if rel not in self.cached:
            self.cached[rel] = self.get_image_features_tensor(Image.open(filename))
        return self.cached[rel].to(device)
    
    def save_cache(self):
        save_file(self.cached, self.cachefile)

    def get_metadata(self):
        return self.metadata
    
    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        raise NotImplementedError()


class OpenAICLIP(CLIP):  
    def __init__(self, pretrained="ViT-L/14", device="cuda", image_directory="."):
        self.metadata = {"clip_model":pretrained}
        self.model, self.preprocess = clip.load(pretrained, device=device)
        self.device = device
        self.setup(pretrained, image_directory)

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.to(torch.float)
   
class TransformersCLIP(CLIP):
    def __init__(self, pretrained="", device="cuda", image_directory="."):
        self.metadata = {"clip_model":pretrained}
        self.model = CLIPVisionModel.from_pretrained(pretrained)
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(pretrained)

        self.setup(pretrained, image_directory)

    def get_image_features_tensor(self, image:Image) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            for k in inputs:
                if isinstance(inputs[k],torch.Tensor): inputs[k] = inputs[k].to(self.device)
            outputs = self.model(**inputs)
            return outputs.pooler_output.flatten()
