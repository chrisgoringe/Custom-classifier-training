import torch

class Patched(torch.nn.Module):
    def __init__(self, in_channels:int, dim:int, out_channels:int, dtype=None, device=None):
        super().__init__()
        self.p = torch.nn.Sequential(
            torch.nn.Linear(in_channels, dim,  bias=False, dtype=dtype, device=device),
            torch.nn.Linear(dim, out_channels, bias=False, dtype=dtype, device=device),
        )
        self.weight = 1.0

    def hook(self, module, args, output):
        return output + self(*args) * self.weight

    def forward(self, h):
        return self.p(h)

class PatchedModelWrapper:
    def __init__(self, model:torch.nn.Module, dim, dtype=None, device=None):
        self.model = model
        self.dim = dim
        self.dtype = dtype
        self.device = device
        self.patches = {}
        for n,m in self.model.named_modules(): 
            m.requires_grad_(self.require_grad(n,m))
            if self.should_patch(n,m):
                self.add_patch(n,m)

    def add_patch(self, module_name, module:torch.nn.Module):
        patch = Patched(module.in_features, self.dim, module.out_features, self.dtype, self.device)
        module.register_forward_hook(patch.hook)
        print(f"Patched {module_name} {module.in_features} x {self.dim} x {module.out_features}")
        self.patches[module_name] = patch

    def require_grad(self, module_name, module):
        rg = "classifier" in module_name or "embeddings" in module_name or "vit.layernorm"==module_name
        #if rg: print(f"Retaining requires_grad for {module_name}")
        #else:  print(f"Dropping requires_grad for {module_name}")
        return rg
    
    def should_patch(self, module_name, module):
        return "attention.attention." in module_name and "dropout" not in module_name
    
    def save(self, filepath):
        raise NotImplementedError()

    def load(self, filepath):
        raise NotImplementedError()