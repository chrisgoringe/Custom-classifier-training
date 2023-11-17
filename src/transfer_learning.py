import torch
import re

class TransferLearning:
    nop = 0
    freeze = 1
    replace = 2
    std = 0.01

    @classmethod
    def init(cls,m):
        if hasattr(m,'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight,0,cls.std)
        if hasattr(m,'bias') and m.bias is not None:
            torch.nn.init.normal_(m.bias,0,cls.std)

    @classmethod    
    def prepare_model_for_transfer_learning(cls, model:torch.nn.Module, layer_rule):
        for n,m in model.named_modules():
            if len(list(m.parameters(False)))==0:
                continue

            rule = layer_rule(n)
            print(f"{n} {['thaw','freeze','replace'][rule]}")
            if rule==TransferLearning.replace:
                m.apply(cls.init)
            m.requires_grad_(rule!=TransferLearning.freeze)

    @classmethod
    def createRule(cls, replace, thaw):
        def layer(n:str):
            if n=="classifier": return 13
            if n=="vit.layernorm": return 12
            r = re.match('.*layer\.([0-9]{1,2})\.',n)
            if r: return int(r.group(1))
            return -1
        def rule(n):
            l = layer(n)
            if l>13-replace: return TransferLearning.replace
            if l>13-replace-thaw: return TransferLearning.nop
            return TransferLearning.freeze
        return rule


