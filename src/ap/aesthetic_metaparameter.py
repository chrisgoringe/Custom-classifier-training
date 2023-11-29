import math, random
from ..metaparameter_searcher import ParameterSet

class AMP:
    def __init__(self, even_batch):
        self.best_score = None
        self.last_delta = None
        self.even_batch = even_batch
        self.scale_guide = 1.0

    def update_mps(self, params:ParameterSet, current_score, badsteps):
        if badsteps==0 and self.last_delta is not None: # we've moved, and it was good, so make save move
            delta = self.last_delta
        else:
            delta = params.get_scaled( lambda : math.pow(2,self.scale_guide) * (0.4 * random.random() - 0.2) )
            delta.arguments['per_device_train_batch_size'] = 2 if random.random()<0.5 else -2
        new_params = params.add_delta(delta)
        self.last_delta = new_params.get_delta_from(params)
        new_params = new_params.normalise()
        new_params.print()
        return new_params