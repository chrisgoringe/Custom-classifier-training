import math, random
from ..metaparameter_searcher import ParameterSet

class AMP:
    def __init__(self, even_batch):
        self.best_score = None
        self.last_delta = None
        self.even_batch = even_batch
        self.scale_guide = 1.0

    def normalise(self, params:ParameterSet):
        params.epochs = int(params.epochs)
        if params.epochs < 2 : params.epochs = 2
        params.batch = 2*(int(params.batch)//2) if self.even_batch else int(params.batch)
        if params.batch < 2: params.batch = 2
        return params

    def update_mps(self, params:ParameterSet, current_score, badsteps):
        if badsteps==0 and self.last_delta is not None: # we've moved, and it was good, so make save move
            delta = self.last_delta
            #self.scale_guide *= 1.1
        else:
            #self.scale_guide *= 0.8
            delta = params.get_scaled( lambda : math.pow(2,self.scale_guide) * (0.4 * random.random() - 0.2) )
        new_params = params.add_delta(delta)
        self.last_delta = new_params.get_delta_from(params)
        new_params = self.normalise(new_params)
        new_params.print()
        return new_params