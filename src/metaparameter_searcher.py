from .time_context import Timer

class ParameterSet:
    def __init__(self, epochs, lr, batch):
        self.epochs = epochs
        self.lr = lr
        self.batch = batch

    @classmethod
    def from_args(cls, args):
        return ParameterSet(args['num_train_epochs'],args['learning_rate'],args['per_device_train_batch_size'])

    def to_args(self, args):
        args['num_train_epochs'] = self.epochs
        args['learning_rate'] = self.lr
        args['per_device_train_batch_size'] = self.batch

    def get_delta_from(self, other):
        return ParameterSet( self.epochs-other.epochs, self.lr-other.lr, self.batch-other.batch )
    
    def add_delta(self, other):
        return ParameterSet( self.epochs+other.epochs, self.lr+other.lr, self.batch+other.batch )
    
    def get_scaled(self, scale:callable):
        return ParameterSet( self.epochs*scale(), self.lr*scale(), self.batch*scale() )
    
    def print(self, f=None):
        txt = "{:>4} epochs, {:>12.3} lr, {:>3} batch size".format(self.epochs, self.lr, self.batch)
        print(txt)
        if f: print(txt,file=f)

class MetaparameterSearcher:
    def __init__(self, initial_parameters:ParameterSet, evaluation_function:callable, new_parameter_function:callable, callback:callable=None, best_so_far_callback:callable=None,
                 scale_decay:float=0.6, scale_gain:float=1.2, minimise=True):
        initial_parameters.print()
        self.parameters = initial_parameters
        self.evalaution_function = evaluation_function
        self.new_parameter_function = new_parameter_function
        self.current_score = None
        self.bad_steps = 0
        self.callback = callback
        self.best_so_far_callback = best_so_far_callback
        self.invert = -1 if minimise else 1

    def search(self, bad_steps_to_end=5):
        with Timer("eval") as t:
            self.current_score = self.evalaution_function(self.parameters)
            if self.callback: self.callback(self.parameters, self.current_score, self.bad_steps, t(None), "")
        while self.bad_steps < bad_steps_to_end:
            with Timer("eval") as t:
                new_params = self.new_parameter_function(self.parameters, self.current_score, self.bad_steps)
                new_score = self.evalaution_function(new_params)
                if (new_score - self.current_score) * self.invert > 0:
                    self.bad_steps = 0
                    self.current_score = new_score
                    self.parameters = new_params
                    if self.best_so_far_callback: self.best_so_far_callback()
                else:
                    self.bad_steps += 1
                if self.callback: self.callback(new_params, new_score, self.bad_steps, t(None), "rejected" if self.bad_steps else "accepted")
        return self.parameters, self.current_score