from .time_context import Timer

def make_constraint(require_int=False, require_even=False, absolute=False, minimum=None, maximum=None):
    def constraint(a):
        if require_int: a = int(a)
        if require_even: a = 2*(a//2)
        if minimum and a<minimum: a = minimum
        if maximum and a>maximum: a = maximum
        if absolute and a<0: a = -1
        return a
    return constraint
    
class ParameterSet:
    arg_names = ['num_train_epochs', 'learning_rate', 'per_device_train_batch_size', 'warmup_ratio']
    constraints = { 
        'num_train_epochs' : make_constraint(require_int=True, minimum=2),
        'learning_rate' : make_constraint(absolute=True),
        'per_device_train_batch_size' : make_constraint(require_int=True, require_even=True, minimum=2, maximum=128),
        'warmup_ratio' : make_constraint(minimum=0.0, maximum=1.0)
    }
    txt = "{:>4} epochs, {:>12.3} lr, {:>3} batch size, {:>5.3f} warmup ratio"

    def __init__(self, args={}):
        self.arguments = args

    @classmethod
    def from_args(cls, args):
        return ParameterSet({arg_name:args[arg_name] for arg_name in cls.arg_names})

    def to_args(self, args):
        for arg_name in self.arg_names:
            args[arg_name] = self.arguments[arg_name]

    def get_delta_from(self, other):
        return ParameterSet( {arg_name:self.arguments[arg_name] - other.arguments[arg_name] for arg_name in self.arguments} )
    
    def add_delta(self, other):
        return ParameterSet( {arg_name:self.arguments[arg_name] + other.arguments[arg_name] for arg_name in self.arguments} )
    
    def get_scaled(self, scale:callable):
        return ParameterSet( {arg_name:self.arguments[arg_name]*scale() for arg_name in self.arguments} )
    
    def normalise(self):
        return ParameterSet( {arg_name:self.constraints[arg_name](self.arguments[arg_name]) for arg_name in self.arguments} )

    @property
    def description(self):
        return self.txt.format( *(self.arguments[arg_name] for arg_name in self.arg_names) )
    
    def print(self, f=None):
        txt = self.description
        print(txt)
        if f: print(txt,file=f)

class MetaparameterSearcher:
    def __init__(self, initial_parameters:ParameterSet, evaluation_function:callable, new_parameter_function:callable, callback:callable=None, best_so_far_callback:callable=None,
                 minimise=True):
        initial_parameters.print()
        self.parameters = initial_parameters
        self.evaluation_function = evaluation_function
        self.new_parameter_function = new_parameter_function
        self.current_score = None
        self.bad_steps = 0
        self.callback = callback
        self.best_so_far_callback = best_so_far_callback
        self.invert = -1 if minimise else 1

    def search(self, bad_steps_to_end=5):
        with Timer("eval") as t:
            self.current_score, self.all_score = self.evaluation_function(self.parameters)
            if self.callback: self.callback(self.parameters, self.current_score, self.all_score, self.bad_steps, t(None), "")
        while self.bad_steps < bad_steps_to_end:
            with Timer("eval") as t:
                new_params = self.new_parameter_function(self.parameters, self.current_score, self.bad_steps)
                new_score, self.all_score = self.evaluation_function(new_params)
                if (new_score - self.current_score) * self.invert > 0:
                    self.bad_steps = 0
                    self.current_score = new_score
                    self.parameters = new_params
                    if self.best_so_far_callback: self.best_so_far_callback()
                else:
                    self.bad_steps += 1
                if self.callback: self.callback(new_params, new_score, self.all_score, self.bad_steps, t(None), "rejected" if self.bad_steps else "")
        return self.parameters, self.current_score