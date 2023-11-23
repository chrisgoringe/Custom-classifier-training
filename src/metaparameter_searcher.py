from .time_context import Timer

class MetaparameterSearcher:
    def __init__(self, initial_parameters:list, evaluation_function:callable, new_parameter_function:callable, callback:callable=None, best_so_far_callback:callable=None,
                 scale_decay:float=0.6, scale_gain:float=1.2, minimise=True):
        self.parameters = initial_parameters
        self.evalaution_function = evaluation_function
        self.new_parameter_function = new_parameter_function
        self.current_score = None
        self.bad_steps = 0
        self.scale_indication = 1.0
        self.scale_decay = scale_decay
        self.scale_gain = scale_gain
        self.callback = callback
        self.best_so_far_callback = best_so_far_callback
        self.invert = -1 if minimise else 1

    def search(self, bad_steps_to_end=5):
        with Timer("eval") as t:
            self.current_score = self.evalaution_function(self.parameters)
            if self.callback: self.callback(self.parameters, self.current_score, self.bad_steps, self.scale_indication, t(None), "")
        while self.bad_steps < bad_steps_to_end:
            with Timer("eval") as t:
                new_params = self.new_parameter_function(self.parameters, self.scale_indication)
                new_score = self.evalaution_function(new_params)
                if (new_score - self.current_score) * self.invert > 0:
                    self.bad_steps = 0
                    self.scale_indication *= self.scale_gain
                    self.current_score = new_score
                    self.parameters = new_params
                    if self.best_so_far_callback: self.best_so_far_callback()
                else:
                    self.bad_steps += 1
                    self.scale_indication *= self.scale_decay
                if self.callback: self.callback(new_params, new_score, self.bad_steps, self.scale_indication, t(None), "rejected" if self.bad_steps else "accepted")
        return self.parameters, self.current_score