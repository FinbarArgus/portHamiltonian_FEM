from fenics import *

class Passive_control_input(UserExpression):
    def __init__(self, t, h_1, input_stop_t, control_start_t, **kwargs):
        super().__init__(kwargs)
        self.t = t
        self.h_1 = h_1
        self.input_stop_t = input_stop_t
        self.control_start_t = control_start_t
        self.eps = 0.001

    def eval_cell(self, value, x, ufc_cell):
        if self.t < self.input_stop_t:
            value[0] = 10*sin(8*pi*self.t)
        elif self.t < self.control_start_t:
            value[0] = 0.0
        else:
            value[0] = -100.0*self.h_1

    def value_shape(self):
        return ()
