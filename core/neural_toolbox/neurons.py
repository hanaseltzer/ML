from math_utils import sigmoid, get_value


class Neuron:
    def __init__(self, value=0):
        self.value = value


class LayerNuran(Neuron):
    def __init__(self, value=0, in_cons=None, bias=None):
        super().__init__(value)
        self.in_cons = in_cons if in_cons is not None else []
        self.bias = bias if bias else None
        self.back_propagation_value = 0

    def calc_value(self):
        # sigmoid(w1*a1 + w2*a2 + ... + wn*an + b)
        value_to_sigmoid = 0
        for con in self.in_cons:
            value_to_sigmoid += con.weight * con.in_nuran.value
        value_to_sigmoid += self.bias
        self.value = sigmoid(value_to_sigmoid)
