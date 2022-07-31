from layers import Layer


class NeuronData:
    def __init__(self, bias=None) -> None:
        self.bias = bias
        self.in_cons = []


class DataLayer(Layer):
    def __init__(self, nurans_count):
        super().__init__(nurans_count, NeuronData)


class NeuralNetworkData:
    def __init__(self, input_neuron_count, output_neuron_count, hidden_layers_count, hidden_layer_neuron_count) -> None:
        self.layers = [DataLayer(hidden_layer_neuron_count)
                       for i in range(hidden_layers_count)]
