from neural_toolbox.neurons import LayerNuran, Neuron


class Layer(list):
    def __init__(self, nurans_count, nurans_type=LayerNuran):
        for i in range(nurans_count):
            self.append(nurans_type())


class InputLayer(Layer):
    def __init__(self, nurans_count):
        super().__init__(nurans_count, Neuron)
