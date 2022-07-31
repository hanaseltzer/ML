from neural_toolbox.connections import Connection
from neural_toolbox.layers import Layer, InputLayer
from math_utils import array_2_number, get_value, number_2_array


class NeuralNetwork:
    def __init__(self, input_neuron_count, output_neuron_count, hidden_layers_count, hidden_layer_neuron_count, weights_and_biases=None):
        self.layers = self.create_layers(
            input_neuron_count, output_neuron_count, hidden_layers_count, hidden_layer_neuron_count)
        self.set_weights_and_biases(weights_and_biases)

    def create_layers(self, input_neuron_count, output_neuron_count, hidden_layers_count, hidden_layer_neuron_count):
        layers = [InputLayer(input_neuron_count)]
        for i in range(hidden_layers_count):
            layers.append(Layer(hidden_layer_neuron_count))
        layers.append(Layer(output_neuron_count))
        return layers

    def set_weights_and_biases(self, weights_and_biases_=None):
        if not weights_and_biases_:
            weights_and_biases = weights_and_biases_
        else:
            weights_and_biases = weights_and_biases_.copy()
        previous_layer = self.layers[0]
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.bias = weights_and_biases.pop() if weights_and_biases else get_value()
                if previous_layer:
                    for previous_layer_neuron in previous_layer:
                        conn = Connection(previous_layer_neuron, weights_and_biases.pop()
                                          if weights_and_biases else get_value())
                        neuron.in_cons.append(conn)
            previous_layer = layer

    def calc(self, inputs):
        for neuron, input_value in zip(self.layers[0], inputs):
            neuron.value = input_value
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.calc_value()
        return [neuron.value for neuron in self.layers[-1]]

    def get_cost(self, expected_output, received_output=None):
        received_output = received_output if received_output else [
            neuron.value for neuron in self.layers[-1]]
        expected_output = expected_output if type(expected_output) is list else number_2_array(
            expected_output, len(self.layers[-1]))
        cost = 0
        for expected_value, received_value in zip(expected_output, received_output):
            cost += (expected_value - received_value)**2
        return cost

    def backpropagation(self, expected_output, received_output=None):
        received_output = received_output if received_output else [
            neuron.value for neuron in self.layers[-1]]
        expected_output = expected_output if type(expected_output) is list else number_2_array(
            expected_output, len(self.layers[-1]))
        params = []
        for neuron, output_value in zip(self.layers[-1], expected_output):
            neuron.back_propagation_value = -2*(output_value - neuron.value)
        for layer in self.layers[1:-1]:
            for neuron in layer:
                neuron.back_propagation_value = 0
        for layer in self.layers[:0:-1]:
            for neuron in layer:
                first = neuron.back_propagation_value
                second = neuron.value * (1 - neuron.value)
                bias = first * second
                params.append(bias)
                for conn in neuron.in_cons:
                    weight = conn.in_nuran.value * bias
                    params.append(weight)
                    if hasattr(conn.in_nuran, 'back_propagation_value'):
                        conn.in_nuran.back_propagation_value += conn.weight * bias
        return params

    def adjust_weights_and_biases(self, changes):
        for layer in self.layers[:0:-1]:
            for neuron in layer:
                neuron.bias -= changes.pop(0) * 2
                for con in neuron.in_cons:
                    con.weight -= changes.pop(0) * 2

    def learn(self, data, batch_size, time_to_stop_func=lambda: None):
        i = 0
        while batch_size*(i + 1) < len(data) and not time_to_stop_func():
            changes = []
            count = 0
            cost = 0
            succeeded = 0
            batch = data[batch_size*i:batch_size*(i + 1)]
            for inputs, output in batch:
                a = array_2_number(self.calc(inputs))
                if a == output:
                    succeeded += 1
                if count == 0:
                    changes = self.backpropagation(output)
                else:
                    temp_changes = self.backpropagation(output)
                    changes = [x + y for x, y in zip(
                        changes, temp_changes)]
                count += 1
                cost += self.get_cost(output)
            changes = [x / count for x in changes]
            cost /= count
            self.adjust_weights_and_biases(changes)
            succeeded = int(100 * (succeeded / count))
            print(f'cost: {cost}\n succeeded: {succeeded}')
            i += 1

    def get_state(self):
        params = []
        previous_layer = None
        for layer in self.layers[::-1]:
            for i, neuron in enumerate(layer):
                params.append(neuron.bias)
                if previous_layer:
                    for previous_layer_neuron in previous_layer:
                        params.append(previous_layer_neuron.in_cons[i].weight)
            previous_layer = layer
        return params[::-1]
