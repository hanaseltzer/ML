# Deep Learning
<!-- neural network -->
![alt text](https://cdn.pixabay.com/photo/2020/06/02/09/39/banner-5250178__340.jpg)
## NeuralNetwork()
>### object initialization

### params:

#### *input_neuron_count -*
> The amount of input neurons.

#### *output_neuron_count -*
> The amount of output neurons.

#### *hidden_layers_count -*
> How many hidden layers we want.

#### *hidden_layer_neuron_count -*
> How many neurons we want in the hidden layers.

#### *weights_and_biases (optional) -*
> List of predefined weights and biases we want the neural network to have.


## learn()
>### running over large list of datasets and learn from them.

### method params:

#### *data -*
> A list of datasets, a dataset is an array with two values. <br />
> the first value is an array containing the input values <br />
> the second value is an integer or array represnting the wanted output. 

#### *batch_size -*
> The amount of datasets per batch.

#### *time_to_stop_func (optional) -*
> A function that returns boolean false while running <br />
> and true when we want to stop learning. <br />
> if not specified will stop after running over all the data


## calc()
>### calculate ai result for inputs array sample.

### method params:

#### *inputs -*
> An array containing the input values

### method output:
> result array, representing the values of the output layer.
