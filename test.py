import random
from neural_network import NeuralNetwork
from keras.datasets import mnist


(train_X, train_y), (test_X, test_y) = mnist.load_data()
images = train_X
results = train_y

data = []
i = 0
for image, result in zip(images, results):
    # if i > 300:
    #     break
    i += 1
    inputs = []
    for row in image:
        inputs.extend(row)
    inputs = [value/255 for value in inputs]
    data.append((inputs, result))


nn = NeuralNetwork(784, 10, 2, 16)


def time_to_stop():
    with open('stop', 'r') as f:
        if f.read() == 'stop':
            return True
        else:
            return False


while True:
    random.shuffle(data)
    nn.learn(data, 100)
