from abc import ABC
import random
from math import tanh
from tqdm import tqdm


def relu(x):
    """Simple ReLU function"""
    return max(0.0, x)


def relu_prime(x):
    """Derivative of ReLU function"""
    return 1.0 if x > 0.0 else 0.0


def tanh_prime(x):
    """Derivative of TanH func"""
    return 1 - (tanh(x))**2


class Network(ABC):

    def __init__(self, layers=None):
        self.layers = []
        if layers is not None:
            [self.add_layer(i) for i in layers]

    def describe(self):
        for i, layer in enumerate(self.layers):
            print(f'L{i}:\t{layer.describe()}')

    def add_layer(self, size):
        """Add layer of specified size to right of network"""
        if len(self.layers) > 0:
            inshape = self.layers[-1].size
        else:
            inshape = 1
        self.layers.append(Layer([Neuron(inshape) for i in range(size)]))

    def feed_forward(self, inputs):
        """Forward propagate inputs through each layer in net"""
        # Process input layer
        outs = [n.activate([inputs[i]])
                for i, n in enumerate(self.layers[0].neurons)]
        # Process hidden layers
        for layer in self.layers[1:]:
            outs = layer.forward(outs)
        return outs

    def backprop(self, targets):
        """Backpropagate error and update weights/bias"""
        sum_error = 0.0
        for i in range(1, len(self.layers) + 1):
            # Move from last layer backwards
            layer = self.layers[-i]
            for j, neuron in enumerate(layer.neurons):
                # Calculate error and delta for each neuron in the layer
                if i == 1:
                    # Output layer has simple error func
                    sum_error += neuron.calc_output_delta(targets[j])
                else:
                    # Hidden layers have complex error func
                    sum_error += neuron.calc_hidden_delta(
                        self.layers[-i+1].neurons, j)
        return sum_error

    def update_weights(self, inputs, learn_rate):
        """Update weights in network for given inputs"""
        for i, layer in enumerate(self.layers):
            if i != 0:
                # If hidden layer, use outputs of previous layer
                inputs = [
                    neuron.last_activation for neuron in self.layers[i-1].neurons]
            for neuron in layer.neurons:
                neuron.update(inputs, learn_rate)

    def train(self, X, y, n_classes, learn_rate=0.01, n_epochs=3):
        """Train the network on X matrix and y vector"""
        for epoch in range(n_epochs):
            sum_error = 0.0
            if n_classes != self.layers[-1].size:
                raise ValueError('Must have output neuron for each class')
            if len(X) != len(y):
                raise ValueError(
                    'Training data and target values must be same shape')
            width = len(X[0])
            if self.layers[0].size != width:
                raise ValueError(
                    'Input layer and training data must have same shape')
            if not all([len(x) == width for x in X]):
                raise ValueError('Training data must have consistent shape')

            for i, row in tqdm(enumerate(X), total=len(X)):
                # One-hot encode output vector
                expected = [0 if m != y[i] else 1 for m in range(n_classes)]
                self.feed_forward(row)
                error = self.backprop(expected)
                sum_error += error
                self.update_weights(row, learn_rate)
            print(
                f'\rEpoch: {epoch}\tError: {sum_error}\tLearn Rate:{learn_rate}')

    def predict(self, x):
        """Given a single input, predict output"""
        self.feed_forward(x)
        outputs = [n.last_activation for n in self.layers[-1].neurons]
        return outputs

    def test(self, X, y):
        """Test the network on X matrix and y vector"""
        if len(X) != len(y):
            raise ValueError(
                'Test set data and target values must be same length')
        hits, misses = [0, 0]
        for i, x in tqdm(enumerate(X), total=len(X)):
            pred_vec = self.predict(x)
            pred_val = -100
            pred_loc = 0
            for j in range(len(pred_vec)):
                if pred_vec[j] > pred_val:
                    pred_loc = j
                    pred_val = pred_vec[j]
            if (pred_loc + 1) == y[i]:
                hits += 1
            else:
                misses += 1
        print('\rResults--------')
        print(f'Hits:\t{hits}({hits*100/len(y)}%)')
        print(f'Misses:\t{misses}({misses*100/len(y)}%)')


class Layer(ABC):

    def __init__(self, neurons=None):
        if neurons is not None:
            self.neurons = neurons
        else:
            self.neurons = []

    @property
    def size(self):
        return len(self.neurons)

    def describe(self):
        desc = ''
        for i, neuron in enumerate(self.neurons):
            desc += f'N{i}({neuron.describe()}) '
        return desc

    def forward(self, inputs):
        """Feed inputs through the layer"""
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.activate(inputs))
        return outputs


class Neuron(ABC):

    def __init__(self, size, act_func=tanh):
        # Size is the number of neurons in previous layer / inputs
        self.__weights = [random.random() for i in range(size)]
        self.__bias = 0.0
        self.__act_func = act_func
        self.__last_activation = None
        self.__delta = None

    @property
    def size(self):
        return len(self.__weights)

    @property
    def weights(self):
        return self.__weights

    @property
    def bias(self):
        return self.__bias

    @property
    def last_activation(self):
        return self.__last_activation

    @property
    def delta(self):
        return self.__delta

    def describe(self):
        return f"{', '.join([str(x) for x in self.__weights])} + {self.__bias}"

    def activate(self, inputs):
        """Activate neuron with given inputs"""
        assert(self.size == len(inputs))
        activation = self.__bias
        activation += sum([self.__weights[i] * inputs[i]
                           for i, _ in enumerate(inputs)])
        out = self.__act_func(activation)
        self.__last_activation = out
        return out

    def calc_output_delta(self, target):
        """Calculate error and delta for output layer neuron"""
        assert(self.__last_activation is not None)
        error = (target - self.__last_activation)
        self.__delta = error * tanh_prime(self.__last_activation)
        return error

    def calc_hidden_delta(self, downstream_neurons, mypos):
        """Calculate error and delta for hidden layer neuron"""
        error = 0.0
        for node in downstream_neurons:
            error += (node.weights[mypos] * node.delta)
        self.__delta = error * tanh_prime(self.__last_activation)
        return error

    def update(self, inputs, learn_rate):
        """Update weights and bias"""
        if self.size == 1:
            # Treat as input layer
            for inp in inputs:
                self.__weights[0] += learn_rate * self.delta * inp
            self.__bias += learn_rate * self.delta
        else:
            # Treat as hidden layer
            for j, inp in enumerate(inputs):
                self.__weights[j] += learn_rate * self.delta * inp
            self.__bias += learn_rate * self.delta
