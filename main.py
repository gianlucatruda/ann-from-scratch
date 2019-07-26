from abc import ABC

def relu(x):
    # Simple ReLU function
    return max(0, x)

def mean_squared_error(truth, pred):
    # MSE function
    assert(len(truth) == len(pred))
    n = len(truth)
    return 1/n * sum([(y - x)**2 for y,x in zip(truth, pred)])

class Network(ABC):

    def __init__(self, layers=None):
        if layers is not None:
            self.layers = layers
        else:
            self.layers = []

    def describe(self):
        for i, layer in enumerate(self.layers):
            print(f'L{i}:\t{layer.describe()}')

    def feed_forward(self, inputs):
        # Forward propagate inputs through each layer in net
        outs = inputs
        for layer in self.layers:
            outs = layer.forward(outs)
            print(len(outs))
        return outs


class Layer(ABC):

    def __init__(self, neurons=None):
        if neurons is not None:
            self.neurons = neurons
        else:
            self.neurons = []
    
    def describe(self):
        desc = ''
        for i, neuron in enumerate(self.neurons):
            desc += f'N{i}({neuron.size}) '
        return desc

    def forward(self, inputs):
        # Feed forward inputs through the layer
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.activate(inputs))
        return outputs


class Neuron(ABC):
   
    def __init__(self, size, act_func=relu):
        # Size is the number of neurons in previous layer / inputs
        self.__weights = [1.0 for i in range(size)]
        self.__bias = 0
        self.__act_func = act_func

    @property
    def size(self):
        return len(self.__weights)

    def activate(self, inputs):
        # Activate neuron with given inputs
        assert(self.size) == len(inputs))
        activation = self.__bias
        activation += sum([self.__weights[i] * inputs[i] 
            for i, _ in enumerate(inputs)]) 
        return self.__act_func(activation)

    
