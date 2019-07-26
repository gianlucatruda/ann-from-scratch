from abc import ABC

def relu(x):
    # Simple ReLU function
    return max(0, x)

def mean_squared_error(truth, pred):
    # MSE function
    assert(len(truth) == len(pred))
    n = len(truth)
    return 1/n * sum([(y - x)**2 for y,x in zip(truth, pred)])

class Neuron(ABC):
   
    def __init__(self):
        activation = None
        input_function = None
        output_function = None


class Connection(ABC):

    def __init__(self):
        weight = None
        bias = None

