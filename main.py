from abc import ABC

class Neuron(ABC):
   
    def __init__(self):
        activation = None
        input_function = None
        output_function = None


class Connection(ABC):

    def __init__(self):
        weight = None
        bias = None

