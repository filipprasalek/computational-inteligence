import random
import string


class Neuron:

    def __init__(self):
        self.__name = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
        self.input_connections = []
        self.output_connections = []

        self.value = 0
        self.delta = 0
        self.weighted_sum = 0

    def calculate_delta(self):
        pass

    def calculate_weighted_sum(self):
        pass

    def calculate_output_value(self):
        pass

    def connect_input(self, synapse):
        self.input_connections.append(synapse)

    def connect_output(self, synapse):
        self.output_connections.append(synapse)

    def is_connected(self, other_neuron):
        return (self.input_connections and any(
            connection for connection in self.input_connections if connection.is_connected(self, other_neuron))) or (
                       self.output_connections and any(
                   connection for connection in self.output_connections if
                   connection.is_connected(self, other_neuron)))

    def __repr__(self):
        return "%d " % self.value


class InputNeuron(Neuron):

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.input_connections = None


class OutputNeuron(Neuron):

    def __init__(self, class_name):
        super().__init__()
        self.output_connections = None
        self.output_class = class_name
