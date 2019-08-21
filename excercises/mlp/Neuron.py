import random
import string
from math import exp


class Neuron:

    def __init__(self):
        self.__name = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
        self.input_connections = []
        self.output_connections = []

        self.value = 0
        self.delta = 0
        self.weighted_sum = 0

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

    # Calculation of weighted sum
    def activate(self):
        self.weighted_sum = sum([conn.weight * conn.get_other(self).value for conn in self.input_connections])

    def calculate_delta(self):
        errors = sum([conn.weight * conn.get_other(self).delta for conn in self.output_connections])
        self.delta = errors * self.transfer_derivative()

    # Transfer neuron activation
    def transfer(self):
        self.value = 1.0 / (1.0 + exp(-self.weighted_sum))

    def transfer_derivative(self):
        return self.value * (1.0 - self.value)

    def __repr__(self):
        return "%d " % self.value


class InputNeuron(Neuron):

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.input_connections = None

    def calculate_delta(self):
        return 0

class OutputNeuron(Neuron):

    def __init__(self, class_name, desired):
        super().__init__()
        self.output_connections = None
        self.output_class = class_name
        self.desired = desired

    def calculate_delta(self):
        self.delta = (self.desired - self.value) * self.transfer_derivative()

