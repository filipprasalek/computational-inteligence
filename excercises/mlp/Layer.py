from excercises.mlp.Synapse import Synapse
from excercises.mlp.Neuron import Neuron, InputNeuron, OutputNeuron

class Layer:
    def __init__(self, number_of_neurons=0):
        self.neurons = [Neuron() for _ in range(number_of_neurons)]

    def connect_input(self, other_layer):
        for neuron in self.neurons:
            for other_neuron in other_layer.neurons:
                if neuron.is_connected(other_neuron):
                    break
                else:
                    connection = Synapse(other_neuron, neuron)
                    neuron.connect_input(connection)
                    other_neuron.connect_output(connection)

    def connect_output(self, other_layer):
        for neuron in self.neurons:
            for other_neuron in other_layer.neurons:
                if neuron.is_connected(other_neuron):
                    break
                else:
                    connection = Synapse(neuron, other_neuron)
                    neuron.connect_output(connection)
                    other_neuron.connect_input(connection)

    def __repr__(self):
        return str(self.neurons)


class InputLayer(Layer):

    def __init__(self, values):
        super().__init__()
        self.neurons = [InputNeuron(value) for value in values]


class OutputLayer(Layer):

    def __init__(self, classes):
        super().__init__()
        self.neurons = [OutputNeuron(class_name) for class_name in classes]

