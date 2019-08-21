from excercises.mlp.core.Synapse import Synapse
from excercises.mlp.core.Neuron import Neuron, InputNeuron, OutputNeuron

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

    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            for conn in neuron.input_connections:
                new_weight = conn.weight + learning_rate * neuron.delta * conn.get_other(neuron).value
                conn.set_weight(new_weight)

    def __repr__(self):
        return str(self.neurons)


class InputLayer(Layer):
    def __init__(self, row):
        super().__init__()
        self.neurons = [InputNeuron(value) for value in row]

    def set_values(self, row):
        for i in range(len(row)):
            self.neurons[i].value = row[i]

class OutputLayer(Layer):

    def __init__(self, classes, expected):
        super().__init__()
        for i in range(len(expected)):
            self.neurons.append(OutputNeuron(classes[i], expected[i]))

    def set_values(self, classes):
        for i in range(len(classes)):
            self.neurons[i].desired = classes[i]