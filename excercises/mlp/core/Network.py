from excercises.mlp.core.DataLoader import DataLoader
from excercises.mlp.core.Layer import InputLayer, OutputLayer, Layer


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.metrics = {
            'error_per_iteration': [],
            'guesses': 0,
            'success': 0,
            'error': 0
        }

    def dump_metrics(self):
        return self.metrics

    def reset_metrics(self):
        self.metrics = {
            'error_per_iteration': [],
            'guesses': 0,
            'success': 0,
            'error': 0
        }

    def connect_layers(self):
        for i in range(0, len(self.layers) - 1):
            self.layers[i].connect_output(self.layers[i + 1])

    def set_input(self, row):
        self.layers[0].set_values(list(row)[:-1])

    def set_output(self, class_name, mapping):
        self.layers[-1].set_values(mapping[class_name])

    def forward_propagate(self):
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron.activate()
                neuron.transfer()

    def backward_propagate(self):
        for layer in reversed(self.layers):
            for neuron in layer.neurons:
                neuron.calculate_delta()

    def update_weights(self, learning_rate):
        for layer in self.layers[1:]:
            layer.update_weights(learning_rate)

    def train(self, learning_rate, epoch, prepared_data, mapping):
        for i in range(epoch):
            for row in prepared_data:
                class_name = row[-1]

                self.set_input(row)
                self.set_output(class_name, mapping)

                self.forward_propagate()
                self.backward_propagate()
                self.update_weights(learning_rate)
            error_sum = sum([(neuron.desired-neuron.value)**2 for neuron in self.layers[-1].neurons])
            self.metrics['error_per_iteration'].append({i: error_sum})

    def predict(self, row):
        self.metrics['guesses'] += 1
        self.set_input(row)
        self.forward_propagate()
        prediction = max(self.layers[-1].neurons, key=lambda neuron: neuron.value)
        print("Expected %s" % row[-1])
        print("Actual %s" % prediction.output_class)
        if row[-1] == prediction.output_class:
            self.metrics['success'] += 1
        else:
            self.metrics['error'] += 1

    def print(self):
        for layer in self.layers:
            print(layer)

