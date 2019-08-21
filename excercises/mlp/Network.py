from excercises.mlp.DataLoader import DataLoader
from excercises.mlp.Layer import InputLayer, OutputLayer, Layer

# TODO: Display general error during training to debug
class Network:
    def __init__(self, layers):
        self.layers = layers

    def connect_layers(self):
        for i in range(0, len(self.layers) - 1):
            self.layers[i].connect_output(self.layers[i + 1])

    def set_input(self, row):
        self.layers[0].set_values(list(row)[:-1])

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

    def train(self, learning_rate, epoch, prepared_data):
        iris_mapping = {
            'Iris-setosa': [1, 0, 0],
            'Iris-versicolor': [0, 1, 0],
            'Iris-virginica': [0, 0, 1]
        }
        for i in range(epoch):
            for row in prepared_data:
                self.set_input(row)
                self.layers[-1].set_values(iris_mapping[row[-1]])
                self.forward_propagate()
                self.backward_propagate()
                self.update_weights(learning_rate)

    def predict(self, row):
        self.set_input(row)
        self.forward_propagate()
        print("Expected %s" % row[-1])
        print("Actual %s" % [output_neuron.value for output_neuron in self.layers[-1].neurons])

    def print(self):
        for layer in self.layers:
            print(layer)


dataset = '../../datasets/iris.data'
data = DataLoader.load_csv_data_from_file(dataset)
normalized_data = DataLoader.normalize_data(data)
classes = DataLoader.get_class_names(normalized_data)
number_of_inputs = len(normalized_data[0]) - 1

layers = [
    InputLayer([1 for _ in range(number_of_inputs)]),
    Layer(5),
    OutputLayer(classes, [1 for _ in range(len(classes))])
]

network = Network(layers)
network.connect_layers()
network.train(0.5, 1, list(normalized_data)[:-1])
network.predict(normalized_data[5])
print("End")
