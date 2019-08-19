from excercises.mlp.DataLoader import DataLoader
from excercises.mlp.Layer import InputLayer, OutputLayer, Layer


class Network:
    def __init__(self, layers):
        self.layers = layers

    def connect_layers(self):
        for i in range(0, len(self.layers) - 1):
            self.layers[i].connect_output(self.layers[i + 1])

    def print(self):
        for layer in self.layers:
            print(layer)


dataset = '../../datasets/iris.data'
data = DataLoader.load_csv_data_from_file(dataset)
normalized_data = DataLoader.normalize_data(data)
classes = DataLoader.get_class_names(normalized_data)
number_of_inputs = len(normalized_data[0]) - 1

layers = [InputLayer([1 for _ in range(number_of_inputs)]), Layer(5), Layer(3), OutputLayer(classes)]
network = Network(layers)
network.connect_layers()
