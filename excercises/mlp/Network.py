from excercises.mlp.DataLoader import DataLoader


class Network:
    def __init__(self, layers):
        self.layers = layers

    def append_layer(self, layer):
        self.layers.append(layer)

    def connect_layers(self, ):
        pass


data = DataLoader.load_csv_data_from_file('../../datasets/iris.data')
normalized_data = DataLoader.normalize_data(data)
print(normalized_data)
