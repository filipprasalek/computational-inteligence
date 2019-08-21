from excercises.mlp.core.DataLoader import DataLoader
from excercises.mlp.core.Layer import InputLayer, OutputLayer, Layer
from excercises.mlp.core.Network import Network

import random


dataset = '../../../datasets/iris.data'
data = DataLoader.load_csv_data_from_file(dataset)
normalized_data = DataLoader.normalize_data(data)
classes = DataLoader.get_class_names(normalized_data)
number_of_inputs = len(normalized_data[0]) - 1

list_with_data = DataLoader.randomly_prepare_data(normalized_data)
training_set_len = int(len(list_with_data) * 0.75)

training_set = list(list_with_data)[:training_set_len]
test_set = list(list_with_data)[training_set_len:]

layers = [
    InputLayer([1 for _ in range(number_of_inputs)]),
    Layer(5),
    OutputLayer(classes, [1 for _ in range(len(classes))])
]

iris_mapping = {
    'Iris-setosa': [1, 0, 0],
    'Iris-versicolor': [0, 1, 0],
    'Iris-virginica': [0, 0, 1]
}

network = Network(layers)
network.connect_layers()
network.train(0.1, 500, list(training_set), iris_mapping)
for row in test_set:
    network.predict(row)
    print("-----")

print("End")
