from excercises.mlp.core.DataLoader import DataLoader
from excercises.mlp.core.Layer import InputLayer, OutputLayer, Layer
from excercises.mlp.core.Network import Network

import random


dataset = '../../../datasets/iris.data'
data = DataLoader.load_csv_data_from_file(dataset)
normalized_data = DataLoader.normalize_data(data)
classes = DataLoader.get_class_names(normalized_data)
number_of_inputs = len(normalized_data[0]) - 1

training_set = list()
test_set = list()

for decision in classes:
    list_with_data_subset = [row for row in list(normalized_data) if row[-1] == decision]
    training_subset_len = int(len(list_with_data_subset) * 0.7)
    training_set += list_with_data_subset[:training_subset_len]
    test_set += list_with_data_subset[training_subset_len:]

random.shuffle(training_set)
random.shuffle(test_set)

iris_mapping = {
    'Iris-setosa': [1, 0, 0],
    'Iris-versicolor': [0, 1, 0],
    'Iris-virginica': [0, 0, 1]
}

layers = [
    InputLayer([1 for _ in range(number_of_inputs)]),
    Layer(5),
    Layer(3),
    OutputLayer(list(iris_mapping.keys()), [1 for _ in range(len(classes))])
]

network = Network(layers)
network.connect_layers()
network.train(0.1, 100, training_set, iris_mapping)
for row in test_set:
    network.predict(row)
    print("-----")

metrics = network.dump_metrics()
print(metrics['guesses'])
print(metrics['success'])
print(metrics['error'])
print("Accuracy %s" % (metrics['success'] / metrics['guesses']))
print(metrics['error_per_iteration'])

print("End")
