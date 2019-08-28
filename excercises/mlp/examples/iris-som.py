from excercises.mlp.core.DataLoader import DataLoader
from excercises.mlp.core.SOM import SOM
import numpy as np

iris_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

rows = 5
cols = 5

raw_data = DataLoader.load_csv_data_from_file('../../../datasets/iris.data')
normalized_data = DataLoader.normalize_data(raw_data)

attributes = np.asarray([list(record)[:-1] for record in normalized_data])
classes = np.asarray([iris_mapping[list(record)[-1]] for record in normalized_data])

som = SOM(rows, cols, len(attributes[0]))
som.self_organize(5000, attributes, classes)
result = som.get_result()
som.draw_result(attributes, classes)

print("END")