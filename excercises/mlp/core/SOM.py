import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class SOM:

    def __init__(self, no_rows, no_cols, dimension):
        np.random.seed(1)
        self.som = np.random.random_sample(size=(no_rows, no_cols, dimension))
        self.som_class = np.empty(shape=(no_rows, no_cols), dtype=object)
        self.rows = no_rows
        self.cols = no_cols

    def __calculate_euclidan_distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)

    def __calculate_manhattan_distance(self, r1, c1, r2, c2):
        return np.abs(r1 - r2) + np.abs(c1 - c2)

    def find_best_matching_unit(self, vectors, to_be_matched_index):
        result = (0, 0)
        smallest_distance = float('inf')
        for row_index in range(self.rows):
            for col_index in range(self.cols):
                distance = self.__calculate_euclidan_distance(self.som[row_index][col_index], vectors[to_be_matched_index])
                if distance < smallest_distance:
                    smallest_distance = distance
                    result = (row_index, col_index)
        return result

    def self_organize(self, steps, attributes, classes):
        max_learn = 0.5
        max_range = self.rows + self.cols
        for step in range(steps):
            left_iterations_percentage = 1.0 - ((step * 1.0) / steps)
            curr_range = int(left_iterations_percentage * max_range)
            curr_rate = left_iterations_percentage * max_learn

            random_vector_index = np.random.randint(len(attributes))
            (bmu_row, bmu_col) = self.find_best_matching_unit(attributes, random_vector_index)

            for row in range(self.rows):
                for col in range(self.cols):
                    if self.__calculate_manhattan_distance(bmu_row, bmu_col, row, col) < curr_range:
                        self.som[row][col] += curr_rate * (attributes[random_vector_index] - self.som[row][col])
                        self.som_class[row][col] = classes[random_vector_index]

    def draw_result(self, attributes, classes):
        mapping = np.empty(shape=(self.rows, self.cols), dtype=object)
        for row in range(self.rows):
            for col in range(self.cols):
                mapping[row][col] = []

        for record_index in range(len(attributes)):
            (m_row, m_col) = self.find_best_matching_unit(attributes, record_index)
            mapping[m_row][m_col].append(classes[record_index])

        label_map = np.zeros(shape=(self.rows, self.cols), dtype=np.int)
        for row in range(self.rows):
            for col in range(self.cols):
                label_map[row][col] = stats.mode(mapping[row][col])[0] if mapping[row][col] != [] else -1

        plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4))
        plt.colorbar()
        plt.show()

    def get_result(self):
        return self.som

    def get_result_as_list(self):
        result = []
        for row in range(self.rows):
            for col in range(self.cols):
                result.append(list(self.som[row][col]) + [str(self.som_class[row][col])])
        return np.asarray(result)



