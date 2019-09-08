from numpy import recfromcsv
from random import shuffle

class DataLoader:

    @staticmethod
    def load_csv_data_from_file(filename):
        return recfromcsv(filename, delimiter=',', encoding='UTF-8', names=None)

    @staticmethod
    def normalize_data(input):
        border_values = {}
        number_of_attributes = len(input[0]) - 1
        for attribute_number in range(number_of_attributes):
            possible_attributes = set([float(record[attribute_number]) for record in input])
            border_values.update({attribute_number: (min(possible_attributes), max(possible_attributes))})

        for record in input:
            for attribute_number in range(number_of_attributes):
                record[attribute_number] = (record[attribute_number] - border_values[attribute_number][0]) / (
                            border_values[attribute_number][1] - border_values[attribute_number][0])
        return input

    @staticmethod
    def randomly_prepare_data(normalized_input):
        normalized_input_list = list(normalized_input)
        # shuffle(normalized_input_list)
        return normalized_input_list

    @staticmethod
    def get_class_names(dataset):
        return list(set([record[-1] for record in dataset]))