from numpy import recfromcsv
from excercises.agds.Iris import Iris
from copy import deepcopy


class AGDS:
    __args_mapper = {
        0: "sepal_len",
        1: "sepal_width",
        2: "petal_len",
        3: "petal_width",
        4: "class_name"
    }

    __iris_class_map = {
        'Iris-setosa': 1,
        'Iris-versicolor': 2,
        'Iris-virginica': 3
    }

    def __init__(self, path):
        data = recfromcsv(path, delimiter=',', encoding='UTF-8', names=None)
        self.irises = []
        self.params = {param: {} for param in self.__args_mapper.values()}
        for record in data:
            *attributes, class_name = record
            parsed_record = (*attributes, self.__iris_class_map[class_name])
            iris = Iris(*parsed_record)
            self.irises.append(iris)
            for i in range(len(parsed_record)):
                param_name = self.__args_mapper[i]
                self.__append_value_if_not_exists(param_name, parsed_record[i])
                self.params[param_name][parsed_record[i]].append(iris)

    def __append_value_if_not_exists(self, param, value):
        if value not in self.params[param]:
            self.params[param][value] = []

    def __calculate_node_value_weights(self, iris):
        weights = deepcopy(self.params)
        iris.weight = 1.0
        for param in self.params.items():
            param_name, param_values = param
            max_val = max(param_values.keys())
            min_val = min(param_values.keys())
            for value in param_values.keys():
                if value != iris.get_value(param_name):
                    weights[param_name][value] = 1 - abs(value - iris.get_value(param_name)) / (max_val - min_val)
                else:
                    weights[param_name][value] = 1
        return weights

    def __get_similarities(self, weights):
        result = list()
        for iris in self.irises:
            conn_weights = [iris.weight * weights[param_name][iris.get_value(param_name)] for param_name in
                            self.__args_mapper.values()]
            similarity = round(sum(conn_weights) * 100, 2)
            result.append((iris, similarity))
        return sorted(result, reverse=True, key=lambda record: record[1])

    @staticmethod
    def __display_weights(weights):
        print("Node value weights:")
        for param in weights.items():
            print(" parameter: %s" % param[0])
            for param_value, weight in sorted(param[1].items()):
                print("    %s - %s" % (param_value, weight))

    @staticmethod
    def __display_similarities(similarities):
        print("Similarities: ")
        for record in similarities:
            print(" %s - %f" % (record[0].get_lookup(), record[1]))

    def get_data(self):
        return self.irises

    def associate(self, iris):
        print("Looking for iris with %s" % iris.get_lookup())
        weights = self.__calculate_node_value_weights(iris)
        # self.__display_weights(weights)
        similarities = self.__get_similarities(weights)
        self.__display_similarities(similarities)
        return similarities


agds = AGDS('../../datasets/iris.data')
irises = agds.get_data()

# sample_iris = Iris(4.4, 3.5, 1.8, 0.4, 1)
sample_iris = irises[1]

similarities = agds.associate(sample_iris)

print('end')
