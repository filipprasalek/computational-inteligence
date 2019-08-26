import random
import string


class Iris:

    def __init__(self, sepal_len, sepal_width, petal_len, petal_width, class_name):
        self.lookup = {
            'sepal_len': sepal_len,
            'sepal_width': sepal_width,
            'petal_len': petal_len,
            'petal_width': petal_width,
            'class_name': class_name
        }
        self.weight = 0.2
        self.__name = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))

    def get_value(self, param_name):
        return self.lookup[param_name]

    def get_lookup(self):
        return self.lookup
