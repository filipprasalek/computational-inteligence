import random
import string


class Synapse:
    def __init__(self, left, right):
        self.__name = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
        self.left = left
        self.right = right
        self.weight = random.uniform(-0.5, 0.5)

    def set_weight(self, weight):
        self.weight = weight

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_other(self, one):
        if one == self.left:
            return self.right
        elif one == self.right:
            return self.left
        else:
            return None

    def connect(self, left, right):
        left.connect(self)
        right.connect(self)

    def is_connected(self, one, another):
        return (self.left == one and self.right == another) or (self.left == another and self.right == another)

