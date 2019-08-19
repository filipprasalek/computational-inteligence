import random
import string

class Synapse:

    def __init__(self, left, right, weight=0.1):
        self.__name = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
        self.left = left
        self.right = right
        self.weight = weight

    def set_weight(self, weight):
        self.weight = weight

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def connect(self, left, right):
        left.connect(self)
        right.connect(self)

    def is_connected(self, one, another):
        return (self.left == one and self.right == another) or (self.left == another and self.right == another)
