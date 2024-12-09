import numpy as np

class BasePerceptron:
    """
    Base abstract class that defines what every perceptron should be able to do.
    Think of it as a blueprint for all perceptron.
    """
    def __init__(self):
        self.weight = None
        self.bias = None

class ZeroPerceptron(BasePerceptron):
    """
    A perceptron that starts with zero weight and bias.
    """
    def __init__(self):
        self.weight = 0.0
        self.bias = 0.0

class RandomPerceptron(BasePerceptron):
    """
    A perceptron that starts with random weight and bias.
    This can help find solutions faster in some cases.
    """
    def __init__(self):
        self.weight = np.random.rand() # Random value between 0 and 1
        self.bias = np.random.rand() # Random value between 0 and 1
