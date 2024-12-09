import numpy as np

class BasePerceptron:
    """
    Base abstract class that defines what every perceptron should be able to do.
    Think of it as a blueprint for all perceptron.
    """
    def __init__(self):
        self.weight = None
        self.bias = None

    def predict(self, input_number):
        """
        Make a prediction whether a number is positive or negative

        Args:
            input_number : The number to classify

        Returns:
            1 if predicted positive, -1 if predicted negative
        """
        # The "weighted sum" is like our balance calculation
        weighted_sum = input_number * self.weight + self.bias

        # If the sum is positive, predict 1 (positive), else -1 (negative)
        return 1 if weighted_sum >= 0 else -1

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
