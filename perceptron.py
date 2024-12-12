import numpy as np
import json
from datetime import datetime

class BasePerceptron:
    """
    Base abstract class that defines what every perceptron should be able to do.
    Think of it as a blueprint for all perceptron.
    """
    ZERO_MODEL_FILE = "zero_perceptron.json"
    RANDOM_MODEL_FILE = "random_perceptron.json"

    def __init__(self):
        self.weight = None
        self.bias = None
        self.history = []

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

    def log_state(self, input_number=None, prediction=None, correct_answer=None):
        """
        Log the current state and training details
        """
        log_entry = {
            'weight': float(self.weight),
            'bias': float(self.bias),
            'input': input_number,
            'prediction': prediction,
            'correct_answer': correct_answer,
            'was_correct': prediction == correct_answer if prediction is not None else None
        }
        self.history.append(log_entry)

        # Print current state
        print(f"\nTraining Step {len(self.history)}:")
        print(f"Weight: {self.weight:.4f}, Bias: {self.bias:.4f}")
        if input_number is not None:
            print(f"Input: {input_number}")
            print(f"Prediction: {prediction}, Correct Answer: {correct_answer}")
            print(f"Status: {'✓' if log_entry['was_correct'] else '✗'}")

    def train (self, input_number, correct_answer, learning_rate=0.1):
        """
        Train the perceptron with one example and log the process
        
        Args:
            input_number: Number to classify
            correct_answer: Expected answer (1 for positive, -1 for negative)
            learning_rate: How fast the perceptron learns (small steps = 0.1)
        """
        # Make a prediction
        prediction = self.predict(input_number)

        # Log the training step
        self.log_state(input_number, prediction, correct_answer)

        # If prediction is wrong, adjust weights and bias
        if prediction != correct_answer:
             # Calculate error and update parameters
             error = correct_answer - prediction
             self.weight = self.weight + learning_rate * error * input_number
             self.bias = self.bias + learning_rate * error

             # Log the changes
             print(f"Made a mistake! Adjusting parameters...")
        else:
            print("Correct prediction! No adjustments needed.")

    def analyze_performance(self):
        """
        Analyze the training history
        """
        total_steps = len(self.history)
        correct_predictions = sum(1 for step in self.history if step['was_correct'])

        print(f"\nPerformance Analysis:")
        print(f"Total training steps: {total_steps}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {(correct_predictions/total_steps)*100:.2f}%")

    def save_model(self, filename=None):
        """
        Save the perceptron's state to a JSON file.
        Uses the appropriate default filename if none is provided.
        """
        if filename is None:
            if isinstance(self, ZeroPerceptron):
                filename = self.ZERO_MODEL_FILE
            elif isinstance(self, RandomPerceptron):
                filename = self.RANDOM_MODEL_FILE
            else:
                filename = "perceptron_model.json"
                
        model_data = {
            'type': self.__class__.__name__,  # ZeroPerceptron or RandomPerceptron
            'parameters': {
                'weight': float(self.weight),
                'bias': float(self.bias)
            },
            'history': self.history,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=4)

        print(f"Model saved to {filename}")

    def load_model(self, filename='perceptron_model.json'):
        """
        Load a previously saved perceptron model
        """
        with open(filename, 'r') as f:
            model_data = json.load(f)

        # Restore parameters
        self.weight = model_data['parameters']['weight']
        self.bias = model_data['parameters']['bias']
        self.history = model_data['history']

        print(f"Model loaded from {filename}")
        print(f"Model type: {model_data['type']}")
        print(f"Last saved: {model_data['saved_at']}")

class ZeroPerceptron(BasePerceptron):
    """
    A perceptron that starts with zero weight and bias.
    """
    def __init__(self):
        super().__init__()
        self.weight = 0.0
        self.bias = 0.0

class RandomPerceptron(BasePerceptron):
    """
    A perceptron that starts with random weight and bias.
    This can help find solutions faster in some cases.
    """
    def __init__(self):
        super().__init__()
        self.weight = np.random.rand() # Random value between 0 and 1
        self.bias = np.random.rand() # Random value between 0 and 1
