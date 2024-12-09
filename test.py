from perceptron import ZeroPerceptron, RandomPerceptron

zero_perceptron = ZeroPerceptron()
random_perceptron = RandomPerceptron()

training_data = [
    (5, 1),     # Big positive number
    (-3, -1),   # Average negative number
    (2, 1),     # Small positive number
    (-1, -1),   # Small negative number
    (7, 1),     # Big positive number
    (-6, -1)    # Big negative number
]

print("=== Entraînement du Zero Perceptron ===")
for number, expected in training_data:
    zero_perceptron.train(number, expected)

print("\n=== Entraînement du Random Perceptron ===")
for number, expected in training_data:
    random_perceptron.train(number, expected)

print("\n=== Analyse des performances ===")
print("Zero Perceptron:")
zero_perceptron.analyze_performance()
print("\nRandom Perceptron:")
random_perceptron.analyze_performance()

print("\n=== Sauvegarde des modèles ===")
zero_perceptron.save_model()
random_perceptron.save_model()