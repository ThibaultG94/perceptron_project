import numpy as np
from perceptron import ZeroPerceptron, RandomPerceptron

def generate_training_data(n_samples, min_val=-10, max_val=10):
    # Génère les nombres aléatoires
    numbers = np.random.uniform(min_val, max_val, n_samples)
    # Génère les labels (-1 ou 1)
    labels = np.sign(numbers)
    # Combine en liste de tuples
    return list(zip(numbers, labels))

# Création des perceptrons
zero_perceptron = ZeroPerceptron()
random_perceptron = RandomPerceptron()

# Génération des données d'entraînement
training_data = generate_training_data(100)

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