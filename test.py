import numpy as np
from perceptron import ZeroPerceptron, RandomPerceptron

def generate_training_data(n_samples, min_val=-10, max_val=10, random_seed=42):
    """
    Génère un ensemble fixe de données d'entraînement.
    Le random_seed assure que nous obtenons toujours la même séquence.
    """
    np.random.seed(random_seed)  # Fixe la séquence aléatoire
    numbers = np.random.uniform(min_val, max_val, n_samples)
    labels = np.sign(numbers)
    return list(zip(numbers, labels))

def train_perceptrons(zero_perceptron, random_perceptron, training_data):
    """Entraîne les deux perceptrons sur les mêmes données."""
    print("\n=== Phase d'entraînement ===")
    for number, expected in training_data:
        print(f"\nEntraînement sur {number:.2f} (attendu: {expected})")
        print("Zero Perceptron:")
        zero_perceptron.train(number, expected)
        print("Random Perceptron:")
        random_perceptron.train(number, expected)

def test_perceptrons(zero_perceptron, random_perceptron, test_numbers):
    """Teste les deux perceptrons sur des nombres spécifiques."""
    print("\n=== Phase de test ===")
    for number in test_numbers:
        expected = np.sign(number)
        zero_pred = zero_perceptron.predict(number)
        random_pred = random_perceptron.predict(number)
        
        print(f"\nNombre testé : {number}")
        print(f"Zero Perceptron: prédit {zero_pred}, {'correct' if zero_pred == expected else 'incorrect'}")
        print(f"Random Perceptron: prédit {random_pred}, {'correct' if random_pred == expected else 'incorrect'}")

def save_perceptrons(zero_perceptron, random_perceptron):
    """Sauvegarde l'état final des deux perceptrons."""
    print("\n=== Sauvegarde des modèles ===")
    zero_perceptron.save_model()
    random_perceptron.save_model()

# Programme principal
if __name__ == "__main__":
    # Initialisation des perceptrons
    zero_perceptron = ZeroPerceptron()
    random_perceptron = RandomPerceptron()

    # Génération des données d'entraînement
    training_data = generate_training_data(100)
    test_numbers = [-5, -2, -0.5, 0.5, 2, 5]

    # Exécution des phases
    train_perceptrons(zero_perceptron, random_perceptron, training_data)
    test_perceptrons(zero_perceptron, random_perceptron, test_numbers)
    save_perceptrons(zero_perceptron, random_perceptron)