class Perceptron:
    """
    Un perceptron qui apprend à classifier des nombres comme positifs ou négatifs.
    """
    def __init__(self):
        """
        Constructeur : s'execute à la création d'un nouveau Perceptron
        C'est ici qu'on initialise les attributs de notre perceptron
        """
        # Les attributs sont définis avec self.
        self.poids = 0.0 # Le poids commence à zéro
        self.biais = 0.0 # Le biais commence à zéro