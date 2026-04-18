import numpy as np
from sklearn.linear_model import SGDClassifier


class SVM:
    def __init__(self, C: float, epochs: int = 1000, lr: float = 1e-4):
        self.C = C
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        N = X.shape[0]

        # SGDClassifier(loss="hinge") = Soft-Margin SVM với SGD
        # alpha = 1/(C*N) là regularization, tương đương lambda trong Assignment 1
        self.model = SGDClassifier(
            loss = "hinge",  # hinge loss → SVM
            alpha = 1.0 / (self.C * N),  # regularization
            max_iter = self.epochs,  # số epochs
            eta0 = self.lr,  # learning rate
            learning_rate = "constant",  # giữ lr cố định như Assignment 1
            random_state = 42,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)  # shape (N,)