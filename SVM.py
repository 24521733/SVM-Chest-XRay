import numpy as np
from tqdm import tqdm


class SVM:
    def __init__(self, C:float, epochs:int = 1000, lr:float = 1e-4):
        self.C = C
        self.epochs = epochs
        self.lr = lr

        self.W = None
        self.b = None
        self.losses = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        _, dim = X.shape
        self.W = np.zeros((dim, 1))
        self.b = 0

        pbar = tqdm(range(self.epochs), desc="Training")
        for _ in pbar:
            idx = np.random.permutation(len(X))
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            for ith, x_i in enumerate(X_shuffled):
                x_i = np.expand_dims(x_i, axis=0)
                condition = y_shuffled[ith] * self.predict(x_i)

                if condition >= 1:
                    # Only regularization term
                    dw = self.W
                    db = 0
                else:
                    # Hinge loss active
                    dw = self.W - self.C * y_shuffled[ith] * x_i.T
                    db = -self.C * y_shuffled[ith]

                self.W = self.W - self.lr * dw
                self.b = self.b - self.lr * db

            y_hat = self.predict(X)
            loss = self.hinge_loss(y, y_hat)
            self.losses.append(loss)

            pbar.set_postfix({
                "loss": loss
            })

    def predict(self, X: np.ndarray):
        y_hat = X @ self.W + self.b

        return y_hat

    def hinge_loss(self, y: np.ndarray, y_hat: np.ndarray):
        delta = 1 - y * y_hat
        reg_term = 0.5 * (self.W.T @ self.W).item()
        hinge_term = self.C * np.sum(np.maximum(0, delta))
        return reg_term + hinge_term