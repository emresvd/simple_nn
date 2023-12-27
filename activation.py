import numpy as np


class ActivationFunction:
    @staticmethod
    def function(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement the 'function' method.")

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement the 'derivative' method.")


class Sigmoid(ActivationFunction):
    @staticmethod
    def function(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return x * (1 - x)


class Tanh(ActivationFunction):
    @staticmethod
    def function(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return 1 - x ** 2


class ReLU(ActivationFunction):
    @staticmethod
    def function(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class LeakyReLU(ActivationFunction):
    @staticmethod
    def function(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.01 * x, x)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0.01)


class Softmax(ActivationFunction):
    @staticmethod
    def function(x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return x * (1 - x)


class Linear(ActivationFunction):
    @staticmethod
    def function(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return 1


class ELU(ActivationFunction):
    @staticmethod
    def function(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.01 * (np.exp(x) - 1))

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0.01 * np.exp(x))
