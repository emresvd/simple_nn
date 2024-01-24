from nn import SimpleNeuralNetwork, Sigmoid, Linear
import numpy as np
import sys


def xor():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model = SimpleNeuralNetwork(2, 4, 1, activation_function=Sigmoid)
    print(f"Model has {model.total_parameters()} parameters.")

    losses = model.train(x, y, epochs=10000, learning_rate=0.1)

    print(model.feed_forward(np.array([[0, 1]])))


def linear():
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 2 * x - 1

    model = SimpleNeuralNetwork(1, 1, activation_function=Linear)
    print(f"Model has {model.total_parameters()} parameters.")

    model.train(x, y, epochs=2000, learning_rate=0.01)

    print(model.feed_forward(np.array([[12]])))


if __name__ == "__main__":
    getattr(sys.modules[__name__], sys.argv[1] if len(sys.argv) > 1 else "xor")()
