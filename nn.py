import numpy as np
from activation import *
from typing import List, Tuple, Type


class SimpleNeuralNetwork:
    """
    A simple neural network implementation.

    Args:
        layer_sizes (Tuple[int]): Sizes of the layers in the neural network.
        activation_function (Type[ActivationFunction]): Activation function to be used in the network.

    Attributes:
        layer_sizes (List[int]): Sizes of the layers in the neural network.
        activation_function (ActivationFunction): Activation function used in the network.
        weights (List[np.ndarray]): Weights of the neural network.
        biases (List[np.ndarray]): Biases of the neural network.
        activations (List[np.ndarray]): Activations of the neural network.

    Methods:
        feed_forward: Performs forward propagation in the neural network.
        back_propagation: Performs backpropagation in the neural network.
        train: Trains the neural network.
        total_parameters: Calculates the total number of parameters in the neural network.
    """

    def __init__(self, *layer_sizes: Tuple[int], activation_function: Type[ActivationFunction]) -> None:
        """
        Initializes a neural network with the given layer sizes and activation function.

        Args:
            *layer_sizes (Tuple[int]): Sizes of each layer in the neural network.
            activation_function (Type[ActivationFunction]): Activation function to be used in the network.

        Returns:
            None
        """
        self.layer_sizes: List[int] = layer_sizes
        self.activation_function: ActivationFunction = activation_function

        self.weights: List[np.ndarray] = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)]
        self.biases: List[np.ndarray] = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(self.layer_sizes) - 1)]
        self.activations: List[np.ndarray] = [np.zeros((1, size)) for size in layer_sizes]

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation in the neural network.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the neural network.
        """
        self.activations[0] = x

        for i in range(len(self.layer_sizes) - 1):
            weighted_sum = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.activations[i + 1] = self.activation_function.function(weighted_sum)

        return self.activations[-1]

    def back_propagation(self, y: np.ndarray, learning_rate: float) -> None:
        """
        Performs backpropagation in the neural network.

        Args:
            y (np.ndarray): Target output.
            learning_rate (float): Learning rate.

        Returns:
            None
        """
        errors = [y - self.activations[-1]]
        deltas = [errors[0] * self.activation_function.derivative(self.activations[-1])]

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            errors.insert(0, deltas[0].dot(self.weights[i].T))
            deltas.insert(0, errors[0] * self.activation_function.derivative(self.activations[i]))

        for i in range(len(self.layer_sizes) - 1):
            self.weights[i] += learning_rate * self.activations[i].T.dot(deltas[i])
            self.biases[i] += learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> List[float]:
        """
        Trains the neural network model using the given input data and labels.

        Args:
            x (np.ndarray): The input data.
            y (np.ndarray): The labels.
            epochs (int): The number of training epochs.
            learning_rate (float): The learning rate for gradient descent.

        Returns:
            List[float]: The list of losses for each epoch.
        """
        losses: List[float] = []

        for epoch in range(epochs):
            output = self.feed_forward(x)
            self.back_propagation(y, learning_rate)

            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                losses.append(loss)
                print(f"Epoch {epoch}, Loss: {loss}")

        return losses

    def total_parameters(self) -> int:
        """
        Calculates the total number of parameters in the neural network.

        Returns:
            int: Total number of parameters.
        """
        return sum([self.weights[i].size + self.biases[i].size for i in range(len(self.layer_sizes) - 1)])
