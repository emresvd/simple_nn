from nn import SimpleNeuralNetwork, Sigmoid, Linear
import numpy as np
import sys


def xor():
    # Define a simple dataset
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Define the neural network architecture
    model = SimpleNeuralNetwork(2, 4, 1, activation_function=Sigmoid)
    print(f"Model has {model.total_parameters()} parameters.")

    # Train the neural network
    losses = model.train(x, y, epochs=10000, learning_rate=0.1)

    loss, accuracy = model.evaluate(x, y)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    # Make predictions
    print(model.feed_forward(np.array([[0, 1]])))


def linear():
    # Define a simple dataset
    np.random.seed(42)
    train_x, test_x = np.random.rand(100, 1), np.random.rand(100, 1)
    train_y, test_y = 2 * train_x - 1, 2 * test_x - 1

    # Define the neural network architecture
    model = SimpleNeuralNetwork(1, 1, activation_function=Linear)
    print(f"Model has {model.total_parameters()} parameters.")

    # Train the neural network
    losses = model.train(train_x, train_y, epochs=2000, learning_rate=0.01)

    # Evaluate the neural network on the test set
    test_loss, test_accuracy = model.evaluate(test_x, test_y)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Make predictions
    print(model.feed_forward(np.array([[12]])))


def classification():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Define a simple dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    y_one_hot = np.eye(2)[y]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Define the neural network architecture
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    hidden_size = 10

    # Create an instance of the SimpleNeuralNetwork
    model = SimpleNeuralNetwork(input_size, hidden_size, output_size, activation_function=Sigmoid)

    # Train the neural network
    epochs = 5000
    losses = model.train(X_train, y_train, epochs=epochs, learning_rate=0.01)

    # Evaluate the neural network on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    # Print the evaluation results
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Plot the training loss over epochs
    plt.plot(range(0, epochs, 1000), losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    getattr(sys.modules[__name__], sys.argv[1] if len(sys.argv) > 1 else "linear")()
