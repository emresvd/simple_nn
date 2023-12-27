from nn import SimpleNeuralNetwork, Sigmoid
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = SimpleNeuralNetwork(2, 4, 1, activation_function=Sigmoid)
print(f"Model has {model.total_parameters()} parameters.")

losses = model.train(x, y, epochs=10000, learning_rate=0.1)

print(model.feed_forward(np.array([[0, 1]])))
