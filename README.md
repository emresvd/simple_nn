# Simple Neural Network Implementation in Python

This repository contains a simple implementation of a neural network in Python. The neural network is designed to be easily configurable with different activation functions and layer sizes. The implementation includes the core neural network class, various activation functions, and a sample usage in a main script.

## Contents

1. [nn.py](nn.py): The core implementation of the neural network, including forward and backward propagation, training, and parameter calculation.
2. [activation.py](activation.py): Different activation functions that can be used in the neural network, such as Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax, Linear, and ELU.
3. [main.py](main.py): An example of using the neural network to solve a simple XOR problem.

## How to Use

1. **Clone the repository:**

    ```bash
    git clone https://github.com/emresvd/simple_nn.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd simple_nn
    ```

3. **Run the main script:**

    ```bash
    python main.py
    ```

    This script demonstrates the creation of a neural network with specified layer sizes and activation function (Sigmoid), training it on XOR data, and making predictions.

## Neural Network Configuration

You can customize the neural network by modifying the `SimpleNeuralNetwork` class in `nn.py`. Adjust the layer sizes and activation function according to your requirements.

```python
# Example configuration in main.py
model = SimpleNeuralNetwork(2, 4, 1, activation_function=Sigmoid)
```

## Training

The neural network is trained using the train method, where you provide input data (x), target output (y), the number of training epochs, and the learning rate. The method now returns a list of losses for each epoch.

```python
losses = model.train(x, y, epochs=10000, learning_rate=0.1)
```

Training progress and loss are printed at regular intervals during training.

## Model Evaluation

After training, you can use the feed_forward method to make predictions with the trained model:

```python
prediction = model.feed_forward(np.array([[0, 1]]))
print(prediction)
````

Feel free to explore different activation functions and tweak hyperparameters to see how they affect the model's performance.

## Total parameters

You can check the total number of parameters in the neural network using the total_parameters method:

```python
print(f"Model has {model.total_parameters()} parameters.")
```

This provides insights into the complexity of the neural network.

## Dependencies
- [numpy](https://numpy.org/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
