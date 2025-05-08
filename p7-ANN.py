import numpy as np

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid activation function
def sigmoid_derivative(a):
    return a * (1 - a)

# Function to train the neural network for a given gate
def train_neural_network(inputs, expected_output, epochs=1001, learning_rate=0.1):
    np.random.seed(42)
    weights = np.random.randn(2, 1)  # Randomly initialize weights for 2 inputs
    bias = 0.0  # Initialize bias
    for epoch in range(epochs):
        # Feedforward step
        z = np.dot(inputs, weights) + bias
        predictions = sigmoid(z)
        
        # Backpropagation step
        error = expected_output - predictions
        d_predictions = error * sigmoid_derivative(predictions)
        gradient_w = np.dot(inputs.T, d_predictions)  # Gradient for weights
        gradient_b = np.sum(d_predictions)  # Gradient for bias
        
        # Update weights and bias
        weights += learning_rate * gradient_w
        bias += learning_rate * gradient_b
        
        # Print loss every 100 epochs
        if epoch % 100 == 0:
            loss = np.mean(error ** 2)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Weights = {weights.T}, Bias = {bias:.4f}")
    
    return weights, bias

# Testing function to predict the output
def test_neural_network(inputs, weights, bias):
    print("\nFinal Testing (After Training):")
    for x in inputs:
        z = np.dot(x, weights) + bias
        y_pred = sigmoid(z)
        print("Input :", x, " Predicted :", round(y_pred[0]))

# Define inputs and expected outputs for AND gate
and_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

and_expected_output = np.array([
    [0],
    [0],
    [0],
    [1]
])

# Train the neural network for AND gate
print("Training Neural Network for AND gate:")
and_weights, and_bias = train_neural_network(and_inputs, and_expected_output)

# Test the neural network for AND gate
test_neural_network(and_inputs, and_weights, and_bias)

# Define inputs and expected outputs for OR gate
or_expected_output = np.array([
    [0],
    [1],
    [1],
    [1]
])

# Train the neural network for OR gate
print("\nTraining Neural Network for OR gate:")
or_weights, or_bias = train_neural_network(and_inputs, or_expected_output)

# Test the neural network for OR gate
test_neural_network(and_inputs, or_weights, or_bias)
