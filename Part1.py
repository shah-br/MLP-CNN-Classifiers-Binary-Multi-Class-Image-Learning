import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Step 1: Load and preprocess data
def load_data():
    filepath=''  # Specify the path to your data files here
    # Load data
    data_class0 = loadmat(filepath + 'train_class0.mat')['x']
    data_class1 = loadmat(filepath + 'train_class1.mat')['x']
    test_class0 = loadmat(filepath + 'test_class0.mat')['x']
    test_class1 = loadmat(filepath + 'test_class1.mat')['x']
    
    return data_class0, data_class1, test_class0, test_class1

def process_data(train_data, val_data, test_data):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)

    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (test_data - mean) / std
    return train_data, val_data, test_data

# SELU activation functions
def selu(x):
    alpha = 1.6733
    scale = 1.0507
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu_derivative(x):
    alpha = 1.6733
    scale = 1.0507
    return scale * np.where(x > 0, 1, alpha * np.exp(x))

# Forward propagation
def forward_propagation(X, weights_input_hidden, weights_hidden_output):
    hidden_layer = selu(np.dot(X, weights_input_hidden))
    output_layer = np.dot(hidden_layer, weights_hidden_output)
    return hidden_layer, output_layer

# Backward propagation
def backward_propagation(X, y, output, hidden_layer, weights_input_hidden, weights_hidden_output, v_w_ih, v_w_ho, learning_rate, momentum, lambda_reg):
    
    err = y - output
    d_output = err
    d_hidden = d_output.dot(weights_hidden_output.T) * selu_derivative(hidden_layer)
    
    v_w_ih = momentum * v_w_ih + learning_rate * (X.T.dot(d_hidden) + lambda_reg * weights_input_hidden)
    v_w_ho = momentum * v_w_ho + learning_rate * (hidden_layer.T.dot(d_output) + lambda_reg * weights_hidden_output)
    
    weights_input_hidden = weights_input_hidden + v_w_ih
    weights_hidden_output = weights_hidden_output + v_w_ho
    
    return weights_input_hidden, weights_hidden_output, v_w_ih, v_w_ho

# Training function
def train_mlp(X, y, input_size, hidden_size, output_size, epochs=100, batch_size=32, learning_rate=0.0001, momentum=0.9, lambda_reg=0.0001, 
              validation_data=None, test_data=None, early_stopping=13):
    
    np.random.seed(5)
    weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
    weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
    training_loss, validation_loss, test_loss = [], [], []
    v_w_ih = np.zeros_like(weights_input_hidden)
    v_w_ho = np.zeros_like(weights_hidden_output)
    
    if validation_data:
        X_val, y_val = validation_data
        best_value = float('inf')
        patience_counter = 0
    
    if test_data:
        X_test, y_test = test_data

    for epoch in range(epochs):
        np.random.shuffle(X)
        for start in range(0, len(X), batch_size):
            batch_X = X[start:start + batch_size]
            batch_y = y[start:start + batch_size]
            
            hidden_layer, output = forward_propagation(batch_X, weights_input_hidden, weights_hidden_output)
            weights_input_hidden, weights_hidden_output, v_w_ih, v_w_ho = backward_propagation(
                batch_X, batch_y, output, hidden_layer, weights_input_hidden, weights_hidden_output, 
                v_w_ih, v_w_ho, learning_rate, momentum, lambda_reg)

        _, train_output = forward_propagation(X, weights_input_hidden, weights_hidden_output)
        train_loss = np.mean((y - train_output)**2) + (lambda_reg / 2) * (np.sum(weights_input_hidden**2) + np.sum(weights_hidden_output**2))
        training_loss.append(train_loss)

        if validation_data:
            _, val_output = forward_propagation(X_val, weights_input_hidden, weights_hidden_output)
            loss_value = np.mean((y_val - val_output)**2) + (lambda_reg / 2) * (np.sum(weights_input_hidden**2) + np.sum(weights_hidden_output**2))
            validation_loss.append(loss_value)

            if loss_value < best_value:
                best_value = loss_value
                patience_counter = 0
            else:
                patience_counter = patience_counter + 1
                if patience_counter >= early_stopping:
                    print(f"Early stopping at {epoch+1} epoch")
                    break
        if test_data:
            _, test_output = forward_propagation(X_test, weights_input_hidden, weights_hidden_output)
            test_loss_value = np.mean((y_test - test_output)**2) + (lambda_reg / 2) * (np.sum(weights_input_hidden**2) + np.sum(weights_hidden_output**2))
            test_loss.append(test_loss_value)

    return weights_input_hidden, weights_hidden_output, training_loss, validation_loss, test_loss


# Training, evaluation and prediction 
def train_and_evaluate(train_class0, train_class1, test_class0, test_class1, hidden_sizes, epochs=50, batch_size=32):
    input_size = train_class0.shape[1]
    output_size = 1 
    
    X_train = np.vstack((train_class0[:1500], train_class1[:1500]))
    y_train = np.vstack((np.zeros((1500, 1)), np.ones((1500, 1))))
    X_val = np.vstack((train_class0[1500:], train_class1[1500:]))
    y_val = np.vstack((np.zeros((500, 1)), np.ones((500, 1))))
    X_test = np.vstack((test_class0, test_class1))
    y_test = np.vstack((np.zeros((1000, 1)), np.ones((1000, 1))))
    
    X_train, X_val, X_test = process_data(X_train, X_val, X_test)

    results = {}

    for value in hidden_sizes:
        print(f"Training with value = {value}...")
        weights_input_hidden, weights_hidden_output, train_loss, loss_value, test_loss = train_mlp(
            X_train, y_train, input_size, value, output_size, 
            epochs, batch_size, validation_data=(X_val, y_val), test_data=(X_test, y_test)
        )
        
        # Plot
        plt.figure(figsize=(14, 8))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(loss_value, label='Validation Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.title(f'Learning Curves for value = {value}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()
        
        # Evaluate on test set
        _, output = forward_propagation(X_test, weights_input_hidden, weights_hidden_output)
        test_accuracy = np.mean((output > 0.5).astype(int) == y_test)
        print(f"Test Accuracy for value = {value}: {test_accuracy:.4f}")
        results[value] = test_accuracy

    bestvalue = max(results, key=results.get)
    print(f"Best hidden layer value = {bestvalue} with test accuracy of {results[bestvalue]:.4f}")

train_class0, train_class1, test_class0, test_class1 = load_data()
train_and_evaluate(train_class0, train_class1, test_class0, test_class1, hidden_sizes=[2, 4, 6, 8, 10])
