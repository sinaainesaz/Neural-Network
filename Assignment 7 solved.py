#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np


# In[16]:


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


# In[17]:


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


# In[18]:


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test


# In[19]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# In[20]:


def deriv_sig(x):
    return sigmoid(x) * (1 - sigmoid(x))


# In[21]:


def mean_Squared_loss(actual, predicted):
    return 0.5 * np.sum((actual - predicted)**2)


# In[22]:


def mean_squared_loss_deri(actual, predicted):
    return predicted - actual


# In[32]:


def training_testing(X_train, y_train, n_epochs, learning_rate, X_test, y_test):
    # defining the number of input, hidden and output units
    num_in_units = 2
    num_hid_units = 2
    num_out_units = 1
    # generating the random weights for the transfer among the layers
    w1 = np.random.normal(0, 0.1, (num_hid_units ,num_in_units))
    w2 = np.random.normal(0, 0.1, (num_out_units ,num_hid_units))

    for epoch in range(n_epochs):
        for x, y in zip(X_train, y_train):
            
            #Forward
            hidden = sigmoid(np.dot(w1, x))
            output = np.dot(w2, hidden)
            
            #computing error for the output layer
            output_error = mean_squared_loss_deri(y, output)
            #Backward
            hidden_error = np.dot(output_error, w2) * deriv_sig(np.dot(w1, x))
                             
            # Update weights
            w2 -= learning_rate * np.outer(output_error, hidden)
            w1 -= learning_rate * np.outer(hidden_error, x)

        
        # Compute the total MSE loss over all training examples
        total_loss = 0
        for x, y in zip(X_train, y_train):
            hidden = sigmoid(np.dot(w1, x))
            output = np.dot(w2, hidden)
            total_loss += mean_Squared_loss(y, output)
        total_loss /= len(X_train)

        # Print the total loss every 50 epochs
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss={total_loss}")
    print("Training is Done.")     
    print("----------------------------")
    print("Testing starts.")
    # Computing the total MSE on test set examples
    test_error = 0
    for x_test, y_test in zip(X_test, y_test):
        hidden_test = sigmoid(np.dot(w1, x_test))
        output_test = np.dot(w2, hidden_test)
        test_error += mean_Squared_loss(y_test, output_test)
    test_error /= len(X_test)
    print("Test error is:", test_error)


# In[33]:


if __name__ == "__main__":
    
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    print("===========================================================")
    print("Using 0.1 as learning rate.")
    training_testing(X_train, y_train, 1000, 0.1, X_test, y_test)
    print("===========================================================")
    print("Using 0.01 as learning rate.")
    training_testing(X_train, y_train, 1000, 0.01, X_test, y_test)
    print("===========================================================")
    print("Using 0.01 as learning rate but higher number for epoch.")
    training_testing(X_train, y_train, 3000, 0.01, X_test, y_test)


# In[ ]:




