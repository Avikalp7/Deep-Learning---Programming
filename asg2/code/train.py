'''
Deep Learning Programming Assignment 1
--------------------------------------
Name: Avikalp Srivastava
Roll No.: 14CS10008

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
from __future__ import division
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import math

def sigmoid(z):
    ''' 
    Compute sigmoid for np.array z 
    '''
    nz = np.multiply(-1, z)
    return 1 / (1 + np.exp(nz))

def forward_pass(X, W1, W2):
    ''' 
    Compute forward pass through our neural network given input X and weights W1 and W2 
    '''
    X = np.matrix(X)
    m = X.shape[0]
    n = X.shape[1]
    # Adding bias column
    a1 = np.concatenate((np.ones((m,1)), X), axis = 1)
    z1 = a1 * (W1.T)
    m2 = z1.shape[0]
    a2 = np.concatenate((np.ones((m2, 1)), sigmoid(z1)), axis = 1)
    z2 = a2 * W2.T
    o = sigmoid(z2)
    return a1, z1, a2, z2, o


def sigmoidGradient(z):
    ''' 
    Gradient for sigmoid(z) 
    '''
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def cost(W1, W2, num_labels, X, y, a1, z1, a2, z2, o, regularisation_const = math.exp(-4)):  
    ''' 
    Compute gradients for W1 and W2 given X and y, and intermediate activations
    '''
    m = X.shape[0]
    n = X.shape[1]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # compute the cost --- NOT NEEDED
    J = 0
    for i in range(m):
        current_y = y[i].T
        #         current_y = np.zeros((num_labels,1))
        #         current_y[y[i]] = 1
        #         J = J - (1/m) * (log(a3(i,:)) * current_y + log(1-a3(i,:))*(1-current_y));
        term1 = np.log(o[i,:]) * current_y
        term2 = (np.log(1 - o[i,:])) * (1-current_y)
        # print term1[0][0], term2
        J-= np.sum(term2[0][0] + term1[0][0])
        #         first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        #         second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        #         J += np.sum(first_term - second_term)
    J = J / m
    print 'Cost: ',
    print J

    # Running Backpropagation
    big_delta1 = np.zeros(W1.shape)
    big_delta2 = np.zeros(W2.shape)
    for i in xrange(m):
        current_y = y[i].T
        delta3 = np.subtract(o[i].T, current_y)
        z1t = (np.concatenate((np.ones((z1[i].shape[0],1)), z1[i]), axis = 1)).T
        delta2 = np.multiply((W2.T * delta3), sigmoidGradient(z1t))
        big_delta1 = np.add(big_delta1, delta2[1:]*a1[i,:])
        big_delta2 = np.add(big_delta2, delta3*a2[i,:])
        

    # Final Gradients
    temp1 = (regularisation_const / m) * np.concatenate((np.zeros((W1.shape[0], 1)), W1[:, 1:]), axis = 1)
    W1_grad = np.add(np.divide(big_delta1, m), temp1)
    temp2 = (regularisation_const / m) * np.concatenate((np.zeros((W2.shape[0], 1)), W2[:, 1:]), axis = 1)
    W2_grad = np.add(np.divide(big_delta2, m), temp2)
    
    return J,W1_grad, W2_grad



def train(trainX, trainY):
    '''
    Complete this function.
    '''
    # We decide the hidden layer size here along with initializing some useful variables #
    num_samples = trainX.shape[0]
    n = trainX.shape[1] * trainX.shape[2]
    input_layer_size = n
    hidden_layer_size = 40
    num_labels = 10

    # Randomly initialising weight matrices uniformly between the values [-0.12, 0.12) #
    W1 = 0.24*np.random.random_sample((hidden_layer_size, input_layer_size+1)) - 0.12
    W2 = 0.24*np.random.random_sample((num_labels, hidden_layer_size + 1)) - 0.12

    # Unrolling the trainX array to the shape (num_samples, n) from (num_samples, n1, n2, 1)
    trainX_ = np.reshape(np.ravel(trainX), (num_samples, n))

    # Using one-hot encoder for trainY to get trainY_. Changes shape from (num_samples, 1) to (num_samples, num_labels)
    one_hot_encoder = OneHotEncoder(sparse=False)  
    trainY_onehot = one_hot_encoder.fit_transform(trainY.reshape(-1,1))

    # max_iter defines the maximum number of iterations for gradient descent #
    # Ideally, if the changes in weights are below a threshold, we should stop iterating, and hence the name max_iter #
    # However, that is not immplemented yet, hence max_iter = num_iter for now #
    max_iter = 9000
    learning_rate = math.exp(-4)
    batch_size = 200
    position = 0
    current_vt1 = 0
    current_vt2 = 0
    gamma = 0.9
    for i in xrange(max_iter):
        print 'ITERATION NUMBER: ',
        print i
        # start = position
        # end = 0
        # if position + batch_size >= num_samples:
        #     end = num_samples
        # else:
        #     end = position + batch_size
        indices = np.random.choice(num_samples, batch_size, replace = False)
        current_trainX = trainX_[indices]
        current_trainY = trainY_onehot[indices]
        # end = position + batch_size >= num_samples? num_samples-1 : position + batch_size
        # run the feed-forward pass
        # a1, z1, a2, z2, o = forward_pass(trainX_[start:end], W1, W2)
        a1, z1, a2, z2, o = forward_pass(current_trainX, W1, W2)
        # Get gradients for 2 weight matrices
        J, W1_grad, W2_grad = cost(W1, W2, num_labels, current_trainX, current_trainY, a1, z1, a2, z2, o)
        # if J < 1.5 and J >= 1:
            # learning_rate = 1
        # if J < 2 and J >= 1.5:
        #     # learning_rate = 0.3
        #     pass
        # elif J < 1.5 and J >= 0.8:
        #     learning_rate /= exp(1)
        # elif J > 0.5 and J < 0.8:
        #     learning_rate /= exp(1)
        # elif J <= 0.5:
        #     learning_rate = exp
        # Perform gradient update
        current_vt1 = gamma*current_vt1 + np.multiply(learning_rate, W1_grad)
        current_vt2 = gamma*current_vt2 + np.multiply(learning_rate, W2_grad) 
        W2 = W2 - current_vt2
        W1 = W1 - current_vt1
        # if position + batch_size >= num_samples:
        #     position = 0
        # else:
        #     position = position + batch_size
        # position = position + batch_size >= num_samples? 0 : position + batch_size

    # save = raw_input('Do you want to save the weights? [y/n]: ')
    # if save == 'y':
    # Saving Weights here
    np.save('W1.npy', W1)
    np.save('W2.npy', W2)



def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    # Initializing useful variables #
    m = testX.shape[0]
    n = testX.shape[1]
    num_labels = 10
    testX_ = np.reshape(np.ravel(testX), (m, n*n))
    
    # Load weights here #
    W1 = np.load('W1.npy')
    W2 = np.load('W2.npy')
    
    # Forward Pass to get output in o #
    a1, z1, a2, z2, o = forward_pass(testX_, W1, W2)

    # Concatenating bias column to testX to form testX_
    testX_ = np.concatenate((np.ones((m, 1)), testX_), axis = 1)
    testX_ = np.matrix(testX_)

    # Array to store predicted labels
    prediction = np.zeros(m)

    # Forward Pass is on
    h1 = sigmoid(testX_ * W1.T)
    h1_ = np.concatenate((np.ones((m,1)), h1), axis = 1)
    h2 = sigmoid(h1_ * W2.T)

    count = 0
    for sample in h2:
        # Label with highest output probability value is chosen as the label
        prediction[count] = np.argmax(sample)
        count += 1

    # Return predictions
    return prediction
