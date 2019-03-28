import numpy as np
import time

# Declare variables
# https://adventuresinmachinelearning.com/neural-networks-tutorial/
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
w2 = np.zeros((1, 3))
w2[0,:] = np.array([0.5, 0.5, 0.5])
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])

def f(x):
    return 1 / (1 + np.exp(-x))

def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers-1): #For each layer
        #Setup the input array which the weights will be multiplied by for each layer
        #If it's the first layer, the input array will be the x input vector
        #If it's not the first layer, the input to the next layer will be the
        #output of the previous layer
        if l == 0:
            node_in = x
        else:
            node_in = h
        #Setup the output array for the nodes in layer l + 1
        h = np.zeros((w[l].shape[0],)) # Gives the number of rows
        #loop through the rows of the weight array
        for i in range(w[l].shape[0]): # For each row
            #setup the sum inside the activation function
            f_sum = 0
            #loop through the columns of the weight array
            for j in range(w[l].shape[1]): #for each column

                f_sum += w[l][i][j] * node_in[j]
            #add the bias
            f_sum += b[l][i]

            #finally use the activation function to calculate the
                #i-th output i.e. h1, h2, h3
            h[i] = f(f_sum)

    return h

def matrix_feed_forward_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l] # this is NOT THE SAME as *
        h = f(z)
    return h


if __name__ == "__main__":
    w  = [w1, w2]
    b = [b1, b2]
    x = [1.5, 2.0, 3.0]

    
    start = time.time()
    retVal = simple_looped_nn_calc(3, x, w, b)
    end = time.time()

    ns = end-start

    print("Nano seconds (simple): ")
    print(ns*1000000)

    start = time.time()
    retVal = matrix_feed_forward_calc(3, x, w, b)
    end = time.time()

    ns = end-start

    print("Nano seconds (matrix): ")
    print(ns*1000000)


    print(retVal)
