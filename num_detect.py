from sklearn.datasets import load_digits    # loads in the data set
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


digits = load_digits()                      # loads the digit data, quite well I might add 
# plt.gray()
# plt.matshow(digits.images[1])
# plt.show()

X_scale = StandardScaler()

# By default normalizes by subtracting the mean and dividing by the standard deviation
X = X_scale.fit_transform(digits.data)

from sklearn.model_selection import train_test_split
y = digits.target

# It is standard practice to save between 20 to 40% of the data to use for testing
    # Doesn't just take the first 60% for training. This helps account for collection details
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Because the ouput will be in the form of a vector, we need to convert the data that we use into vector form

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect

y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)

# Because there are 64 nodes in each of the images, we need to have 64 input neurons
# Because the output of the nn is going to be 1 of 10 things, there needs to be 10 final neurons
# For the hidden layer (which is needed for the complexity of the task), pick somewhere in the middle

nn_structure = [64,30,10]

# Set up the activation functions
def f(x):
    return 1 / (1 + np.exp(-1))

def f_deriv(f):
    return f(x)/(1-(f(x)))

print(y_v_train[0])

# Initialize the W and b dictionaries with random values

import numpy.random as rand
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}

    for l in range (1,len(nn_structure)):
        W[l] = rand.random_sample((nn_structure[l],nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))

    return W, b

def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z

# print(X[0,:])
