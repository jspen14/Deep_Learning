# This is a basic representation of the sigmoid function
    # As can be observed, it is not a step function meaning that it doesn't immediately transition from
    # 0 to 1. Rather, there is a gradual shift. To be able to use this to fire a neuron, we need to
    # manipulate it a little bit

import matplotlib.pylab as plt
import numpy as np
x = np.arange(-8, 8, 0.1)
f = 1 / (1 + np.exp(-x))
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
