import numpy as np
import matplotlib.pyplot as plt



#Taylor polynomial of degree 2

def taylorPoly2nd(function, x0, h, range):
    
    '''
    function --> original function
    x0 --> point of approximation
    h --> step of of approximation
    range --> range of approximation to calculate
    '''

    # derivatives
    f_d1 = (function(x0 + h) - function(x0 - h)) / (2 * h) #1st derivative at x0
    f_d2 = (function(x0 + h) + function(x0 - h) - 2 * function(x0)) / (h**2) #2nd derivative at x0

    nums = range / h
    x = np.linspace(x0 - range/2, x0 + range/2, int(nums))

    # Taylor polynomial
    poly = lambda x: function(x0) + f_d1 * (x - x0) + 0.5 * f_d2 * (x - x0)**2 
    y = poly(x)

    return x, y


my_f = lambda x: np.cos(x) - 0.5 * np.sin(x) * np.cos(x**2)

results = taylorPoly2nd(my_f, 0.5*np.pi, 1e-3, 1.5*np.pi)
real_cos = my_f(results[0])

plt.plot(results[0], results[1], label = "Polynomial")
plt.plot(results[0], real_cos, label = "Real function")
plt.ylim(-3,3)
plt.legend()
plt.show()

#print(type(results[1]))
#print(results[1].shape)