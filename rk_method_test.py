import numpy as np
import matplotlib.pyplot as plt

from runge_kutta_function import solverRungeKutta

# ========================================
# 2 equations problem
# ========================================
# [A]{y'} + [B]{y} = {F}

# matrix A 2x2
A = np.array(([1,0],[0,1]))

# matrix B 2x2
B = np.array(([0,0],[-1,0]))

# Forces matrix
g = 9.81 # [m/s**2]
l = 1.2 # length [m]

# {y'} = invA*({F} - [B]{y}) --> 2 x 1
fPrime = lambda t, y: np.matmul(np.linalg.inv(A),(np.array([(-g/l)*np.sin(y[1]) , 0]) - np.matmul(B,y)))

# {y0} 2 x 1
y0 = np.array([0,np.pi/3])


# ========================================
# Initial parameters
# ========================================
h0 = 0.06 # initial time step size
hMin = 1e-8 # minimal allowed time step size
hMax = 0.2 # maximal allowed time step size

nMax = 5e4 # max limit of iterations

eMin = 1e-10
eMax = 1e-9

t0 = 0
tF = 5

y,t = solverRungeKutta(fPrime, y0, t0, tF, eMin, eMax, h0, hMin, hMax, nMax)


#print(y.shape)
plt.plot(t,y[0],label='angular speed')
plt.plot(t,y[1],label='angle')
plt.legend()
plt.show()
