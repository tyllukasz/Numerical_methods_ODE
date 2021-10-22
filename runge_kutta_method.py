# === ORDINARY DIFFERENTIALL EQUATIONS ===
# ========================================
# ===== Runge-Kutta Adaptive Method ======
# ========================================

import numpy as np
import matplotlib.pyplot as plt

# ========================================
# function to find solution
# y' = - y + exp(-t)
# y(t0) = 0
# exact solution = t * exp(-t)
# ========================================

# equation
fPrime = lambda t, y: -y + np.exp(-t)

# exact solution
exact_sol = lambda t: t * np.exp(-t)

# ========================================
# INPUT DATA
# ========================================
y0 = 0 # y(t0)

t0 = 0 # start time
tf = 5 # stop time


# ========================================
# SOLVER PARAMETERS
# ========================================

h0 = 0.06 # initial time step size
hMin = 0.001 # minimal allowed time step size
hMax = 0.2 # maximal allowed time step size

eMin = 8e-10 # minimal allowed estimated error
eMax = 1e-9 # maximal allowed estimated error

nMax = 100000 # max limit of iterations


# ==============================================================================================================
# Runge-Kutta equations (for R4 and R5 approximations)
# ==============================================================================================================

# weight coefficients
w4 = np.zeros(5)
w4[0] = 25/216
w4[1] = 0
w4[2] = 1408/2565
w4[3] = 2197/4104
w4[4] = -1/5

w5 = np.zeros(6)
w5[0] = 16/135
w5[1] = 0
w5[2] = 6656/12825
w5[3] = 28561/56430
w5[4] = -9/50
w5[5] = 2/55


# ============================
# SOLVER LOOP
# ============================

# initial values of variables
h = h0 # time step
t = t0 # current time
y = y0 # y(t0) value
i = 0 # iterator initial value
error = []
solution = []
solution.append(y0)
time = []
time.append(t0)

while (i < nMax-1 and t < tf):

    if (h > hMax): h = hMax

    if (h < hMin): h = hMin

    # R4 and R5 calculations

    k1 = h * fPrime(t, y)
    k2 = h * fPrime(t + 0.25*h , y + 0.25*k1)
    k3 = h * fPrime(t + (3/8)*h , y + (3/32)*k1 + (9/32)*k2)
    k4 = h * fPrime(t + (12/13)*h , y + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
    k5 = h * fPrime(t + h , y + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
    k6 = h * fPrime(t + 0.5*h , y - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1865/4104)*k4 - (11/40)*k5)

    y4 = y + w4[0]*k1 + w4[1]*k2 + w4[2]*k3 + w4[3]*k4 + w4[4]*k5
    y5 = y + w5[0]*k1 + w5[1]*k2 + w5[2]*k3 + w5[3]*k4 + w5[4]*k5 + w5[5]*k6

    e = np.abs(y4 - y5)

    if (e > eMax and h > hMin):
        h = 0.5*h
    else:
        t = t + h
        y = y5
        solution.append(y5)
        time.append(t)
        error.append(e)
        i += 1

        if (e < eMin): h = 2*h

    if (h < hMin and e > eMax):
        i = nMax
        y = 0
        print('Wrong solver parameters!')


#test_time = np.arange(t0, tf, 0.1)
exact = exact_sol(np.array(time))

plt.plot(time, exact, label='exact solution')
plt.plot(time, solution, label='R-K method')
plt.legend()
#plt.plot(time,error)
plt.show()

plt.plot(error)
plt.show()