# === ORDINARY DIFFERENTIALL EQUATIONS ===
# ========================================
# ===== Runge-Kutta Adaptive Method ======
# ========================================

import numpy as np
import matplotlib.pyplot as plt



def solverRungeKutta(yPrime, y0, t0, tF, eMin, eMax, h0 = 0.06, hMin = 0.001, hMax = 0.2, nMax = 10000):

    # =============================================================
    # Runge-Kutta equations (for R4 and R5 approximations)
    # =============================================================

    # weight coefficients
    w4 = np.zeros(5)
    w4[0] = 25/216
    w4[1] = 0
    w4[2] = 1408/2565
    w4[3] = 2197/4104
    w4[4] = -1/5
    #w4 vector 5 x 1

    w5 = np.zeros(6)
    w5[0] = 16/135
    w5[1] = 0
    w5[2] = 6656/12825
    w5[3] = 28561/56430
    w5[4] = -9/50
    w5[5] = 2/55
    #w5 vector 6 x 1


    # ============================
    # SOLVER LOOP
    # ============================

    n = y0.shape[0] # number of solutions

    # initial values of variables
    h = h0 # time step
    t = t0 # current time
    
    #y = np.zeros((n,nMax))
    y = y0 # y(t0) value

    i = 0 # iterator initial value
    error = []
    solution = []
    solution.append(y0)
    time = []
    time.append(t0)

    while (i < nMax-1 and t < tF):

        if (h > hMax): h = hMax

        if (h < hMin): h = hMin

        # Matrix of k coefficients (n x 6) where n - order of equation
     
        k = np.zeros((n,6))

        k[:,0] = h * yPrime(t, y)
        k[:,1] = h * yPrime(t + 0.25*h , y + 0.25*k[:,0])
        k[:,2] = h * yPrime(t + (3/8)*h , y + (3/32)*k[:,0] + (9/32)*k[:,1])
        k[:,3] = h * yPrime(t + (12/13)*h , y + (1932/2197)*k[:,0] - (7200/2197)*k[:,1] + (7296/2197)*k[:,2])
        k[:,4] = h * yPrime(t + h , y + (439/216)*k[:,0] - 8*k[:,1] + (3680/513)*k[:,2] - (845/4140)*k[:,3])
        k[:,5] = h * yPrime(t + 0.5*h , y - (8/27)*k[:,0] + 2*k[:,1] - (3544/2565)*k[:,2] + (1856/4104)*k[:,3] - (11/40)*k[:,4])

        # R4 and R5 calculations

        yR4 = y + np.matmul(k[:,:-1],w4)
        yR5 = y + np.matmul(k,w5)
        
        e = np.abs(yR4 - yR5)

        if (np.max(e) > eMax and h > hMin):
            h = 0.5*h
        else:
            t = t + h
            y = yR5
            solution.append(yR5)
            time.append(t)
            error.append(np.max(e))
            i += 1

            if (np.min(e) < eMin): h = 2*h

        if (h < hMin and np.max(e) > eMax):
            i = nMax
            y = 0
            print('Wrong solver parameters!')

    solution = np.array(solution).transpose()
    time = np.array(time)

    return solution, time