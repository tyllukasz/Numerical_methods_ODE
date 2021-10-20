# === ORDINARY DIFFERENTIALL EQUATIONS ===
# ========================================
# ======== Taylor/Euler Method ===========
# ========================================

import numpy as np
import matplotlib.pyplot as plt

# ========================================
# function to find solution
# y' = - y + exp(-t)
# y(t0) = 0
# ========================================

# derivatives
# 1st derivative:
f_1st_der = lambda y, t: -y + np.exp(-t)

# 2nd derivative
f_2nd_der = lambda y, t: y - 2 * np.exp(-t)

# ========================================

h = 0.1 # time step [sec]
t = np.arange(0,5,h) # time series range for calculations [sec]

# solution for n = 1
y1 = np.zeros(len(t)) # array to store values for (n=1) solution
y1[0] = 0 # boundary condition

for i in range (0,len(t)-1):
    
    y1[i+1] = y1[i] + f_1st_der(y1[i] , t[i]) * h


# solution for n = 2
y2 = np.zeros(len(t)) # array to store values for (n=1) solution
y2[0] = 0 # boundary condition

for i in range (0,len(t)-1):

   y2[i+1] = y2[i] + f_1st_der(y2[i] , t[i]) * h + 0.5 * f_2nd_der(y2[i] , t[i]) * h**2

# exact solution
exact_sol = lambda t: t * np.exp(-t)
y_exact = exact_sol(t)

plt.plot(t,y_exact, label = 'Exact solution')
plt.plot(t, y1, label = 'Numerical sol n=1')
plt.plot(t, y2, label = 'Numerical sol n=2')
plt.legend()
plt.show()

