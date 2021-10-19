from os import stat
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import scipy.signal
import scipy.stats
from scipy.stats.stats import kurtosis, skew
import seaborn as sns

#=========FUNCTIONS=========

# gaussian distribution
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#===========================


# contributor distribution definition
tol_width = 16
c_dist = np.ones(tol_width + 1) # +/- 8 units per contributor
c_dist_const = np.ones(tol_width + 1)


# empty lists initialization to store kurtosis and skewness values duirng iterations
my_kurtosis = []
my_skewness = []

my_std_dev = []
my_mean = []

no_contr = 6 # number of contributors

for i in range(0,no_contr-1):

    conv = scipy.signal.fftconvolve(c_dist , c_dist_const) #convolution
    conv = conv / max(conv) #normalization (0:1)

    #set reconstruction based on pdf (result of convolution)
    temp_set = []
    temp_len = len(conv)

    for i0 in range(0,temp_len):

        for i1 in range(0, int(100 * conv[i0])):
            temp_set.append(i0)

    actual_kurt = kurtosis(temp_set)
    actual_skew = skew(temp_set)
    actual_std_dev = np.std(temp_set)
    actual_mean = np.mean(temp_set)

    my_kurtosis.append(actual_kurt)
    my_skewness.append(actual_skew)
    my_std_dev.append(actual_std_dev)
    my_mean.append(actual_mean)

    c_dist = conv



my_len = len(c_dist)
my_x = np.arange(0,my_len)

my_gauss = gaussian(my_x, my_mean[-1], my_std_dev[-1])

tolerance_stat = my_std_dev[-1] * 6 * 1.33 #for Cp = 1.33
tolerance_wc = tol_width * no_contr

print(f'Mean = {my_mean[-1]}')
print(f'Standard deviation = {my_std_dev[-1]}')
print(f'Statistic tolerance = {tolerance_stat}')
print(f'Worst Case tolerance = {tolerance_wc}')
print(f'Ratio Stat/WC = {tolerance_stat/tolerance_wc}')





plt.plot(c_dist, label = 'real dist')
plt.plot(my_gauss, label = 'syntetic dist')
plt.legend()
plt.show()
plt.plot(my_kurtosis)
plt.show()

