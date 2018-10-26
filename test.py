import numpy as np
# import sympy as sp
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import scipy.fftpack as spf
data = np.array([1.0, 2.0, 3, 4], dtype=float)


N = data.size

g = np.zeros_like(data)

print range(N)
for i in range(1, N+1):
    for v in range(N):
        cot = 1.0/np.tan(v - i)
        if cot == np.inf:
            cot = 0
        g[i-1] += data[i-1]*(1.0 - int((-1)**(v-i)))*cot*np.pi/N

g /= N

print N/2
print g

h = hilbert(data)
h1 = spf.hilbert(data)


print h
print h1

print np.abs(h)
print np.abs(h1)