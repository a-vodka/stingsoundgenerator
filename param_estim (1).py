import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import wavfile
from sympy.ntheory import factorint

from scipy.integrate import quad

l = 0.87
l0 = 0.18
ls = 0.1274
h0 = 0.5e-2


def omega(k):
    return 2.0 * np.pi * 55.0 * k


def ak(k):
    return 2. * l ** 2 / (np.pi ** 2 * k ** 2 * (l - l0) * l0) * np.sin(np.pi * k * l0 / l)

    #return (2.*(-2.*np.pi*l0*k*(l-l0)*np.cos(l0*np.pi*k/l)+l*np.sin(l0*np.pi*k/l)*(l-2*l0)))*h0/(l0*np.pi**2*k**2*(l-l0))


def bk(k, a, b):

    res = np.zeros(k.size)
    for i in range(1, res.size):
        f = lambda x: b*np.exp(-(x - l0) ** 2 / a**2)*np.sin(np.pi*k[i]*x/l)
        res[i], _ = quad(f, 0, l)

    return res + eps(k) * ak(k)


def eps(k):
    return 6e-4


amp = np.array([ 0.84428443,  1.19126206,  1.00743436,  0.19814912,  0.06228217,  0.1999152, 0.26437769,  0.03857564,  0.05373119])
#amp = np.array([293.22367809,   825.80286961,  1049.44812622,   275.4124454,    108.52872887,   429.64901827,  644.91530325] )
amp /= amp.max()

func = lambda k, a, b, c: np.sqrt(c * ak(k)**2 + bk(k, a, b)**2)

from scipy.optimize import curve_fit

fit_y = amp
fit_x = np.arange(1., amp.size+1, 1.)

popt, pcov = curve_fit(func, fit_x, fit_y)

print popt

plt.plot(fit_x, func(fit_x, *popt))
plt.plot(fit_x, fit_y)

plt.show()