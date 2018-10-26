import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import wavfile
from sympy.ntheory import factorint


def FFT(x, dTT):
    x_fft = np.fft.rfft(x)
    rho = 2.0 * np.abs(x_fft) / x.size
    angle = np.angle(x_fft)
    fr = np.fft.rfftfreq(x.size, d=dTT)
    return fr, rho, angle


filename1 = 'test.wav'
filename2 = './narezannye_zvuki/string_1_fret_0.wav'

samplerate1, data1 = wavfile.read(filename1)
samplerate2, data2 = wavfile.read(filename2)

data1 = np.array(data1, dtype=float)
data2 = np.array(data2, dtype=float)

dt1 = 1.0 / samplerate1
dt2 = 1.0 / samplerate2

data1 /= np.max(np.abs(data1))
data2 /= np.max(np.abs(data2))

t1 = np.linspace(0, data1.size * dt1, data1.size)
t2 = np.linspace(0, data2.size * dt2, data2.size)


plt.plot(t1 ,data1)
plt.plot(t2,data2)

plt.show()


fr1, rho1, angle1 = FFT(data1, dt1)
fr2, rho2, angle2 = FFT(data2, dt1)

plt.plot(fr1, rho1 / rho1.max() )
plt.plot(fr2, rho2 / rho2.max())
plt.xlim(0, 1000)
plt.show()