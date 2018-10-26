import numpy as np
#import sympy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import wavfile
from sympy.ntheory import factorint

from scipy.integrate import quad


def FFT(x, dTT):
    x_fft = np.fft.rfft(x)
    rho = 2.0 * np.abs(x_fft) / x.size
    angle = np.angle(x_fft)
    fr = np.fft.rfftfreq(x.size, d=dTT)
    return fr, rho, angle


def cmpwav():
    #filename1 = '0vel.wav'
    #filename1 = 'test.wav'
    #filename2 = './narezannye_zvuki/string_3_fret_0.wav'
    #filename2 = './cort1/a_0.wav'

    samplerate1, data1 = wavfile.read(filename1)
    samplerate2, data2 = wavfile.read(filename2)

    data1 = np.array(data1, dtype=float)
    data2 = np.array(data2, dtype=float)

    dt1 = 1.0 / samplerate1
    dt2 = 1.0 / samplerate2

    data1 /= np.max(np.abs(data1))
    data2 /= np.max(np.abs(data2))

    time_shift1 = np.argmax(np.abs(data1)) * dt1
    time_shift2 = np.argmax(np.abs(data2)) * dt2
    arg_shift2 = np.argmax(np.abs(data2))
    t1 = np.linspace(0, data1.size * dt1, data1.size) - time_shift1
    t2 = np.linspace(0, data2.size * dt2, data2.size) - time_shift2

    ddata2 = np.gradient(data2, dt2)
    ddata1 = np.gradient(data1, dt1)
    nn = 3000
    plt.plot(data2[ arg_shift2: arg_shift2+nn], ddata2[ arg_shift2: arg_shift2+nn])
    #plt.plot(data2, ddata2)
    #plt.show()

    plt.plot(data1[0:nn], ddata1[0:nn])
    plt.show()

    plt.plot(t1, data1)
    plt.plot(t2, data2, '--')
    plt.xlim(0,  0.11)
    plt.show()

    fr1, rho1, angle1 = FFT(data1, dt1)
    fr2, rho2, angle2 = FFT(data2, dt1)

    plt.plot(fr1, rho1 / rho1.max())
    plt.plot(fr2, rho2 / rho2.max())
    plt.xlim(0, 500)
    plt.show()


FACTOR_LIMIT = 128

nf = 0

# l = 0.87
ll = 0.8728  # standart bass-guitar measure
l = ll / (2.0 ** (nf / 12.0))
l0 = 0.18
ls = 0.1274
#h0 = 0.5e-2
h0 = 0.89951946
#h0 = 0.14019215

a_arr = [98.0, 73.41, 55.0, 41.2]

a0 = a_arr[0] * (2.0 ** (nf / 12.0)) * 2 * np.pi
a = a0 * l / np.pi

z = 20


def omega(k):
    return 2.0 * np.pi * 55.0 * k


def ak(k):
    return 2. * l ** 2 * h0 / (np.pi ** 2 * k ** 2 * (l - l0) * l0) * np.sin(np.pi * k * l0 / l)

    #return (2.*(-2.*np.pi*l0*k*(l-l0)*np.cos(l0*np.pi*k/l)+l*np.sin(l0*np.pi*k/l)*(l-2*l0)))*h0/(l0*np.pi**2*k**2*(l-l0))


def bk(k):
    a = 0.11163776
    b = 5.96577607

#    a = 0.64632872
#    b = 7.42273815

    f = lambda x: b*np.exp(-(x - l0) ** 2 / a**2)*np.sin(np.pi*k*x/l)
    bki, _ = quad(f, 0, l)
    return bki + eps(k) / omega(k) * ak(k)


def eps(k):
    return 8e-4


if False:
    ii = np.arange(0, z, 1)
    betai = np.zeros_like(ii, dtype=float)
    for i in ii:
        betai[i] = eps(i)
        print i, betai[i]

    plt.plot(ii, betai)
    plt.show()

if True:
    nn = np.arange(1.0, 10.0, 1.0)
    plt.plot(nn, ak(nn))
    plt.grid()
    plt.show()
    bk1 = np.zeros_like(nn, dtype=float)
    for i in range(0, nn.size):
        bk1[i] = bk(nn[i])
    plt.plot(nn, bk1)
    plt.grid()
    plt.show()

if False:
    la = np.linspace(0, l, 100)
    zl = np.zeros_like(la)
    for i in range(1, z):
        zl += ak(i) * np.sin(np.pi * i * la / l)

    plt.plot(la, zl)
    plt.show()

dt = 1.0 / 44100.0
t = np.arange(0, 5, dt)

y = np.zeros_like(t)
v = np.zeros_like(t)

for k in range(1, z):
    yk = np.sin(np.pi * k * ls / l) * (bk(k) * np.sin(omega(k) * t) + ak(k) * np.cos(omega(k) * t)) * np.exp(
        -eps(k) * omega(k) * t)
    vk = - np.gradient(y, t[1] - t[0])
    y += yk
    v += vk

#wavfile.write("test.wav", 44100, y / np.max(np.abs(y)) * 0.9)
wavfile.write("test.wav", 44100, v / np.max(np.abs(v)) * 0.9)

cmpwav()
