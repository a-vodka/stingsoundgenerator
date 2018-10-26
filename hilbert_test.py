import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def FFT(x, dTT):
    x_fft = np.fft.rfft(x)
    rho = 2.0 * np.abs(x_fft) / x.size
    angle = np.angle(x_fft)
    fr = np.fft.rfftfreq(x.size, d=dTT)
    return fr, rho, angle


def estimateDampingHilbert(y, t):
    # Hilbert transform method
    hb_bband = hilbert(y)

    lb, rb = int(0.01 * y.size), int(0.99 * y.size)
    envelop = np.abs(hb_bband)[lb:rb]
    t1 = t[lb:rb]
    envelop_log = np.log(envelop)

    cond = envelop > 0.3 * np.max(envelop)

    A = np.vstack([t1[cond], np.ones(t1[cond].size)]).T
    m, c = np.linalg.lstsq(A, envelop_log[cond])[0]

    plt.plot(t1, envelop_log, t1[cond], m * t1[cond] + c)
    plt.show()

    return -m, hb_bband


def fYoshida(x):
    # x = A*cos(Om*n+p).*exp(-n*D)
    N = x.size
    Xw = np.fft.fft(x)[1:N / 2]
    # [Xabs, ind] = max(abs(Xw(K)))
    Xabs = np.max(np.abs(Xw))
    ind = np.argmax(np.abs(Xw))

    plt.plot(np.abs(Xw))
    plt.show()

    k = np.array([ind - 1, ind, ind + 1])
    if ind - 2 > 0:
        if np.abs(Xw[ind + 2]) > np.abs(Xw[ind - 2]):
            k = np.append(k, ind + 2)
        else:
            k = np.append(ind - 2, k)
    else:
        k = np.append(k, ind + 2)

    R = (Xw[k[0]] - 2. * Xw[k[1]] + Xw[k[2]]) / (Xw[k[1]] - 2. * Xw[k[2]] + Xw[k[3]])  # (8.1)
    D = (2.0 * np.pi) / N * np.imag(-3. / (R - 1.) - 1.)  # (8.2)
    if k[3] - ind == 1:
        Om = (2.0 * np.pi / N) * np.real(ind - 1. - 3. / (R - 1.) - 2.)  # (8.2)
    else:
        Om = (2.0 * np.pi / N) * np.real(ind - 1. - 3. / (R - 1.) - 1.)  # (8.2)

    return Om, D


def fIpDFTd(x, M):
    # x = A*cos(Om*n+p).*exp(-n*D)
    # M - order of RVCI window form 0 to 6
    # % 0-rectangular window, 1-Hanning (Hann) window
    N = x.size
    K = np.arange(0, np.round(N / 2.), 1, dtype=int)

    wind, Am = window_cos(N, M)  # function defined in Program 1
    # may be used only for RVC1 windows
    # that is for M=0,...,6
    plt.plot(x * wind)
    plt.show()
    Xdft = np.fft.fft(x * wind)

    plt.plot(np.abs(Xdft))
    plt.show()

    R1, R2, ind, Vk = ratio(Xdft, K)
    delt = -(2. * M + 1.) / 2. * (R1 - R2) / (2. * (M + 1.) * R1 * R2 - R1 - R2 - 2. * M)  # (8.5)
    # delt=0.5
    while np.abs(0.5 - delt) < 1e-3:
        x = np.append(x, 0)
        N = len(x)
        wind, Am = window_cos(N, M)
        Xdft = np.fft.fft(x * wind)
        R1, R2, ind, Vk = ratio(Xdft, K)
        delt = -(2 * M + 1) / 2 * (R1 - R2) / (2 * (M + 1) * R1 * R2 - R1 - R2 - 2 * M)

    # Damping from (8.6a) and (8.6b)
    D1 = np.abs((2 * np.pi / N) * np.sqrt(np.abs(((delt + M) ** 2 - R1 * (delt - M - 1) ** 2) / (R1 - 1))))
    D2 = np.abs((2 * np.pi / N) * np.sqrt(np.abs(((delt - M) ** 2 - R2 * (delt + M + 1) ** 2) / (R2 - 1))))
    # Frequency from (8.3)
    if ind[2] > ind[3]:
        Om = (K[ind[1]] - 1 + delt) * 2 * np.pi / N
    else:
        Om = (K[ind[1]] - 1 - delt) * 2 * np.pi / N

    return Om, D1, D2


def ratio(Xdft, K):
#    Xabs, ind = np.max(np.abs(Xdft[K]))

    Xabs = np.max(np.abs(Xdft[K]))
    ind = np.argmax(np.abs(Xdft[K]))

    Vk = Xdft[K[ind]]
    Xabs_2 = 0
    Xabs_3 = 0
    ind_2 = 0
    ind_3 = 0
    if K[ind] > 1:
        if np.abs(Xdft[K[ind] + 1]) > np.abs(Xdft[K[ind] - 1]):
            Xabs_2 = np.abs(Xdft[K[ind] + 1])
            Xabs_3 = np.abs(Xdft[K[ind] - 1])
            ind_2 = ind + 1
            ind_3 = ind - 1
        else:
            Xabs_2 = np.abs(Xdft[K[ind] - 1])
            Xabs_3 = np.abs(Xdft[K[ind] + 1])
            ind_2 = ind - 1
            ind_3 = ind + 1

    else:
        Xabs_2 = np.abs(Xdft[K[ind] + 1])
        Xabs_3 = Xabs_2
        ind_2 = ind + 1
        ind_3 = ind - 1


    R1 = (Xabs_2 / Xabs)**2  # (8.4)
    R2 = (Xabs_3 / Xabs)**2  # (8.4)

    return R1, R2, np.array([0, ind, ind_2, ind_3]), Vk


def window_cos(N, ord):
    # Exemplary cosine windows
    # N - window length
    # ord - window type
    # ord = 0, 1, 2, 6 RVCI windows 0=rectangular, 1=Hanning
    # ord=7 - Hammig
    # ord=8 - Blackman
    A = np.zeros((10, 8))
    A[1, 1:] = np.array([ 1, 0, 0, 0, 0, 0, 0])
    A[2, 1:] = np.array([ 1, 1, 0, 0, 0, 0, 0])
    A[3, 1:] = np.array([ 1, 4. / 3., 1. / 3., 0, 0, 0, 0])
    A[4, 1:] = np.array([ 1, 3. / 2., 3. / 5., 1. / 10., 0, 0, 0])
    A[5, 1:] = np.array([ 1, 8. / 5., 4. / 5., 8. / 35., 1. / 35., 0, 0])
    A[6, 1:] = np.array([ 1, 105. / 63., 60. / 63., 45. / 126., 5. / 63., 1. / 126., 0])
    A[7, 1:] = np.array([ 1, 396. / 231., 495. / 462., 110. / 231., 33. / 231., 6. / 231., 1. / 462.])
    A[8, 1:] = np.array([ 0.54, 0.46, 0, 0, 0, 0, 0])
    A[9, 1:] = np.array([ 0.42, 0.5, 0.08, 0, 0, 0, 0])
    dw = 2.0 * np.pi / N
    Om = np.arange(0, N , 1) * dw
    NW = A[ord + 1, :]

    ind = np.flatnonzero(NW)
    Am = NW[ind]
    wind = np.zeros(N)
    for k in range(0, ind.size):
        wind = wind + (-1) ** (k) * NW[k+1] * np.cos(k * Om)

    return wind, Am


def fMatPen(x, K):
    # x - analyzed signal - sum of damped sinusoids
    # K - assumed number of real damped sine components
    M = 2 * K  # number of complex components
    N = x.size  # number of signal samples
    L = int(np.floor(N / 3))  # linear prediction order L = N/3
    from scipy.linalg import hankel, svd, eig
    X = hankel(x[0:N - L], x[N - L:N])  # X1=X(:,2:L+1),X0=X(:,1:L)
    U, S, V = svd(X[:, 0:L+1], 0)
    S = np.diag(S)  # SVD of X1
    p = np.log(np.matmul(eig(np.diag(1. / S[0:M]), np.matmul((np.matmul(U[:, 0:M].conj().T , X[:, 0:L]) ) , V[:, 0:M]))))
    Om = np.imag(p)
    Om, indx = np.sort(Om, order='descend')
    Om = Om[0:K]  # frequency
    D = np.real(p[indx[0:K]])  # damping
    return Om, D


def main():
    duration = 2.0 * np.pi * 100.
    fs = 400.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs

    y = 1.0 * np.cos(110.0 * t + 2.7) * np.exp(-0.001 * t)


    freq, rho, phi =  FFT(y, dTT=t[1])

    nm = np.argmax(rho)

    print rho[nm], phi[nm]
    print plt.axes().transAxes

    plt.plot(freq, phi)
    plt.plot(freq, rho)
    plt.xlim(0, 50)
    plt.text(0.5, 0.95, 'hello', transform=plt.axes().transAxes, horizontalalignment='center')
    plt.show()

    # #print fMatPen(y, 1)
    # Om, D1, D2 =  fIpDFTd(y, 1)
    # print Om*fs, D1*fs, D2*fs, D1
    # Omega, damp = fYoshida(y)
    # print Omega * fs, damp * fs
    # m, hb_bband = estimateDampingHilbert(y, t)
    # print m
    #
    # phase = np.angle(hb_bband)
    # plt.plot(t, np.abs(hb_bband), t, y, t, phase)
    # plt.show()


main()
