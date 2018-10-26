import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import wavfile
from sympy.ntheory import factorint
from scipy.optimize import brenth, brentq
from waw_cmp import cmpdata
from scipy.optimize import curve_fit

FACTOR_LIMIT = 128


def FFT(x, dTT):
    x_fft = np.fft.rfft(x)
    rho = 2.0 * np.abs(x_fft) / x.size
    angle = np.angle(x_fft)
    fr = np.fft.rfftfreq(x.size, d=dTT)
    return fr, rho, angle


def wnd(inc_time, dt, size):
    res = np.ones(size)
    n = int(inc_time / dt)
    res[:n] *= np.arange(0, n, 1) * dt / inc_time
    return res


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[window_len - 2:-1]


def estimateDampingHilbert(y, t, plot=False):
    from scipy.signal import hilbert
    # Hilbert transform method
    hb_bband = hilbert(y)

    lb, rb = int(0.01 * y.size), int(0.99 * y.size)
    envelop = np.abs(hb_bband)[lb:rb]
    t1 = t[lb:rb]
    envelop_log = np.log(envelop)

    cond = envelop > 0.09 * np.max(envelop)

    A = np.vstack([t1[cond], np.ones(t1[cond].size)]).T
    m, c = np.linalg.lstsq(A, envelop_log[cond])[0]

    if plot:
        plt.close()
        plt.plot(t1, envelop_log, t1[cond], m * t1[cond] + c)
        plt.show()

    return -m, hb_bband


def fYoshida(rho, ind):
    # x = A*cos(Om*n+p).*exp(-n*D)
    N = rho.size
    Xw = rho
    # plt.plot(np.abs(rho))
    # plt.show()

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

    return D / 2.0


def fIpDFTd(x, M):
    # x = A*cos(Om*n+p).*exp(-n*D)
    # M - order of RVCI window form 0 to 6
    # % 0-rectangular window, 1-Hanning (Hann) window
    N = x.size
    K = np.arange(0, np.round(N / 2.), 1, dtype=int)

    wind, Am = window_cos(N, M)  # function defined in Program 1
    # may be used only for RVC1 windows
    # that is for M=0,...,6
    # plt.plot(x * wind)
    # plt.show()
    Xdft = np.fft.fft(x * wind)

    # plt.plot(np.abs(Xdft))
    # plt.show()

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

    return (D1 + D2) / 2.


def ratio(Xdft, K):
    # Xabs, ind = np.max(np.abs(Xdft[K]))

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

    R1 = (Xabs_2 / Xabs) ** 2  # (8.4)
    R2 = (Xabs_3 / Xabs) ** 2  # (8.4)

    return R1, R2, np.array([0, ind, ind_2, ind_3]), Vk


def window_cos(N, ord):
    # Exemplary cosine windows
    # N - window length
    # ord - window type
    # ord = 0, 1, 2, 6 RVCI windows 0=rectangular, 1=Hanning
    # ord=7 - Hammig
    # ord=8 - Blackman
    A = np.zeros((10, 8))
    A[1, 1:] = np.array([1, 0, 0, 0, 0, 0, 0])
    A[2, 1:] = np.array([1, 1, 0, 0, 0, 0, 0])
    A[3, 1:] = np.array([1, 4. / 3., 1. / 3., 0, 0, 0, 0])
    A[4, 1:] = np.array([1, 3. / 2., 3. / 5., 1. / 10., 0, 0, 0])
    A[5, 1:] = np.array([1, 8. / 5., 4. / 5., 8. / 35., 1. / 35., 0, 0])
    A[6, 1:] = np.array([1, 105. / 63., 60. / 63., 45. / 126., 5. / 63., 1. / 126., 0])
    A[7, 1:] = np.array([1, 396. / 231., 495. / 462., 110. / 231., 33. / 231., 6. / 231., 1. / 462.])
    A[8, 1:] = np.array([0.54, 0.46, 0, 0, 0, 0, 0])
    A[9, 1:] = np.array([0.42, 0.5, 0.08, 0, 0, 0, 0])
    dw = 2.0 * np.pi / N
    Om = np.arange(0, N, 1) * dw
    NW = A[ord + 1, :]

    ind = np.flatnonzero(NW)
    Am = NW[ind]
    wind = np.zeros(N)
    for k in range(0, ind.size):
        wind = wind + (-1) ** (k) * NW[k + 1] * np.cos(k * Om)

    return wind, Am


def genSound_s1(res_arr, t):
    # print 'N\tNN\tomega\tA\tAmax\tbeta\tphi\tdelta\tdzeta\tm\tyoshida\tDFT1\tDFT6'

    out_float = np.zeros_like(t)
    for i in range(res_arr[:, 0].size):
        omega0 = res_arr[i, 2]
        A = res_arr[i, 3]
        phi = res_arr[i, 4]
        beta = res_arr[i, 5]
        delta = res_arr[i, 6]
        Amax = res_arr[i, 7]
        zeta = res_arr[i, 8]
        m = res_arr[i, 9]
        yoshida = res_arr[i, 10]
        DFT1 = res_arr[i, 11]

        damp = (yoshida + m + beta + DFT1) / 4.0

        out_float += A * np.exp(-damp * t) * np.cos(2.0 * np.pi * omega0 * t + phi)

        print i + 1, '\t', res_arr[
            i, 1], '\t', omega0, '\t', A, '\t', Amax, '\t', beta, '\t', phi, '\t', delta, '\t', zeta, '\t', m, '\t', yoshida, '\t', \
            res_arr[i, 11], '\t', res_arr[i, 12], '\t', damp

    vk = - np.gradient(out_float, t[1])
    return vk


def genSound_s2(res_arr, t, fret_num=0):
    omega0 = res_arr[0, 2]
    beta = res_arr[:, 5]
    m = res_arr[:, 9]
    yoshida = res_arr[:, 10]
    DFT1 = res_arr[:, 11]

    ############
    l = 0.8728  # sting length
    l0 = 0.18  # distance to pitch

    h0 = 1.0

    def omega(k):
        return 2.0 * np.pi * omega0 * k

    # omega_rad_arr = omega(np.arange(1., res_arr[:, 0].size + 1., 1.))
    # damp = np.mean((yoshida + m + beta + DFT1) / 4.0 / omega_rad_arr)

    def ak(k):
        return 2. * l ** 2 * h0 / (np.pi ** 2 * k ** 2 * (l - l0) * l0) * np.sin(np.pi * k * l0 / l)

    def eps(k):
        # return (0.46 * fret_num + 3.8) * 1e-4 # Ibanezz G-string
        # return (0.172 * fret_num + 4.32) * 1e-4  # Cord D-string
        # return (1.82 * fret_num + 2.7)*1e-4 # Yamoha bass A-string

        coef = np.array([2.46580270e-01, 4.92355006e-14, 4.16582697e-14, 2.21483036e-10])
        return func(omega(k), *coef) / omega(k)

    def bk(k):
        return eps(k) / omega(k) * ak(k)

    ############

    # omega0_c = omega0[:-2]

    # func = lambda x, a, b, c, d, e : a*x**2+b*x+c+d*x**3+e*x**4
    # func = lambda x, a, b, c: a * np.exp(b * x)
    # func = lambda x, a, c: a * x ** 2 + c
    # func = lambda x, a, b, c: a * x ** 2 + b * x + c
    # func = lambda x, c: 0 * x + c
    # from scipy.optimize import curve_fit
    # fit_y = np.concatenate((res[:-2, 5]/omega0_c, res[:-2, 9]/omega0_c, res[:-2, 10]/omega0_c, res[:-2, 11]/omega0_c))
    # fit_x = np.concatenate((res[:-2, 1], res[:-2, 1], res[:-2, 1], res[:-2, 1]))

    # fit_y = np.concatenate((res[:, 5] / omega0, res[:, 9] / omega0, res[:, 10] / omega0, res[:, 11] / omega0))
    # fit_x = np.concatenate((res[:, 1], res[:, 1], res[:, 1], res[:, 1]))

    # popt, pcov = curve_fit(func, fit_x, fit_y)

    # coef[:, k] = popt
    # perr = np.sqrt(np.diag(pcov))
    # print 'error = ', perr
    # data = func(res[:, 1], *popt)
    # plt.plot(res[:, 1], func(res[:, 1], *popt), 'k--', label='fit')
    # y_data[:data.size, k] = data
    # k += 1

    # plt.savefig('fitting.png', dpi=600)
    # plt.savefig('fitting.eps')

    out_float = np.zeros_like(t)

    # for i in range(1, res_arr[:, 0].size):
    for i in range(1, 50):
        out_float += np.sin(np.pi * i * l0 / l) * (
                bk(i) * np.sin(omega(i) * t) + ak(i) * np.cos(omega(i) * t)) * np.exp(
            -eps(i) * omega(i) * t)

    vk = -np.gradient(out_float, t[1])
    return vk


func = lambda x, d0, d1, d2, d3: d0 + d1 * np.sqrt(x) + d2 * x + d3 * x ** 3
func_lin = lambda x, d0, d1: d0 + d1 * x


def dampplot(res_arr, save_to_file=False):
    omega = res_arr[:, 2]
    beta = res_arr[:, 5]
    m = res_arr[:, 9]
    yoshida = res_arr[:, 10]
    DFT1 = res_arr[:, 11]

    omega_rad = omega * 2 * np.pi
    avg_damp = (yoshida + m + beta + DFT1) / 4.0
    damp = np.mean(avg_damp / omega_rad)

    fit_y = np.concatenate((yoshida, m, beta, DFT1))
    fit_x = np.concatenate((omega_rad, omega_rad, omega_rad, omega_rad))

    popt, pcov = curve_fit(func, fit_x, fit_y, bounds=(0, np.inf))
    popt_lin, pcov_lin = curve_fit(func_lin, fit_x, fit_y)
    perr = np.sqrt(np.diag(pcov))
    perr_lin = np.sqrt(np.diag(pcov_lin))
    print 'error = ', perr
    print 'coef = ', popt
    print 'error_lin = ', perr_lin
    print 'coef_lin = ', popt_lin

    # plt.show()

    freq_num = np.arange(1, omega.size + 1, 1)

    plt.plot(omega_rad, func(omega_rad, *popt), 'k--', label='fit')
    plt.plot(omega_rad, func_lin(omega_rad, *popt_lin), 'k--', label='fit lin')
    # ax1.plot(omega_rad, avg_damp, label='damp')
    # ax1.set_xlabel('k, frequency number')

    plt.plot(omega_rad, beta, label='Half-power bandwidth')
    plt.plot(omega_rad, m, label='Hilbert transform')
    plt.plot(omega_rad, yoshida, label='Yoshida')
    plt.plot(omega_rad, DFT1, label='IpDFT')
    # ax1.plot(omega_rad, avg_damp, '--', label='avg', color='k')

    # plt.axhline(damp, 0.0, 1.0, linestyle=':', color='k', label='mean {0:.3e}'.format(damp))
    plt.legend()
    plt.xlabel('k, frequency number')
    plt.ylabel('$\\overline{{\\varepsilon}}$')
    plt.grid()
    # plt.xlim(omega_rad.min(), omega_rad.max())
    # ax1.set_ylim(0, 1.4e-3)
    # ax1.set_xticks(np.arange(1, 15, 2))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
    # ax2.set_title('E-string')
    plt.tight_layout()
    if save_to_file:
        plt.savefig('e-damp-harm.png', dpi=300)
        plt.savefig('e-damp-harm.eps')

    plt.show()

    return popt, popt_lin


def savetowav(data, samplerate, filename):
    data /= np.max(np.abs(data)) * 0.9
    wavfile.write(filename, samplerate, data)


def main():
    # files = np.empty(n, dtype=object)
    # for i in range(n):
    # files[i] = open('workfile_' + str(i + 1) + '.csv', 'w')
    # test_tones = np.array([0, 1, 3, 5, 7, 12])
    # test_tones = np.arange(0, 13, 1)
    test_tones = np.array([0, 1, 2, 3, 4])
    # test_tones = np.array([0])
    damp_data = np.zeros((4, test_tones.size))
    damp_data_lin = np.zeros((2, test_tones.size))

    k = 0
    f = open('damp_values_yamaha_4str.csv', 'ab')

    for i in test_tones:
        #filename = './narezannye_zvuki/string_1_fret_' + str(i) + '.wav'
        #filename = './cort1/e_' + str(i) + '.wav'
        # filename = './Schecter Stiletto Elite/index g_' + str(i) + '.wav'
        # filename = './Ibanez_roadstarII_series_fingered/g_' + str(i) + '.wav'
        filename = './FS_Yamaha/e' + str(i) + '.wav'
        # file = open(filename, 'r')
        samplerate, data = wavfile.read(filename)

        t = np.arange(data.size) / float(samplerate)

        if False:
            data = 10.0 * np.sin(140 * 2.0 * np.pi * t) * np.exp(-0.5 * t) + 8.0 * np.sin(
                280 * 2.0 * np.pi * t + 1.0) * np.exp(-0.3 * t) + 0.0 * (np.random.rand(t.size) - 0.5)

        if np.ndim(data) > 1:
            data = data[:, 0]

        data = data - np.mean(data)

        # print int(np.ceil(np.log2(data.size)))
        new_size = 2 ** int(np.ceil(np.log2(data.size))-2)  ## nearest bigger power of 2
        # new_size = 2 ** 21
        # new_size = data.size
        data_zero_padded = np.zeros(new_size)
        if new_size >= data.size:
            data_zero_padded[0: data.size] = data
        else:
            data_zero_padded = data[0:new_size]

        res = proccess(data_zero_padded, samplerate, i, scale=0.06)
        res = res[np.nonzero(res[:, 2])[0], :]

        res[:, 0] = np.ones(res[:, 0].size) * i
        np.savetxt(f, res, delimiter=';')
        damp_data[:, k], damp_data_lin[:, k] = dampplot(res)
        # print test_tones[k], '\t', damp_data[k]
        k += 1

        t_padded = np.arange(data_zero_padded.size) / float(samplerate)

        #sound1 = genSound_s1(res, t)
        #cmpdata(sound1, samplerate, data, samplerate, picname=str(i) + '_cmp_method_1', plot=False)

        #sound2 = genSound_s2(res, t, fret_num=i)

        # savetowav(sound1, samplerate, str(i) + '_generated_method1.wav')
        # savetowav(sound2, samplerate, str(i) + '_generated_method2.wav')
        # savetowav(data, samplerate, str(i) + '_source.wav')

        #cmpdata(sound2, samplerate, data, samplerate, picname=str(i) + '_cmp_method_2', plot=False)
        print('------------------------')


    f.close()
    omega = np.linspace(40., 1000., 100)
    omega_rad = omega * 2 * np.pi

    for i in range(test_tones.size):
        plt.plot(omega, func(omega_rad, *damp_data[:, i]), 'C' + str(i) + '-', label=str(test_tones[i]))
        plt.plot(omega, func_lin(omega_rad, *damp_data_lin[:, i]), 'C' + str(i) + '--', label=str(test_tones[i]))
    plt.legend()
    plt.grid()
    plt.show()


#    plt.plot(test_tones, damp_data)
#    plt.xlabel('fret number')
#    plt.ylabel('damping')
#    plt.grid()
#    plt.show()


def deleteSameFreq(res_arr):
    res_arr_upd = np.zeros_like(res_arr)
    i = 0
    while i < res_arr[:, 0].size:
        target = res_arr[i, 2]
        alpha = 0.1
        abs_delta = np.abs((res_arr[:, 2] - target) / target)

        mask = abs_delta < alpha

        num_mask_nonzero = np.count_nonzero(mask)
        if num_mask_nonzero > 1:

            maxampindex = np.argmax(res_arr[mask, 1])  # max amplitude

            msk = mask[np.nonzero(mask)]
            msk[:] = False
            msk[maxampindex] = True
            np.put(mask, np.nonzero(mask), msk)
        else:
            num_mask_nonzero = 1
        res_arr_upd[i, :] = res_arr[mask, :]
        i += num_mask_nonzero

    res_arr_upd = res_arr_upd[res_arr_upd[:, 2] > 0]
    return res_arr_upd


def bestFFTlength(n):
    while max(factorint(n)) >= FACTOR_LIMIT and max(factorint(n).values()) < 3:
        n -= 1
    return n


def findParameters(omega0, freq, rho):
    omega0_indx = int(omega0 / freq[1])
    Amax = rho[omega0_indx]
    dfr = 30
    num_resampled = 5000

    freq_resampled = np.linspace(freq[omega0_indx - dfr], freq[omega0_indx + dfr], num_resampled, endpoint=False)
    rho_int = interp1d(freq[omega0_indx - dfr: omega0_indx + dfr + 1], rho[omega0_indx - dfr: omega0_indx + dfr + 1])
    rho_resampled = rho_int(freq_resampled)

    omega0_right_indx = omega0_left_indx = num_resampled / 2

    while rho_resampled[omega0_left_indx] > Amax / np.sqrt(2) and omega0_left_indx >= 0:
        omega0_left_indx -= 1

    while rho_resampled[omega0_right_indx] > Amax / np.sqrt(2) and omega0_right_indx < num_resampled - 1:
        omega0_right_indx += 1

    if omega0_right_indx == num_resampled - 2 or omega0_left_indx == 0:
        return 0, 0

    return freq_resampled[omega0_left_indx], freq_resampled[omega0_right_indx]


def proccess(data, samplerate, num_file=0, scale=0.09):
    dt = 1.0 / samplerate
    size = bestFFTlength(data.size)

    # print data.size, size

    t = np.linspace(0, size * dt, size)
    data = data[0:size]
    freq, rho, angle = FFT(data, dt)
    # significant_level = 0.07
    # significant_level = 0.05
    # significant_level = 0.07
    rho[0] = 0.0

    #rho[1:] /= (2.0 * np.pi * freq[1:])  # integrate spectrum

    # rho_int = smooth(rho, window_len=int(rho.size * significant_level)) * 15 / significant_level
    # rho_int = smooth(rho, window_len=int(rho.size * significant_level)) * 3.5 / significant_level
    rho_int = smooth(rho, window_len=rho.size / 100)
    rho_int = rho_int / rho_int.max() * (rho.max() - rho[freq < 1000].mean()) * scale + rho[freq < 1000].mean()
    # 50 Hz Filter
    freq1 = np.logical_and(freq > 49.5, freq < 50.5)
    # rho[freq1] = np.ones(rho.size)[freq1] * np.mean(rho[freq < 40.])
    rho[freq1] = 0.0
    rho[freq <= 30] = 0
    rho[freq > 2000] = 0
    # print rho.size, rho_int.size

    if False:
        plt.plot(t, data)

        plt.xlabel("t, s")
        plt.ylabel("A")
        plt.tight_layout()

        plt.savefig('signal.png', dpi=600)
        plt.savefig('signal.eps')
        plt.show()
        plt.close()

        plt.plot(freq, rho)
        plt.xlim(0, 600)
        plt.ylim(ymin=0)
        plt.xlabel('$\omega$, Hz')
        plt.ylabel('A')
        plt.tight_layout()
        plt.savefig('spectrum_signal1.png', dpi=600)
        plt.savefig('spectrum_signal1.eps')
        plt.show()
        plt.close()

        plt.plot(freq, rho, label='spectrum')
        plt.plot(freq, rho_int, label='smoothed spectrum')
        plt.xlim(0, 600)
        plt.ylim(ymin=0)
        plt.xlabel('$\omega$, Hz')
        plt.ylabel('A')
        plt.legend()
        plt.tight_layout()
        plt.savefig('spectrum_signal1_smooth.png', dpi=600)
        plt.savefig('spectrum_signal1_smooth.eps')
        plt.show()
        plt.close()

    # rho[freq > 15000] = 0

    plt.subplot(211)
    plt.plot(t, data)

    plt.subplot(212)
    plt.semilogy(freq, rho)
    plt.semilogy(freq, rho_int)
    # plt.plot([freq[0], freq[-1]], [rho.max() * significant_level, rho.max() * significant_level])
    plt.xlim(0, 1000)

    # plt.savefig('spectrum.png')
    plt.show()
    plt.close()

    # cond = rho > rho.max() * significant_level
    # cond = rho_smooth > rho_int
    cond = (rho > rho_int).astype(float)

    diff_cond = np.ediff1d(cond, to_begin=0)

    peak_freq = freq[diff_cond!=0]
    peak_freq_indexes = np.nonzero(diff_cond != 0)[0]

    #print peak_freq_indexes, freq[peak_freq_indexes], np.nonzero(cond), diff_cond[np.nonzero(diff_cond)]
    if peak_freq.size % 2 != 0:
        print "Number of freq bound is not odd  "
        exit(0)

    n = 0
    for i in range(0, peak_freq.size, 2):
        if peak_freq_indexes[i + 1] - peak_freq_indexes[i] > 3:
            n += 1
    # n = int(peak_freq.size / 2)

    res_arr = np.zeros((n, 13), dtype=float)

    numfreq = 15
    res_arr2 = np.zeros((numfreq, 6), dtype=float)

    fft_res = rho * np.exp(1.0j * angle)
    fl = True
    k = 0
    plt.plot(freq, rho)
    # for i in range(0, peak_freq.size, 2):
    for i in range(0, peak_freq.size, 2):
        # print i
        # print peak_freq_indexes[i],peak_freq_indexes[i+1]

        if peak_freq_indexes[i + 1] - peak_freq_indexes[i] <= 3:
            continue
        freq_cutted = freq[peak_freq_indexes[i] - 1:peak_freq_indexes[i + 1] + 1]
        rho_cutted = rho[peak_freq_indexes[i] - 1:peak_freq_indexes[i + 1] + 1]
        angle_cutted = angle[peak_freq_indexes[i] - 1:peak_freq_indexes[i + 1] + 1]
        # print  np.max(rho_cutted), np.min(rho_cutted), np.max(rho_cutted) / np.sqrt(2) > np.min(rho_cutted)

        omega0 = freq_cutted[np.argmax(rho_cutted)]
        omeg0_inx = np.argmax(rho_cutted) + peak_freq_indexes[i] - 1
        lb, rb = int(omeg0_inx - 0.03 * omega0 / freq[1]), int(omeg0_inx + 0.03 * omega0 / freq[1])
        # print lb, rb, freq[lb], freq[rb]

        domega = (omega0 - res_arr[k - 1, 2]) / omega0

        if domega < 0.1:
            continue

        if res_arr[0, 2] and omega0 / res_arr[0, 2] > 20:
            continue

        # plt.plot(freq_cutted, rho_cutted)
        plt.axvline(freq[lb], color='r', linewidth=0.5)
        plt.axvline(freq[rb], color='r', linewidth=0.5)

        rho_inv = np.zeros_like(rho)
        rho_inv[lb:rb] = rho[lb:rb]

        freq_cutted = freq[lb:rb]
        rho_cutted = rho[lb:rb]
        angle_cutted = angle[lb:rb]

        omega0 = freq_cutted[np.argmax(rho_cutted)]
        omeg0_inx = np.argmax(rho_cutted) + lb

        rev_rho = np.fft.irfft(rho_inv * np.exp(1.0j * angle))

        m, hb_bband = estimateDampingHilbert(rev_rho, t, plot=False)
        yoshida = fYoshida(fft_res, omeg0_inx)

        DFT1 = fIpDFTd(rev_rho, 1) * samplerate
        # DFT6 = fIpDFTd(rev_rho, 6) * samplerate
        DFT6 = 0

        # print m, yoshida * samplerate, DFT1, DFT6

        # plt.plot(rev_rho)
        # plt.show()

        Amax = np.max(rho_cutted)
        phi = angle_cutted[np.argmax(rho_cutted)] + np.pi / 2
        rho_int = interp1d(freq_cutted, rho_cutted, kind='linear')

        # freq_resampled = np.linspace(freq_cutted[0], freq_cutted[-1], 18000)
        # rho_resampled = rho_int(freq_resampled)
        # freq_arr = freq_resampled[rho_resampled >= (Amax / np.sqrt(2.0))]

        rf = lambda x: rho_int(x) - Amax / np.sqrt(2.0)
        root1 = brenth(rf, freq_cutted[0], omega0)
        root2 = brenth(rf, omega0, freq_cutted[-1])

        #        plt.close()
        #        plt.plot(freq_resampled, rho_resampled)
        #        plt.show()
        #        delta_omega = np.max(freq_arr) - np.min(freq_arr)
        # print root1, root2, root2 - root1, delta_omega

        delta_omega = root2 - root1

        delta = np.pi * delta_omega / omega0
        beta = np.pi * delta_omega
        # t_1 = np.trim_zeros(data).size*dt
        A = - Amax / (np.exp(-m * t[-1]) - 1.0) * (m * t[-1])
        # A = - Amax / (np.exp(-beta * t[-1]) - 1.0) * (beta * t[-1])
        # A = - Amax / (np.exp(-beta * t_1) - 1.0) * (beta * t_1)
        zeta = 1.0 / np.sqrt(1.0 + (2 * np.pi / delta) ** 2)
        if fl:
            res_arr[0, 2] = omega0
            fl = False

        res_arr[k, :] = [i + 1, int(np.round(omega0 / res_arr[0, 2])), omega0, A, phi, beta, delta, Amax, zeta, m,
                         yoshida * samplerate, DFT1, DFT6]
        k += 1
        # print res_arr[i/2, :]
        # out_float += A*np.exp(-beta*t)*np.sin(2*np.pi*omega0*t + phi)
        # print omega0, '\t', A, '\t', delta, '\t', beta, '\t', phi

        # print res_arr[:, 0]
    #    res_arr = deleteSameFreq(res_arr)
    #    res_arr = deleteSameFreq(res_arr)

    if False:  # alternative method of peak finder
        omega0 = res_arr[0, 0]

        omega0_indx = int(omega0 / freq[1])

        for i in range(1, numfreq):
            omega0i = omega0 * i
            omega0_indx = int(omega0i / freq[1])

            dindex = 50
            omega0_indx = np.argmax(rho[omega0_indx - dindex: omega0_indx + dindex]) - dindex + omega0_indx
            Amax = rho[omega0_indx]

            omega_l, omega_r = findParameters(freq[omega0_indx], freq, rho)

            beta = np.pi * (omega_r - omega_l)
            delta = beta / freq[omega0_indx]
            phi = angle[omega0_indx] + np.pi / 2
            A = - Amax / (np.exp(-beta * t[-1]) - 1.0) * (beta * t[-1])

            plt.plot(freq[omega0_indx], Amax, 'rx')
            plt.plot(omega_l, Amax, 'bx')
            plt.plot(omega_r, Amax, 'bx')

            # freq, amp, phase, beta, delta
            res_arr2[i - 1, :] = [freq[omega0_indx], A, phi, beta, delta]
            # print freq[omega0_indx], '\t', A, '\t', phi, '\t', beta, '\t', delta

        # print res_arr
        plt.plot(freq, rho)
        plt.xlim(0, 1000)
        plt.show()

        res_arr = res_arr2

    # res_arr[:, 1] /= res_arr[0, 1]  # normalize amplitude by first harmonic

    plt.ylim(ymin=0)
    plt.xlim(0, 600)
    plt.xlabel('$\omega$, Hz')
    plt.ylabel('A')
    plt.tight_layout()
    # plt.savefig('freq_band.png', dpi=600)
    # plt.savefig('freq_band.eps')

    plt.show()

    if False:
        freq1, rho1, angle1 = FFT(out_float, dt)

        plt.plot(freq, rho)
        plt.plot(freq1, rho1 / np.max(rho1) * np.max(rho))
        plt.show()

        plt.specgram(data / np.max(np.abs(data)) * 0.9, Fs=samplerate, NFFT=2048 * 4, noverlap=0)
        plt.ylim(0, peak_freq[-1])
        plt.savefig('source.png')

        plt.show()

        plt.specgram(out_float, Fs=samplerate, NFFT=2048 * 4, noverlap=0)
        plt.ylim(0, peak_freq[-1])
        plt.savefig('generated.png')
        plt.show()

    return res_arr


def main_test():
    samplerate = 44100
    duration = 5
    dt = 1.0 / samplerate

    t = np.arange(0, duration, dt)

    eps = 2
    omega = 2 * np.pi * 440
    phi = 0 * 0.9 * np.pi
    y = 0.008 * np.exp(-eps * t) * np.sin(omega * t + phi)

    t1 = 1.0 / 440.0 / 4
    y1 = np.exp(-eps * t1) * np.sin(omega * t1)

    t2 = 1.0 / 440.0 + 1.0 / 440.0 / 4
    y2 = np.exp(-eps * t2) * np.sin(omega * t2)

    print np.log(y1 / y2)

    print "delta = ", eps * 2 * np.pi / omega
    print "eps = ", eps
    print 'phi =', phi

    freq, rho, angle = FFT(y, dt)

    plt.plot(t, y)
    plt.show()

    plt.plot(freq, angle, freq, rho)
    plt.xlim(400, 500)
    plt.show()

    omega0 = freq[np.argmax(rho)]
    phi = angle[np.argmax(rho)]
    Amax = np.max(rho)

    print 'Amax = ', Amax
    print 'phi =', phi + np.pi / 2

    rho_int = interp1d(freq, rho, kind='linear')

    freq_resampled = np.arange(omega0 * 0.8, omega0 * 1.2, freq[1] / 100)
    rho_resampled = rho_int(freq_resampled)
    freq_arr = freq_resampled[rho_resampled >= (Amax / np.sqrt(2.0))]

    plt.plot(freq_resampled, rho_resampled)
    plt.show()

    delta_omega = np.max(freq_arr) - np.min(freq_arr)

    beta = np.pi * delta_omega

    delta = beta / omega0

    A = -Amax / (np.exp(-beta * t[-1]) - 1.0) * (beta * t[-1])

    print "omega \tA \t\t\t\tdOmega \t\tdelta \t\t\tbeta"
    print omega0, '\t', A, '\t', delta_omega, '\t', delta, '\t', beta, phi + np.pi / 2

    # -A * (-1 + exp(-beta * t1)) / (beta * t1)
    print -Amax / (np.exp(-beta * t[-1]) - 1.0) * (beta * t[-1])
    plt.plot(freq, rho)
    plt.xlim(0, 12000)
    plt.show()

    return


# main_test()
main()
