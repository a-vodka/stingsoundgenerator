import numpy as np
# import sympy as sp
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


def cmpwav(filename1, filename2):
    samplerate1, data1 = wavfile.read(filename1)
    samplerate2, data2 = wavfile.read(filename2)
    cmpwav(data1, samplerate1, data2, samplerate2)


def cmpdata(data1, samplerate1, data2, samplerate2, picname='default_cmp'):
    # filename1 = '0vel.wav'
    # filename1 = 'test.wav'
    # filename2 = './narezannye_zvuki/string_3_fret_0.wav'
    # filename2 = './cort1/a_0.wav'

    data1 = np.array(data1, dtype=float)
    data2 = np.array(data2, dtype=float)

    dt1 = 1.0 / samplerate1
    dt2 = 1.0 / samplerate2

    data1 /= np.max(np.abs(data1))
    data2 /= np.max(np.abs(data2))
    t1arg = np.argmax(np.abs(data1))
    t2arg = np.argmax(np.abs(data2))

    if data1[t1arg] * data2[t2arg] < 0:
        data1 *= -1.0

    time_shift1 = t1arg * dt1
    time_shift2 = t2arg * dt2
    arg_shift2 = np.argmax(np.abs(data2))
    t1 = np.linspace(0, data1.size * dt1, data1.size) - time_shift1
    t2 = np.linspace(0, data2.size * dt2, data2.size) - time_shift2

    time = 0.8 * np.min([np.max(t1), np.max(t2)])
    print time, np.max(t1), np.max(t2)

    ddata2 = np.gradient(data2, dt2)
    ddata1 = np.gradient(data1, dt1)

    # ddata1 /= np.max(np.abs(ddata1))
    # ddata2 /= np.max(np.abs(ddata2))

    nn = int(0.3 * samplerate1)  # 0.3 sec
    plt.figure(figsize=(11.7, 8.3))

    ax = plt.subplot2grid((4, 2), (2, 1), rowspan=2)

    # ax1.subplot(224)
    plt.plot(data1[0:nn], ddata1[0:nn], label='generated')
    plt.plot(data2[arg_shift2: arg_shift2 + nn], ddata2[arg_shift2: arg_shift2 + nn], '--', label='source')
    plt.xlabel('$u(t)$')
    plt.ylabel('$\dot{u}(t)$')
    plt.text(0.5, 0.95, 'Phase trajectories', transform=ax.transAxes, horizontalalignment='center')
    plt.legend()

    ax = plt.subplot(411)

    plt.plot(t1, data1, label='generated')
    plt.plot(t2, data2, '--', label='source')
    plt.xlabel('t, s')
    plt.ylabel('u(t)')
    plt.xlim(0, 0.11)
    plt.legend()
    plt.text(0.5, 0.9, 'Signal', transform=ax.transAxes, horizontalalignment='center')
    # plt.show()

    ax = plt.subplot(425)
    plt.specgram(data1, Fs=samplerate1, NFFT=2048 * 6, pad_to=2 ** 15)
    plt.xlim(0, time)
    plt.ylim(0, 500)
    plt.xlabel('t, s')
    plt.ylabel('$\omega$, Hz')
    plt.text(0.5, 0.9, 'Spectrogram of generated signal', color='white', transform=ax.transAxes,
             horizontalalignment='center')

    ax = plt.subplot(427)
    plt.specgram(data2[arg_shift2:], Fs=samplerate2, NFFT=2048 * 6, pad_to=2 ** 15)
    plt.ylim(0, 500)
    plt.xlim(0, time)
    plt.xlabel('t, s')
    plt.ylabel('$\omega$, Hz')
    plt.text(0.5, 0.9, 'Spectrogram of source signal', color='white', transform=ax.transAxes,
             horizontalalignment='center')

    # plt.show()
    ax = plt.subplot(412)
    # ax = plt.gca()
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes

    fr1, rho1, angle1 = FFT(data1, dt1)
    fr2, rho2, angle2 = FFT(data2, dt1)

    plt.semilogy(fr1, rho1 / rho1.max(), label='generated')
    plt.semilogy(fr2, rho2 / rho2.max(), '--', label='source')
    plt.xlim(0, 800)
    plt.ylim(ymin=1e-4)

    plt.xlabel('$\omega$, Hz')
    plt.ylabel('U')
    plt.text(0.5, 0.9, 'Spectrum', transform=ax.transAxes, horizontalalignment='center')
    plt.legend()
    # axins = inset_axes(ax, width="30%",  # width = 30% of parent_bbox
    #                   height=1.,  # height : 1 inch
    #                   loc=1)

    # fmax = fr1[np.argmax(rho1)]
    # lf = 0.98 * fmax
    # rf = 1.02 * fmax
    # axins.set_xlim(lf, rf)
    # axins.plot(fr1, rho1 / rho1.max())
    # axins.plot(fr2, rho2 / rho2.max(), '--')

    plt.tight_layout(h_pad=0.32)
    plt.savefig(picname + '.png', dpi=300)
    plt.savefig(picname+'.eps')
    plt.show()
