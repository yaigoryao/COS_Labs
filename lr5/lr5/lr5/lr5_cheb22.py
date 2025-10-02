import numpy as np
import scipy.signal as sgl
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

#butter,   cheb1,         cheb2,        ellip
#ФВЧ - s2, РФ - s1,       ФНЧ - s1,     ПФ - s2  //s1+s2
#ПФ - s2,  ФНЧ - s1 + s2, РФ - s1 + s3, ФВЧ - s3 // s1+s2+s3

fs = 1000
T = 0.25
N = int (fs * T)

s1 = 25
s2 = 40
s3 = 60

delta = 7
n = 5
Rp = 0.1
Rs = 40

x = np.linspace(0, T, int(T*fs)) 

signal1 = np.sin(2 * np.pi * s1 * x)
signal2 = np.sin(2 * np.pi * s2 * x)
signal3 = np.sin(2 * np.pi * s3 * x)

noisy = signal1 + signal2 + signal3

b, a = sgl.cheby2(n, Rs, [s2 - delta, s2 + delta], 'bandstop', fs=fs)
filtered = sgl.filtfilt(b, a, noisy)

xf = fftfreq(N, 1.0 / fs)[:N//4]
noisyf = fft(noisy)
filteredf = fft(filtered)

plt.figure()

plt.subplot(2, 2, 1)
plt.plot(x, noisy, label=f'Зашумленный сигнал')
plt.grid(True, alpha=0.5)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, filtered, label=f'Отфильтрованный сигнал')
plt.grid(True, alpha=0.5)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(xf, 2.0/N * np.abs(noisyf[0:N//4]), label=f'Спектр зашумленного сигнала')
plt.axvline(x=s1, linestyle='--', linewidth=1, label=f'{s1} Гц')
plt.axvline(x=s2, linestyle='--', linewidth=1, label=f'{s2} Гц')
plt.axvline(x=s3, linestyle='--', linewidth=1, label=f'{s3} Гц')
plt.grid(True, alpha=0.5)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(xf, 2.0/N * np.abs(filteredf[0:N//4]), label=f'Спектр отфильтрованного сигнала')
plt.axvline(x=s1, linestyle='--', linewidth=1, label=f'{s1} Гц')
plt.axvline(x=s2, linestyle='--', linewidth=1, label=f'{s2} Гц')
plt.axvline(x=s3, linestyle='--', linewidth=1, label=f'{s3} Гц')
plt.grid(True, alpha=0.5)
plt.legend()
plt.show()