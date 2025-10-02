import numpy as np
import scipy.signal as sgl
import matplotlib.pyplot as plt

#butter,   cheb1,         cheb2,        ellip
#ФВЧ - s2, РФ - s1,       ФНЧ - s1,     ПФ - s2  //s1+s2
#ПФ - s2,  ФНЧ - s1 + s2, РФ - s1 + s3, ФВЧ - s3 // s1+s2+s3

fs = 1000
T = 0.1

s1 = 25
s2 = 40
s3 = 60

delta = 10
n = 5
Rp = 0.1
Rs = 40

x = np.linspace(0, T, int(T*fs)) 

signal1 = np.sin(2 * np.pi * s1 * x)
signal2 = np.sin(2 * np.pi * s2 * x)
signal3 = np.sin(2 * np.pi * s3 * x)

noisy = signal1 + signal2 + signal3

b, a = sgl.cheby1(n, Rp, s3 - delta, 'lowpass', fs=fs)
filtered = sgl.filtfilt(b, a, noisy)

w, h = sgl.freqz(b, a, fs=fs)
h = np.abs(h)

plt.figure()

plt.subplot(2, 1, 1)
#plt.plot(x, signal1, '-', alpha=0.5, label=f'sin({s1}x)')
#plt.plot(x, signal2, '-', alpha=0.5, label=f'sin({s2}x)')
plt.plot(x, signal1 + signal2, '-', alpha=0.5, label=f'sin(2*pi*{s1}x) + sin(2*pi*{s2}x)')
plt.plot(x, noisy, '-', alpha=0.3, label=f'sin(2*pi*{s1}x) + sin(2*pi*{s2}x) + sin(2*pi*{s3})')
plt.plot(x, filtered, '--', alpha=0.9, label=f'Отфильтрованный сигнал')
plt.grid(True, alpha=0.5)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(w, 20 * np.log10(np.abs(h)), alpha=0.7, label=f'АЧХ фильтра')
plt.axvline(x=s3, linestyle='--', linewidth=1, label=f'{s3} Гц')
plt.legend()
plt.grid(True, alpha=0.5)
plt.xlim(0, 2 * s2)

plt.show()