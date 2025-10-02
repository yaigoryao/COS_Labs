import numpy as np
import scipy.signal as sgl
import matplotlib.pyplot as plt

fs = 1000
T = 0.1

s1 = 25
s2 = 40

x = np.linspace(0, T, int(T*fs)) 

signal1 = np.sin(2.0 * np.pi * s1 * x)
signal2 = np.sin(2.0 * np.pi * s2 * x)

noisy1 = signal1 + signal2

b, a = sgl.butter(4, 30, 'lowpass', fs=fs)

w, h = sgl.freqz(b, a, fs=fs)

filtered = sgl.filtfilt(b, a, noisy1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(x, signal1, alpha=0.8, label=f'sin(2*pi*{s1}*x)')
plt.plot(x, signal2, alpha=0.6, label=f'sin(2*pi*{s2}*x)')
plt.plot(x, noisy1, alpha=0.6, label=f'sin(2*pi*{s1}*x) + sin(2*pi*{s2}*x)')
plt.plot(x, filtered, linewidth=2, label='Отфильтрованный сигнал')
plt.legend()
plt.grid(True, alpha=0.5)
plt.title('Сигналы')
#plt.xlabel('Время (с)')
#plt.ylabel('Амплитуда')

plt.subplot(2, 1, 2)
plt.plot(w, 20 * np.log10(np.abs(h)), 'b')
plt.legend()
plt.grid(True, alpha=0.5)
plt.title('Амплитудно-частотная характеристика фильтра')
plt.xlabel('Частота (Гц)')
#plt.ylabel('Амплитуда (дБ)')
#plt.xlim(0, 60)

plt.tight_layout()
plt.show()