import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq
import scipy.signal as sgl
import scipy.fft as fft

A1 = 0.5
A2 = 2.0
A3 = 1.7
A4 = 2.5

omega1 = 90
omega2 = 30
omega3 = 30
omega4 = 150

phi1 = 0
phi2 = 30
phi3 = -60
phi4 = 60

func = lambda A, omega, phi: (lambda x: A * np.sin(2 * np.pi * x * omega + phi))

s1 = func(A1, omega1, phi1)
s2 = func(A2, omega2, phi2)
s3 = func(A3, omega3, phi3)
s4 = func(A4, omega4, phi4)

fs = 16000
t = 1.0 / fs

T = 0.1
N = int(fs * T)

x = np.linspace(0, T, N)

y1 = s1(x)
y2 = s2(x)
y3 = s3(x)
y4 = s4(x)

y = (y1 + y2 + y3) * y4

yf = fft.fft(y)
yf_full = np.array(yf)
xf_full = fft.fftfreq(N, t)

yf = (2.0 / N) * np.abs(yf[0:N//2])
xf = xf_full[:N//2]

harmonics_freq_idx = sgl.argrelextrema(yf, np.greater)[0]
harmonics_freq_vals = xf[harmonics_freq_idx]

#freq_delta = 3.0
#Rp = 0.1
#Rs = 40
#n = 5

#clamp = lambda x, mn, mx: int(mn) if x < mn else int(mx) if x > mx else int(x)
#clamp_yf = lambda x: clamp(x, 0, yf_full.size)

#epsilon = 3

def extract_harmonic(freq, bandwidth=3.0):
    mask_pos = ((xf_full >= freq - bandwidth) & (xf_full <= freq + bandwidth))
    mask_neg = ((xf_full >= -freq - bandwidth) & (xf_full <= -freq + bandwidth))
    
    yf_filtered = np.zeros_like(yf_full)
    yf_filtered[mask_pos] = yf_full[mask_pos]
    yf_filtered[mask_neg] = yf_full[mask_neg]
    
    return yf_filtered
#filter_func = lambda freq: sgl.filtfilt(*sgl.ellip(n, Rp, Rs, [freq - freq_delta, freq + freq_delta], 'bandpass', fs=fs), y)
#filter_func = lambda freq_idx: np.pad(yf_cplx[clamp_yf(freq_idx-epsilon):clamp_yf(freq_idx+epsilon)], (clamp_yf(freq_idx-epsilon), yf_cplx.size - clamp_yf(freq_idx+epsilon)), 'constant', constant_values=0)

y1f_filtered = extract_harmonic(harmonics_freq_vals[0])
y2f_filtered = extract_harmonic(harmonics_freq_vals[1])
y3f_filtered = extract_harmonic(harmonics_freq_vals[2])
y4f_filtered = extract_harmonic(harmonics_freq_vals[3])

y1_filtered = fft.ifft(y1f_filtered)
y2_filtered = fft.ifft(y2f_filtered)
y3_filtered = fft.ifft(y3f_filtered)
y4_filtered = fft.ifft(y4f_filtered)

# Исходный сигнал
plt.figure()
plt.plot(x, y, label='Исходный сигнал')
plt.grid(True, alpha=0.5)
plt.xlabel('Время, с')
plt.ylabel('Значение функции')
plt.title('График исходного сигнала')
plt.legend()
plt.show()

mx_display_freq = 400

# Спектр основного сигнала

xf = xf[xf < mx_display_freq]
yf = yf[:xf.size]

plt.figure()
plt.plot(xf, yf, label='Спектр исходного сигнала')
plt.grid(True, alpha=0.5)
plt.xlabel('Частота, гц')
plt.ylabel('Коэффициент передачи')
for harmonic in harmonics_freq_vals:
    plt.axvline(harmonic, alpha=0.5, color='r', linestyle='--', label=f'{harmonic} гц')
plt.legend()
plt.title('Спектр исходного сигнала')
plt.show()

# Спектры

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(xf, 2.0 / N * np.abs(y1f_filtered[:xf.size]), label=f'{harmonics_freq_vals[0]} гц')
plt.axvline(harmonics_freq_vals[0], alpha=0.5, color='r', linestyle='--')
plt.grid(True, alpha=0.5)
plt.xlabel('гц')
plt.ylabel('Коэффициент передачи')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(xf, 2.0 / N * np.abs(y2f_filtered[:xf.size]), label=f'{harmonics_freq_vals[1]} гц')
plt.axvline(harmonics_freq_vals[1], alpha=0.5, color='r', linestyle='--')
plt.grid(True, alpha=0.5)
plt.xlabel('гц')
plt.ylabel('Коэффициент передачи')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(xf, 2.0 / N * np.abs(y3f_filtered[:xf.size]), label=f'{harmonics_freq_vals[2]} гц')
plt.axvline(harmonics_freq_vals[2], alpha=0.5, color='r',linestyle='--')
plt.grid(True, alpha=0.5)
plt.xlabel('гц')
plt.ylabel('Коэффициент передачи')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(xf, 2.0 / N * np.abs(y4f_filtered[:xf.size]), label=f'{harmonics_freq_vals[3]} гц')
plt.axvline(harmonics_freq_vals[3], alpha=0.5, color='r', linestyle='--')
plt.grid(True, alpha=0.5)
plt.xlabel('гц')
plt.ylabel('Коэффициент передачи')
plt.legend()

plt.suptitle('Графики спектров отфильтрованных сигналов')
plt.show()

# Графики отфильтрованных сигналов

# def get_signal_phase(y_filtered, freq, fs):
#     yf = fft.fft(y_filtered)
#     freqs = fft.fftfreq(len(y_filtered), 1/fs)
    
#     idx = np.argmin(np.abs(freqs - freq))
    
#     phase = np.angle(yf[idx])
#     return phase

# phase1 = get_signal_phase(y1_filtered, harmonics_freq_vals[0], fs)
# phase2 = get_signal_phase(y2_filtered, harmonics_freq_vals[1], fs)
# phase3 = get_signal_phase(y3_filtered, harmonics_freq_vals[2], fs)
# phase4 = get_signal_phase(y4_filtered, harmonics_freq_vals[3], fs)


plt.figure()
plt.subplot(2, 2, 1)
plt.plot(x[:y1_filtered.size], y1_filtered, label=f'{harmonics_freq_vals[0]} гц')
plt.plot(x[:y1_filtered.size], np.sin(2 * np.pi * harmonics_freq_vals[0] * x[:y1_filtered.size]), alpha = 0.5, color='r', linestyle='--', label=f'Ожидаемый сигнал')
plt.grid(True, alpha=0.5)
plt.xlabel('Время, с')
plt.ylabel('Значение функции')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x[:y2_filtered.size], y2_filtered, label=f'{harmonics_freq_vals[1]} гц')
plt.plot(x[:y2_filtered.size], np.sin(2 * np.pi * harmonics_freq_vals[1] * x[:y1_filtered.size]), alpha = 0.5, color='r', linestyle='--', label=f'Ожидаемый сигнал')
plt.grid(True, alpha=0.5)
plt.xlabel('Время, с')
plt.ylabel('Значение функции')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x[:y3_filtered.size], y3_filtered, label=f'{harmonics_freq_vals[2]} гц')
plt.plot(x[:y3_filtered.size], np.sin(2 * np.pi * harmonics_freq_vals[2] * x[:y1_filtered.size]), alpha = 0.5, color='r', linestyle='--', label=f'Ожидаемый сигнал')
plt.grid(True, alpha=0.5)
plt.xlabel('Время, с')
plt.ylabel('Значение функции')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x[:y4_filtered.size], y4_filtered, label=f'{harmonics_freq_vals[3]} гц')
plt.plot(x[:y4_filtered.size], np.sin(2 * np.pi * harmonics_freq_vals[3] * x[:y1_filtered.size]), alpha = 0.5, color='r', linestyle='--', label=f'Ожидаемый сигнал')
plt.grid(True, alpha=0.5)
plt.xlabel('Время, с')
plt.ylabel('Значение функции')
plt.legend()

plt.suptitle('Графики отфильтрованных сигналов (без коррекции амплитуд и фаз)')
plt.show()