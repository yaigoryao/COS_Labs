import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 1. Параметры сигнала
fs = 1000 # Частота дискретизации, Гц
T = 0.2 # Длительность сигнала, с
t = np.linspace(0, T, int(T*fs), endpoint=False) # Вектор времени

# 2. Создаем тестовый сигнал: сумма синусов 5 Гц и 50 Гц + шум
f_clean = 25 # Низкая частота (полезный сигнал)
f_noise = 40 # Высокая частота (помеха)
signal_clean = 1.0 * np.sin(2 * np.pi * f_clean * t)
signal_noise = 0.5 * np.sin(2 * np.pi * f_noise * t)
signal_noisy = signal_clean + signal_noise # Зашумленный сигнал

# 3. СОЗДАЕМ ФИЛЬТР ОДНОЙ КОМАНДОЙ (ФНЧ Баттерворта 4-го порядка)
cutoff_freq = 25 # Частота среза 20 Гц (подавим 50 Гц, оставим 5 Гц)
b, a = signal.butter(4, cutoff_freq, btype='lowpass', fs=fs)

# 4. Применяем фильтр к зашумленному сигналу
filtered_signal = signal.lfilter(b, a, signal_noisy)

# 5. Визуализация
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal_clean, 'b', label='Чистый сигнал (5 Гц)')
plt.plot(t, signal_noisy, 'r', alpha=0.6, label='Зашумленный сигнал (5 Гц + 50 Гц)')
plt.title('Исходные сигналы')
plt.xlabel('Время [с]')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, filtered_signal, 'g', label='Отфильтрованный сигнал')
plt.title('Результат фильтрации (ФНЧ 20 Гц)')
plt.xlabel('Время [с]')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

# 6. Посмотрим на АЧХ нашего фильтра
w, h = signal.freqz(b, a, worN=2000, fs=fs) # freqz - для цифровых фильтров
plt.subplot(3, 1, 3)
plt.plot(w, 20 * np.log10(np.abs(h)), 'm')
plt.axvline(cutoff_freq, color='k', linestyle='--', label=f'f_cut = {cutoff_freq} Гц')
plt.title('АЧХ цифрового ФНЧ Баттерворта')
plt.xlabel('Частота [Гц]')
plt.ylabel('Коэффициент передачи, [дБ]')
plt.xlim(0, 100)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
