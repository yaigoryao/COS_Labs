import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

n = 2
Rp = 4
Rs = 10


z, p, k = signal.but
#z, p, k = signal.cheb1ap(n, Rp)
#z, p, k = signal.cheb1ap(n, Rp)
#z, p, k = signal.cheb1ap(n, Rp)

b, a = signal.zpk2tf(z, p, k) 

f1 = 1e3 
f2 = 9e3 
w0 = 2 * np.pi * np.sqrt(f1 * f2) 
Bw = 2 * np.pi * (f2 - f1)

b_bp, a_bp = signal.lp2bp(b, a, w0, Bw)

f = np.arange(0, 20001, 1)
w = 2 * np.pi * f

w, h = signal.freqs(b_bp, a_bp, w)

plt.figure(figsize=(10, 6))
plt.plot(f/1000, np.abs(h))
plt.title('АЧХ полосового фильтра Чебышева')
plt.xlabel('Частота, кГц')
plt.ylabel('Коэффициент передачи')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlim(0, 20)
plt.show()