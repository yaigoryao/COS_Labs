import numpy as np
import scipy.signal as sgl
import matplotlib.pyplot as plt
n = 2
Rp = 4
Rs = 10
f1 = 1e3
f2 = 1e4

w0 = 2 * np.pi * np.sqrt(f1 * f2)
Bw = 2 * np.pi * (f2 - f1)
cheb2_b, cheb2_a = sgl.lp2bp(*sgl.zpk2tf(*sgl.cheb2ap(n, Rs)), w0, Bw)

f = np.arange(0, 2e4, 1)
f = 2 * np.pi * f
cheb2_w, cheb2_h = sgl.freqs(cheb2_b, cheb2_a, f)

plt.figure()
plt.plot(f/1e3, np.abs(cheb2_h), alpha=0.6, label="Фильтр Чебышева 2 рода")
plt.title('АЧХ фильтра чебышева 2 рода')
plt.xlabel('Частота, кГц')
plt.ylabel('Коэффициент передачи')
plt.grid(True, which='both', alpha=0.6)
plt.legend()
plt.xscale('log')
plt.show()