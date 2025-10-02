import numpy as np
import scipy.signal as sgl
import matplotlib.pyplot as plt
n = 2
Rp = 4
Rs = 10
f1 = 1e3

w0 = 2 * np.pi * f1

cheb1_b, cheb1_a = sgl.lp2lp(*sgl.zpk2tf(*sgl.cheb1ap(n, Rp)), w0)

f = np.arange(0, 2e4, 1)
f = 2 * np.pi * f
cheb1_w, cheb1_h = sgl.freqs(cheb1_b, cheb1_a, f)

plt.figure()
plt.plot(f/1e3, np.abs(cheb1_h), alpha=0.6, label="Фильтр Чебышева 1 рода")
plt.title('АЧХ фильтра чебышева 1 рода')
plt.xlabel('Частота, кГц')
plt.ylabel('Коэффициент передачи')
plt.grid(True, which='both', alpha=0.6)
plt.legend()
plt.xscale('log')
plt.show()