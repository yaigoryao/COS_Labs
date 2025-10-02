import numpy as np
import scipy.signal as sgl
import matplotlib.pyplot as plt
n = 2
Rp = 4
Rs = 10
f1 = 1000

w0 = 2 * np.pi * f1

butt_b, butt_a = sgl.lp2lp(*sgl.zpk2tf(*sgl.buttap(n)), w0)
cheb1_b, cheb1_a = sgl.lp2lp(*sgl.zpk2tf(*sgl.cheb1ap(n, Rp)), w0)
cheb2_b, cheb2_a = sgl.lp2lp(*sgl.zpk2tf(*sgl.cheb2ap(n, Rs)), w0)
ell_b, ell_a = sgl.lp2lp(*sgl.zpk2tf(*sgl.ellipap(n, Rp, Rs)), w0)
bess_b, bess_a = sgl.lp2lp(*sgl.zpk2tf(*sgl.besselap(n)), w0)

f = np.arange(0, 10000, 1)
f = 2 * np.pi * f
butt_w, butt_h = sgl.freqs(butt_b, butt_a, f)
cheb1_w, cheb1_h = sgl.freqs(cheb1_b, cheb1_a, f)
cheb2_w, cheb2_h = sgl.freqs(cheb2_b, cheb2_a, f)
ell_w, ell_h = sgl.freqs(ell_b, ell_a, f)
bess_w, bess_h = sgl.freqs(bess_b, bess_a, f)

plt.figure()
plt.plot(f/1000, np.abs(butt_h), alpha=0.5, label="Фильтр Баттерводра")
plt.plot(f/1000, np.abs(cheb1_h), alpha=0.5, label="Фильтр Чебышева 1 рода")
plt.plot(f/1000, np.abs(cheb2_h), alpha=0.5, label="Фильтр Чебышева 2 рода")
plt.plot(f/1000, np.abs(ell_h), alpha=0.5, label="Фильтр эллиптический")
plt.plot(f/1000, np.abs(bess_h), alpha=0.5, label="Фильтр Бесселя")
plt.title('АЧХ фильтров')
plt.xlabel('Частота, кгц')
plt.ylabel('Коэффициент передачи')
plt.grid(True, alpha=0.6)
plt.legend()
plt.show()