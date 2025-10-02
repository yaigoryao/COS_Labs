import numpy as np
import matplotlib.pyplot as plt

plot_cols = 6
plot_rows = 4
epsilon = 1.0

A = 9.0
c = 6.0
k = 22

tau = 1

xs = np.linspace(-tau, tau, k)
ys = A * np.sin(c * xs)

yE = [0.0] * k
yO = [0.0] * k
for i, n in enumerate(ys):
    if i % 2 == 0:
        yE[i] = ys[i]
    else:
        yO[i] = ys[i]

yE = np.array(yE)
yO = np.array(yO)

plt.figure()
plt.plot(xs, ys, "b-o")
plt.plot(xs, yE, "r-o")
plt.plot(xs, yO, "g-o")
plt.title("Чередующееся разложение сигнала")
plt.show()