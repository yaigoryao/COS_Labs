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
ys_rev = ys[::-1]

yE = (ys + ys_rev) / 2.0
yO = (ys - ys_rev) / 2.0

plt.figure()
plt.plot(xs, ys, "b-o")
plt.plot(xs, yE, "r-")
plt.plot(xs, yO, "g-")
plt.title("Четное и нечетное разложения сигнала")
plt.show()