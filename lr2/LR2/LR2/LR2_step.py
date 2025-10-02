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
ys_shifted = np.append(np.array([0.0]), ys)
ys_shifted = ys_shifted[:-1]

steps = ys - ys_shifted

steps_ys = np.array( [np.append( np.array([0.0] * i), np.array([steps[i]] * (k - i) ) ) for i in range(k)] )

plt.figure()
plt.plot(xs, ys, "b-o")
plt.title("Исходный сигнал")
plt.show()


plt.figure()

for i in range(k):
    n = i + 1
    plt.subplot(plot_rows, plot_cols, n)
    plt.plot(xs, ys, "b-o")
    plt.plot(xs, steps_ys[i], "r-")
    plt.title(f"Импульс {n}")
    plt.ylim(-epsilon - A, A + epsilon)

plt.suptitle("Ступенчатое кодирование сигнала")

plt.show()