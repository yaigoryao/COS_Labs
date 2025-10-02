import numpy as np
from matplotlib import pyplot as plt
signal_name = "прямоугольного сигнала"
graph_rows = 3
graph_cols = 6

A = 3
T = 6
tau = 5
half_tau = tau/2
k = 18

num_periods = 3
t = np.arange(-num_periods * T/2, num_periods * T/2, 0.01)
q = T/tau

harmonics = 2*A/np.pi * np.array([(1.0/n) * np.sin(np.pi * n / q) * np.cos(2.0 * np.pi * n * t / T) for n in range(1, k + 1)])
superpositions = A/q + np.cumsum(harmonics, axis = 0)

plt.figure()
for i, h in enumerate(harmonics):
    plt.subplot(graph_rows, graph_cols, i + 1)
    plt.plot(t, harmonics[i])
    plt.ylim(-A, A)
    plt.grid(True)
    plt.title(f"Гармоника {i + 1}")

plt.suptitle(f"Гармоники {signal_name}")
plt.show()

plt.figure()
for i, h in enumerate(superpositions):
    plt.subplot(graph_rows, graph_cols, i + 1)
    plt.plot(t, superpositions[i])
    plt.ylim(0, A + 1)
    plt.grid(True)
    plt.title(f"Суперпозиция {i + 1} гармоник")

plt.suptitle(f"Суперпозиции гармоник {signal_name}")
plt.show()