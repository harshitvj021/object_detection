import numpy as np

def range_kutta_4(f, t0, y0, t_final, h):
    t = np.arange(t0, t_final+h, h)
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0, :] = y0
    for i in range(1, n):
        k1 = h * f(t[i-1], y[i-1, :])
        k2 = h * f(t[i-1] + h/2, y[i-1, :] + k1/2)
        k3 = h * f(t[i-1] + h/2, y[i-1, :] + k2/2)
        k4 = h * f(t[i-1] + h, y[i-1, :] + k3)
        y[i, :] = y[i-1, :] + (k1 + 2*k2 + 2*k3 + k4)/6
    return t, y
