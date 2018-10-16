import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

# matplotlib.use('TkAgg')   # use for OSX


def predict_y(x, c0, c1):
    return c0 + x * c1


def draw_line(c0, c1):
    plt.grid(True)
    x0 = 0
    y0 = predict_y(x0, c0, c1)
    x1 = 4
    y1 = predict_y(x1, c0, c1)
    plt.plot([x0, x1], [y0, y1], color="r", label="Final")


if __name__ == "__main__":

    X = [1, 1, 2, 3]
    Y = [2, 3, 1, 4]
    data = zip(X, Y)

    plt.figure(figsize=(15, 8))

    plt.grid(True)
    plt.axis([0, 5, 0, 5])

    for x, y in data:
        plt.plot(x, y, marker='x', color='r')

    sum_x = sum(X)                          # 7
    sum_x2 = map(lambda x: x**2, X)         # 15
    sum_y = sum(Y)                          # 10
    sum_xy = sum([x * y for x, y in data])  # 19
    n = len(X)                              # 4

    # c1*sum_x2 + c0*sum_x = sum_xy
    # c1*sum_x  + c0*n     = sum_y

    # 15c1 + 7c0 = 19
    # 7c1  + 4c0 = 10
    # ---------------

    # c0 = 1.54
    # c1 = 0,55

    a = np.array([[15, 7], [7, 4]])
    b = np.array([19, 10])
    c = np.linalg.solve(a, b)
    print(f"c1={c[1]}  c0={c[0]}")
    print(np.allclose(np.dot(a, c), b))

    draw_line(c[1], c[0])

    plt.legend()
    plt.show()
