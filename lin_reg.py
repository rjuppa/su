
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

# matplotlib.use('TkAgg')   # use for OSX


class Theta(object):
    c0 = 0.0
    c1 = 0.0

    def update(self, c0, c1):
        self.c0 = c0
        self.c1 = c1

    def get(self):
        return self.c0, self.c1


class LinRegression(object):

    def __init__(self):
        self.alpha = 0.1
        self.data = []
        self.xx = []
        self.yy = []
        self.theta = Theta()
        self.errors = []
        self.thetas = []
        self.iteration = 1

    def load_data(self):
        with open("data/data03.txt", "r") as f:
            lines = f.readlines()

        data = []
        for line in lines:
            sx, sy = line.split(",")
            data.append((float(sx), float(sy)))

        self.xx = [x[0] for x in data]
        self.yy = [y[1] for y in data]
        self.count = len(data)
        self.data = data

    def draw_data(self):
        plt.figure(figsize=(15, 8))
        plt.title("Data")
        plt.xlabel('Population of City (in 10 000s)')
        plt.ylabel('Profit ($10.000)')
        plt.grid(False)
        plt.axis([min(self.xx)-1, max(self.xx) + 1, min(self.yy)-1, max(self.yy) + 1])
        for x, y in self.data:
            plt.plot(x, y, marker='x', color='r')

    def draw_hypoteza(self):
        plt.title("Regression")
        plt.xlabel('Population of City (in 10 000s)')
        plt.ylabel('Profit ($10.000)')
        plt.plot([min(self.xx), max(self.xx)], [0, 0], color="g", label='Initial')

        x0 = min(self.xx)
        y0 = self.theta.c0 + min(self.xx) * self.theta.c1
        x1 = max(self.xx)
        y1 = self.theta.c0 + max(self.xx) * self.theta.c1
        plt.plot([x0, x1], [y0, y1], color="r", label="Final")

    def draw_theta(self):
        plt.figure(figsize=(15, 8))
        plt.grid(True)
        plt.title("Theta")
        plt.xlabel('C 1')
        plt.ylabel('C 0')
        cc0 = [c[0] for c in self.thetas]
        cc1 = [c[1] for c in self.thetas]
        plt.plot(cc0, cc1, mew=1, ms=1)

    def draw_error(self):
        plt.figure(figsize=(15, 8))
        plt.grid(True)
        plt.title("Error")
        plt.xlabel('Iterace')
        plt.ylabel('Chyba')
        plt.plot(range(self.iteration), self.errors, mew=1, ms=1)

    def predict_y(self, x, c0, c1):
        return c0 + x * c1

    def compute_error(self, c0, c1):
        sum_sq = 0
        for x, y in self.data:
            sum_sq += (self.predict_y(x, c0, c1) - y)**2

        return sum_sq/float(self.count*2)

    def compute_grad(self):
        c0_grad = 0
        c1_grad = 0
        for x, y in self.data:
            c0_grad -= self.predict_y(x, self.theta.c0, self.theta.c1) - y
            c1_grad -= x * (self.predict_y(x, self.theta.c0, self.theta.c1) - y)

        c0_grad = c0_grad / float(self.count)
        c1_grad = c1_grad / float(self.count)
        size = math.sqrt(c0_grad ** 2 + c1_grad ** 2)
        return c0_grad/size, c1_grad/size   # normalized

    def print_row(self, err):
        i = self.iteration
        a = self.alpha
        c0 = self.theta.c0
        c1 = self.theta.c1
        print(f"I:{i} alpha: {a:0.4f}  C0:{c0:0.5f}, C1:{c1:0.5f}, Error: {err:0.5f}")

    def find_minimum(self, alpha=0.2, is_print=False):
        """
        using Gradient Descent
        """
        c0 = 0
        c1 = 0
        self.alpha = alpha
        err = self.compute_error(0, 0)
        self.errors.append(err)
        self.thetas.append((0, 0))
        if is_print:
            self.print_row(err)

        for i in range(2000):
            dx, dy = self.compute_grad()
            c0 += dx*alpha
            c1 += dy*alpha
            err = self.compute_error(c0, c1)

            self.theta.update(c0, c1)
            self.errors.append(err)
            self.thetas.append((c0, c1))
            self.iteration += 1
            if is_print:
                self.print_row(err)
            if self.errors[-2] < self.errors[-1]:
                self.alpha = self.alpha * 0.8
                print("-----")

            if abs(self.errors[-2] - self.errors[-1]) < 0.0001:
                # STOP
                break


if __name__ == "__main__":

    alpha = 0.2

    reg = LinRegression()
    reg.load_data()
    reg.draw_data()

    reg.find_minimum(alpha=alpha, is_print=True)
    print("====================================================")
    print(f"Regresni primka: y = {reg.theta.c1:0.2f}x + {reg.theta.c0:0.2f}")
    print("====================================================")

    reg.draw_hypoteza()
    reg.draw_theta()
    reg.draw_error()
    plt.legend()
    plt.show()





