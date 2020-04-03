import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.linspace(0, 100, num=111, endpoint=True)
    y = 1 - np.exp(-x ** 1.8 / 700)
    f1 = interp1d(x, y, kind='linear')
    f2 = interp1d(x, y, kind='quadratic')
    f3 = interp1d(x, y, kind='cubic')
    f4 = interp1d(x, y, kind='nearest')
    f5 = interp1d(x, y, kind='previous')
    f6 = interp1d(x, y, kind='next')

    x_new = np.linspace(0, 100, num=401, endpoint=True)

    plt.plot(x, y, 'o', x_new, f1(x_new), '-', x_new, f2(x_new), '--', x_new, f3(x_new), ':')
    plt.legend(['data', 'linear', 'quadratic', 'cubic'], loc='best')
    plt.show()

    plt.plot(x, y, 'o')
    plt.plot(x_new, f4(x_new), '-', x_new, f5(x_new), '--', x_new, f6(x_new), ':')
    plt.legend(['data', 'nearest', 'previous', 'next'], loc='best')
    plt.show()
