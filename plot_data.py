import numpy as np
from matplotlib import pyplot as plt
from matplotlib.mlab import PCA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import scipy
import scipy.fftpack
import pylab
import math

from numpy import fft
from scipy.interpolate import interp1d


class Arrow3D(FancyArrowPatch):
    '''3D arrow for plotting'''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_whole_data(data):
    t, c, x, y, z = zip(*data)
    pca = PCA(np.array(zip(x, y, z)))

    fig = plt.figure()
    # Because mplot3d and leapmotion use different axes, we switch Z and Y here for a more natural look

    # 3d plot
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    plt.title("Tracked position in 3d space")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_zlabel('Y axis')
    ax.set_aspect('equal')
    ax.scatter(x, z, y)

    # Center
    mu_x, mu_y, mu_z = pca.mu
    ax.plot([mu_x], [mu_z], [mu_y], 'o', markersize=10, color='yellow', alpha=0.5)

    # Principal vector
    v = pca.Wt[0]
    pv_x, pv_y, pv_z = v * pca.sigma + pca.mu
    ax.plot([pv_x], [pv_z], [pv_y], 'o', markersize=10, color='green', alpha=0.5)
    a = Arrow3D([mu_x, pv_x], [mu_z, pv_z], [mu_y, pv_y], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
    ax.grid()
    ax.axis('equal')

    # 2d plot
    ax = fig.add_subplot(2, 2, 3)
    plt.title("Tracked position in 2d space")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.grid()
    ax.arrow(mu_x, mu_y, pv_x - mu_x, pv_y - mu_y, width=2, head_width=5, head_length=5, fc='k', ec='k')
    ax.plot(x, y)
    ax.axis('equal')

    # x, y, z over time
    ax = fig.add_subplot(222)
    plt.title("Position over time")
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Millimeters')
    ax.plot(t, x, label='x')
    ax.plot(t, y, label='y')
    ax.plot(t, z, label='z')
    legend = plt.legend()
    ax.add_artist(legend)

    plt.show()


def plot_windows(data):
    windows = sliding_window(data, offset=32)
    t, c, x, y, z = zip(*data)
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    ax2.plot(x, y)
    for i, window in enumerate(windows):
        t, c, x, y, z = zip(*window)
        pca = PCA(np.array(zip(x, y, z)))
        # Center
        mu_x, mu_y, mu_z = pca.mu
        ax1.plot([mu_x], [mu_z], [mu_y], 'o', markersize=10, color='yellow', alpha=0.5)

        # Principal vector
        v = pca.Wt[0]
        pv_x, pv_y, pv_z = v * pca.sigma + pca.mu
        ax1.plot([pv_x], [pv_z], [pv_y], 'o', markersize=10, color='green', alpha=0.5)
        color = str((i + 1.0) / (len(windows) + 1.0))
        a = Arrow3D([mu_x, pv_x], [mu_z, pv_z], [mu_y, pv_y], mutation_scale=20, lw=3, arrowstyle="-|>", color=color)
        ax1.add_artist(a)
        ax2.arrow(mu_x, mu_y, pv_x - mu_x, pv_y - mu_y, width=1, head_width=3, head_length=4, color=color)

    plt.show()


def sliding_window(data, n=128, offset=64):
    '''Takes a list of data and returns the data divided into windows.'''

    assert len(data) > n
    assert offset < n

    # snip off data that doesn't fall neatly in multiples of n
    remainder = len(data) % n
    data = data[:-remainder]
    print len(data)
    windows = []
    i = 0
    while i < len(data):
        windows.append(data[i:i+n])
        i = i + offset
    print len(windows)

    return windows


def plot_fourier(data):
    t, c, x, y, z = zip(*data)
    t = list(t)
    # time data starts from 0 and is set to seconds
    t[:] = [(s - t[0]) for s in t]
    tf, yf = fourier_transform(t, y)

    print(tf.min(), tf.max())

    print("total time: %f" % (t[-1]))
    print("hertz: %f" % get_hertz(t, y, tf, yf))

    # plot the fft with reasonable amount of hertz(1+ Hz); x axis: hertz
    ti, yi = interpolate(t, y)

    print("length of y: %d" % len(y))

    tif, yif = fourier_transform(ti, yi)

    print("total time: %f" % (ti[-1]))
    print("hertz: %f" % get_hertz(t, y, tif, yif))

    plt.grid()
    plt.figure(2)
    plt.subplot(211)
    plt.plot(ti, yi)
    plt.xlabel("time(seconds)")
    plt.ylabel("y-value(handpalm)")
    plt.grid()

    # plot the fft with reasonable amount of hertz(1+ Hz); x axis: hertz
    plt.subplot(212)
    plt.plot(tif * 100, np.abs(yif))
    plt.xlabel('Hertz')
    plt.grid()
    plt.show()


def fourier_transform(x, y):
    # fourier transform on the data
    yf = fft.fft(y)
    # frequency of the data
    xf = fft.fftfreq(len(yf))
    # show frequency range 1Hz to 10Hz
    mask = (xf > 0.01) & (xf <= .10)
    xf = xf[mask]
    yf = yf[mask]
    return xf, yf


def get_hertz(x, y, xf, yf):
    ayf = np.abs(yf)**2
    # find the max argument of ayf except the first one
    idx = np.argmax(ayf)
    # find the frequency in the processed timestamps
    freq = xf[idx]
    print("freq: %f" % freq)
    # framerate is the total amount of frames divided by the total time in seconds
    frate = len(y) / (x[-1])
    print("frate: %f" % frate)
    hertz = abs(freq * frate)
    return hertz


def interpolate(t, y, fps=100):
    '''Interpolate on 100 frames'''
    yi = interp1d(t, y)
    yi2 = interp1d(t, y, kind='cubic')

    # total datapoints at 100 frames/s (rounded up)
    points = int(math.ceil(t[-1] * fps))

    ti = np.linspace(t[0], t[-1], points)

    plt.figure(1)
    plt.plot(t, y, 'o', ti, yi(ti), '-', ti, yi2(ti), '--')
    plt.xlabel("time(seconds)")
    plt.ylabel("y-value(handpalm)")
    plt.legend(['data', 'linear', 'cubic'], loc='best')
    plt.grid()

    ynew = yi(ti)

    return ti, ynew


def fourier_scipy(x, y):
    '''Alternative fft function'''
    yf = abs(scipy.fft(y))
    xf = scipy.fftpack.fftfreq(len(y), x[1] - x[0])

    plt.subplot(211)
    plt.plot(x, y)
    plt.subplot(212)
    plt.plot(xf, 20 * scipy.log10(yf), 'x')
    plt.show()


if __name__ == '__main__':
    data = np.load('measurements.npy')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store_true', help='Plot stuff')
    parser.add_argument('-w', action='store_true', help='Plot windowed stuff')
    parser.add_argument('-f', action='store_true', help='Plot fourier stuff')
    args = parser.parse_args()

    if args.p:
        plot_whole_data(data)
    elif args.w:
        plot_windows(data)
    elif args.f:
        plot_fourier(data)
