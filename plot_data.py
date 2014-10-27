import numpy as np
from matplotlib import pyplot as plt
from matplotlib.mlab import PCA, specgram
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

class Arrow3D(FancyArrowPatch):
    '''3D arrow for plotting'''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

if __name__ == '__main__':
    data = np.load('measurements.npy')
    t,c,x,y,z = zip(*data)
    pca = PCA(np.array(zip(x,y,z)))

    fig = plt.figure()
    # Because mplot3d and leapmotion use different axes, we switch Z and Y here for a more natural look

    # 3d plot
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    plt.title("Tracked position in 3d space")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_zlabel('Y axis')
    ax.scatter(x,z,y)

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
    ax.arrow(mu_x, mu_y, pv_x-mu_x, pv_y-mu_y, width=2, head_width=5, head_length=5, fc='k', ec='k')
    ax.plot(x,y)
    ax.axis('equal')

    # x, y, z over time
    ax = fig.add_subplot(222)
    plt.title("Position over time")
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Millimeters')
    ax.plot(t,x, label='x')
    ax.plot(t,y, label='y')
    ax.plot(t,z, label='z')
    legend = plt.legend()
    ax.add_artist(legend)

    # spectogram
    ax = fig.add_subplot(2, 2, 4)
    ax.specgram(y, scale_by_freq=True)
    
    plt.title("Spectogram")
    plt.show()
