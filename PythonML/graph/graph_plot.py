import matplotlib.pyplot as plt
import numpy as np

def plot_sequence(data, title='graph', labelx='X', labely='Y', marker='o'):
    plt.figure()
    plt.plot(range(1, len(data) + 1), data, marker=marker)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.savefig(title + '.png')


def plot_contour(x_pair, y_pair, z, title='Contour', x=None, y=None):
    plt.figure()
    z = z.reshape(x_pair.shape)
    plt.contourf(x_pair, y_pair, z, cmap=plt.cm.Paired)
    plt.axis('off')
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
    plt.savefig(title + '.png')

def plot_scatters(scatters, title='graph', labelx='X', labely='Y'):
    for scatter in scatters:
        plt.scatter(
            scatter['x'],
            scatter['y'],
            color=scatter['color'],
            marker=scatter['marker'],
            label=scatter['label'])
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.legend(loc='upper left')
    plt.savefig(title + '.png')
