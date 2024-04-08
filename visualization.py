import numpy as np

import matplotlib.pyplot as plt

# Generate some random data for the box plots
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(0, 2, 100)
data3 = np.random.normal(0, 3, 100)

def plot(data123, labels123, ylabel, title, save=True, show=True):
    fig, ax = plt.subplots()
    ax.boxplot(data123, labels=labels123)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if save:
        path = f'Results/{ylabel}/{title}.png'
        plt.savefig(path)
        print(f"Figure saved to {path}")

    if show:
        plt.show()
    
    plt.close(fig)


if __name__ == "__main__":
    data = [
        [3, 4, 3, 3, 3, 3, 4, 3, 4, 4],
        [4, 3, 4, 4, 4, 4, 3, 4, 3, 3],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [2, 2, 2, 2, 2, 1, 2, 1, 1, 1],
        [1, 1, 1, 1, 1, 2, 1, 2, 2, 2]
    ]
    labels = ["KNN", "MissForest", "MICE", "LERP", "SLERP"]
    ylabel = "Quality Rank"
    motion_name = "UniformlyAcceleratedMotion"
    missing_fraction = 0.5
    title = f"{motion_name}, {int(missing_fraction * 100)}% missing data"
    plot(data, labels, ylabel, title)
