import numpy as np

import matplotlib
from matplotlib import animation
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


def plot_trajectory_from_file(file_paths, color_maps, title):
    fig, ax = plt.subplots()
    
    for i, file_path in enumerate(file_paths):
        data = np.genfromtxt(file_path, delimiter=',')
        t = data[:, 0]
        x = data[:, 1]
        y = data[:, 2]
        
        # Normalize t values to range [0, 1]
        t_normalized = (t - np.min(t)) / (np.max(t) - np.min(t))
        
        # Create color map based on normalized t values
        cmap = matplotlib.colormaps[color_maps[i]]
        colors = cmap(t_normalized)
        
        # Plot the trajectory
        scatter = ax.scatter(x, y, c=colors, label=file_path, s=1)

    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    # scatter = ax.scatter([], [], c=colors, label='Time')  # Create an empty scatter plot
    # plt.colorbar(scatter, label='Time')  # Add colorbar using the scatter plot
    plt.show()
    
    
def plot_trajectory_from_data(data, color, title):
    fig, ax = plt.subplots()
    data = np.array([row for row in data if None not in row])
    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    
    # Normalize t values to range [0, 1]
    t_normalized = (t - np.min(t)) / (np.max(t) - np.min(t))
    
    # Create color map based on normalized t values
    cmap = matplotlib.colormaps[color]
    colors = cmap(t_normalized)
    
    # Plot the trajectory
    scatter = ax.scatter(x, y, c=colors, s=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    plt.show()


def animate_keyframes(keyframes, fps=10):
    fig, ax = plt.subplots()
    min_x = np.min(keyframes[:, 1]) - 1e-6
    max_x = np.max(keyframes[:, 1]) + 1e-6
    ax.set_xlim(min_x, max_x)
    min_y = np.min(keyframes[:, 2]) - 1e-6
    max_y = np.max(keyframes[:, 2]) + 1e-6
    ax.set_ylim(min_y, max_y)
    # ax.set_aspect('equal')
    ax.set_axis_off()

    ball, = ax.plot([], [], 'ro')

    def update(frame):
        # i, x, y = frame
        ball.set_data((frame[1],), (frame[2],))
        return ball,

    anim = animation.FuncAnimation(fig, update, frames=keyframes, interval=1000/fps, blit=True)
    plt.show()

    output_file = 'animation.mp4'
    anim.save(output_file, writer='ffmpeg')
    print(f'Animation saved to {output_file}')



if __name__ == "__main__":
    directory = 'Data'
    imp_colormaps = ['Greys', 'Reds', 'Greens', 'Blues']
    erp_colormaps = ['Greys', 'Purples', 'Oranges']
    all_colormaps = ['Greys', 'Reds', 'Greens', 'Blues', 'Purples', 'Oranges']
    
    
    UAM_file_names = [
        "truth_UniformlyAcceleratedMotion-p0=[0, 0],v0=[10, 10],a=[0, -0.02].txt",
        # "inbetweened_UniformlyAcceleratedMotion-p0=[0, 0],v0=[10, 10],a=[0, -0.02]_KNNImputation_0.9_0.612403154373169_0.txt",
        # "inbetweened_UniformlyAcceleratedMotion-p0=[0, 0],v0=[10, 10],a=[0, -0.02]_MissForestImputation_0.9_7.496107339859009_0.txt",
        # "inbetweened_UniformlyAcceleratedMotion-p0=[0, 0],v0=[10, 10],a=[0, -0.02]_MiceImputation_0.9_0.05576825141906738_0.txt",
        # "inbetweened_UniformlyAcceleratedMotion-p0=[0, 0],v0=[10, 10],a=[0, -0.02]_LinearInterpolator_0.9_0.010101318359375_0.txt",
        # "inbetweened_UniformlyAcceleratedMotion-p0=[0, 0],v0=[10, 10],a=[0, -0.02]_SphericalLinearInterpolator_0.9_0.026927947998046875_0.txt",
        # "truth_UniformlyAcceleratedMotion-p0=[0, 0],v0=[1, 2],a=[0.3, 0.4].txt",
        # "inbetweened_UniformlyAcceleratedMotion-p0=[0, 0],v0=[1, 2],a=[0.3, 0.4]_KNNImputation_0.9_0.09024286270141602_9.txt",
        # "inbetweened_UniformlyAcceleratedMotion-p0=[0, 0],v0=[1, 2],a=[0.3, 0.4]_MissForestImputation_0.9_5.82453465461731_9.txt",
        # "inbetweened_UniformlyAcceleratedMotion-p0=[0, 0],v0=[1, 2],a=[0.3, 0.4]_MiceImputation_0.9_0.047312021255493164_9.txt",
        # "inbetweened_UniformlyAcceleratedMotion-p0=[0, 0],v0=[1, 2],a=[0.3, 0.4]_LinearInterpolator_0.9_0.0071563720703125_9.txt",
        # "inbetweened_UniformlyAcceleratedMotion-p0=[0, 0],v0=[1, 2],a=[0.3, 0.4]_SphericalLinearInterpolator_0.9_0.025134563446044922_9.txt"
    ]
    file_paths = [directory + '/' + file_name for file_name in UAM_file_names]
    # plot_trajectory_from_file(file_paths, all_colormaps, "UAM")
    
    
    EM_file_names = [
        "truth_EllipticalMotion,xr=2,yr=3.txt",
        "inbetweened_EllipticalMotion,xr=2,yr=3_KNNImputation_0.9_0.6400108337402344_0.txt",
        "inbetweened_EllipticalMotion,xr=2,yr=3_MissForestImputation_0.9_3.813678503036499_0.txt",
        "inbetweened_EllipticalMotion,xr=2,yr=3_MiceImputation_0.9_0.022425174713134766_0.txt",
        "inbetweened_EllipticalMotion,xr=2,yr=3_LinearInterpolator_0.9_0.008187532424926758_0.txt",
        "inbetweened_EllipticalMotion,xr=2,yr=3_SphericalLinearInterpolator_0.9_0.08840513229370117_0.txt",
        # "inbetweened_EllipticalMotion,xr=2,yr=3_SphericalLinearInterpolator_0.5_0.008874893188476562_8.txt"
    ]
    file_paths = [directory + '/' + file_name for file_name in EM_file_names]
    # plot_trajectory_from_file(file_paths, imp_colormaps, "Elliptical Motion")
    # plot_trajectory_from_file(file_paths, erp_colormaps, "Elliptical Motion")
    
    
    SHM_file_names = [
        "truth_SimpleHarmonicMotion-a=2,f=0.1,p=3.txt",
        "inbetweened_SimpleHarmonicMotion-a=2,f=0.1,p=3_KNNImputation_0.9_0.06306743621826172_0.txt",
        "inbetweened_SimpleHarmonicMotion-a=2,f=0.1,p=3_MissForestImputation_0.9_1.9600608348846436_0.txt",
        "inbetweened_SimpleHarmonicMotion-a=2,f=0.1,p=3_MiceImputation_0.9_0.033025264739990234_0.txt",
        "inbetweened_SimpleHarmonicMotion-a=2,f=0.1,p=3_LinearInterpolator_0.9_0.0_0.txt",
        "inbetweened_SimpleHarmonicMotion-a=2,f=0.1,p=3_SphericalLinearInterpolator_0.9_0.031299591064453125_0.txt"
    ]
    file_paths = [directory + '/' + file_name for file_name in SHM_file_names]
    # plot_trajectory_from_file(file_paths, all_colormaps, "SHM")
    
    
    keyframes = np.genfromtxt(directory + '/' + SHM_file_names[2], delimiter=',')
    animate_keyframes(keyframes, fps=10)
