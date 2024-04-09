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


def plot_trajectory(file_paths, color_maps):
    fig, ax = plt.subplots()
    
    for i, file_path in enumerate(file_paths):
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        t = data[:, 0]
        x = data[:, 1]
        y = data[:, 2]
        
        # Normalize t values to range [0, 1]
        t_normalized = (t - np.min(t)) / (np.max(t) - np.min(t))
        
        # Create color map based on normalized t values
        cmap = plt.cm.get_cmap(color_maps[i])
        colors = cmap(t_normalized)
        
        # Plot the trajectory
        scatter = ax.scatter(x, y, c=colors, label=file_path, s=1)
        
        # Add color (bggy)
        # plt.colorbar(scatter, label='Time')  # Add colorbar using the scatter plot
        
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('UAM, Parabolic Trajectory')
    # ax.set_title('Trajectories')
    # scatter = ax.scatter([], [], c=colors, label='Time')  # Create an empty scatter plot
    # plt.colorbar(scatter, label='Time')  # Add colorbar using the scatter plot
    plt.show()




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
    plot_trajectory(file_paths, all_colormaps)
    
    EM_file_names = [
        "truth_EllipticalMotion,xr=2,yr=3.txt",
        # "inbetweened_EllipticalMotion,xr=2,yr=3_KNNImputation_0.9_0.6400108337402344_0.txt",
        # "inbetweened_EllipticalMotion,xr=2,yr=3_MissForestImputation_0.9_3.813678503036499_0.txt",
        # "inbetweened_EllipticalMotion,xr=2,yr=3_MiceImputation_0.9_0.022425174713134766_0.txt",
        "inbetweened_EllipticalMotion,xr=2,yr=3_LinearInterpolator_0.9_0.008187532424926758_0.txt",
        "inbetweened_EllipticalMotion,xr=2,yr=3_SphericalLinearInterpolator_0.9_0.08840513229370117_0.txt"
    ]
    file_paths = [directory + '/' + file_name for file_name in EM_file_names]
    # plot_trajectory(file_paths, imp_colormaps)
    # plot_trajectory(file_paths, erp_colormaps)
    
