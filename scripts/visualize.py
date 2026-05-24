import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def plot_graph_with_points(graph, path):
    points = np.array(path)
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={"projection": "3d"})
    
    for link in graph["links"]:
        p = np.array(link["path"])
        if len(p):
            ax.plot(p[:, 2], p[:, 1], p[:, 0], color="#625F5F", alpha=0.8, lw=1.8, zorder=1)
            
    if len(points):
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#1f77b4", lw=1.2, alpha=0.6, zorder=2)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="#1f77b4", s=25, ec='black', lw=0.5, zorder=3)
        ax.scatter(*points[0], color="lime", s=130, ec='black', lw=1.5, label="Start", zorder=5)
        ax.scatter(*points[-1], color="red", s=130, ec='black', lw=1.5, label="End", zorder=5)
        
    all_paths = np.concatenate([l["path"] for l in graph["links"]])
    visual_paths = all_paths[:, [2, 1, 0]]
    combined_pts = np.vstack([visual_paths, points]) if len(points) > 0 else visual_paths
    x_min, x_max = combined_pts[:, 0].min(), combined_pts[:, 0].max()
    y_min, y_max = combined_pts[:, 1].min(), combined_pts[:, 1].max()
    z_min, z_max = combined_pts[:, 2].min(), combined_pts[:, 2].max()
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    center_z = (z_max + z_min) / 2
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_zlim(center_z - max_range, center_z + max_range)
    ax.set_xlabel(r"X [$\mu$m]")
    ax.set_ylabel(r"Y [$\mu$m]")
    ax.set_zlabel(r"Z [$\mu$m]")
    ax.set_box_aspect([1, 1, 1])
    um_formatter = FuncFormatter(lambda x, pos: f'{x:.0f}')
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.set_major_formatter(um_formatter)
    ax.view_init(elev=25, azim=35)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

def visualize_result_signal(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    signals = data['signals']
    b_values = data['b_values']
    gradient_directions = data['directions']
    params = data['params'].item()
    print("Parameters used for simulation:")
    for key, value in params.items():
        print(f"{key}: {value}")
    print("B-values:", b_values)
    print("Gradient directions:\n", gradient_directions)
    for i, gradient_direction in enumerate(gradient_directions):
        label = f"G direction: {gradient_direction}"
        plt.plot(b_values, signals[i], marker='o', label=label)
    plt.yscale('log')
    plt.xlabel('b-value [s/mm$^2$]')
    plt.ylabel('Normalized signal intensity')
    plt.title('Signal attenuation')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_matplotlib_ellipsoid(D, title="Elipsoid", ax=None):
    evals, evecs = np.linalg.eigh(D)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    radii = np.sqrt(np.abs(evals))
    x_ell, y_ell, z_ell = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vec = np.array([x[i,j], y[i,j], z[i,j]])
            vec_new = evecs @ (vec * radii) 
            x_ell[i,j] = vec_new[0]
            y_ell[i,j] = vec_new[1]
            z_ell[i,j] = vec_new[2]
    show_at_end = False
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        show_at_end = True 
    ax.plot_surface(x_ell, y_ell, z_ell, color='c', alpha=0.6, edgecolor='k', lw=0.15)
    center = [0, 0, 0]
    for i in range(3):
        axis_vec = evecs[:, i] * radii[i]
        ax.plot([center[0], axis_vec[0]], [center[1], axis_vec[1]], [center[2], axis_vec[2]], color='r', lw=1.0)
    max_radius = np.max(radii)
    limit = max_radius 
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_proj_type('ortho')
    try:
        ax.set_box_aspect([1,1,1], zoom=1.6)
    except TypeError:
        ax.set_box_aspect([1,1,1])
        ax.dist = 6.5
    ax.set_axis_off()
    ax.margins(0, 0, 0)
    ax.set_title(title, pad=0)
    ax.view_init(elev=-90, azim=-90)
    if show_at_end:
        plt.show()

def plot_ivim_tensors(graph_data, f_tensor, D_star_tensor, D_tensor):
    subplot_num = 4 if graph_data else 3
    fig = plt.figure(figsize=(4 * subplot_num, 4)) 
    
    ax1 = fig.add_subplot(1, subplot_num, 1, projection='3d')
    plot_matplotlib_ellipsoid(f_tensor, title="f Tensor", ax=ax1)
    
    ax2 = fig.add_subplot(1, subplot_num, 2, projection='3d')
    plot_matplotlib_ellipsoid(D_star_tensor, title="D* Tensor", ax=ax2)
    
    ax3 = fig.add_subplot(1, subplot_num, 3, projection='3d')
    plot_matplotlib_ellipsoid(D_tensor, title="D Tensor", ax=ax3)
    
    if graph_data:
        ax4 = fig.add_subplot(1, subplot_num, 4, projection='3d')
        ax4.set_title("Vascular Network", pad=0)
        for link in graph_data["links"]:
            p = np.array(link["path"])
            if len(p):
                ax4.plot(p[:, 2], p[:, 1], p[:, 0], color="#625F5F", alpha=0.8, lw=1.8, zorder=1)
        all_paths = np.concatenate([l["path"] for l in graph_data["links"]])
        x_data = all_paths[:, 2]
        y_data = all_paths[:, 1]
        z_data = all_paths[:, 0]
        center_x = (x_data.max() + x_data.min()) / 2
        center_y = (y_data.max() + y_data.min()) / 2
        center_z = (z_data.max() + z_data.min()) / 2
        max_range = max(x_data.max() - x_data.min(), 
                        y_data.max() - y_data.min(), 
                        z_data.max() - z_data.min()) / 2
        ax4.set_xlim(center_x - max_range, center_x + max_range)
        ax4.set_ylim(center_y - max_range, center_y + max_range)
        ax4.set_zlim(center_z - max_range, center_z + max_range)
        try:
            ax4.set_box_aspect([1, 1, 1], zoom=1.6)
        except TypeError:
            ax4.set_box_aspect([1, 1, 1])
            ax4.dist = 6.5
        ax4.set_proj_type('ortho')
        ax4.view_init(elev=-90, azim=-90)
        ax4.set_axis_off()
    plt.tight_layout()
    plt.show()