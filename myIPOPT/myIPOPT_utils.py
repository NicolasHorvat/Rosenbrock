import numpy as np
import casadi as ca

# +++ Plotting +++
import matplotlib.pyplot as plt


def make_grid(f):
    # Create a grid in (x1, x2)
    x1_vals = np.linspace(-5, 5, 100)
    x2_vals = np.linspace(-5, 30, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Evaluate f on the grid
    F_vals = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            # f expects a 2-vector [x1, x2]
            F_vals[i, j] = float(f([X1[i, j], X2[i, j]]))
    return X1, X2, F_vals


def suface_plot(f):
    X1,X2,F_vals = make_grid(f)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, F_vals, rstride=5, cstride=5, alpha=0.9)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    ax.set_title("Rosenbrock Function")


def contour_plot(p, x_path=None, ax=None, title=None, plot_c_e=False,plot_c_i=False):

    # get grid
    X1, X2, F_vals = make_grid(p.f)

    # objective contours
    contours = ax.contourf(X1, X2, F_vals, levels=100,cmap="viridis")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)

    # optimization path
    if x_path is not None and len(x_path) > 0:
        xs, ys = [], []
        for xk in x_path:
            xs.append(float(xk[0]))
            ys.append(float(xk[1]))
        ax.plot(xs, ys, marker='o',color='orange', linewidth=1.5)

    # equality constraints c_e(x) = 0 as red lines
    if plot_c_e and p.me > 0:
        C_vals = np.zeros_like(X1)
        for k in range(p.me):
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    x_ij = ca.DM([X1[i, j], X2[i, j]])
                    C_vals[i, j] = float(p.c_e(x_ij)[k])

            ax.contour(X1, X2, C_vals,levels=[0.0],colors='limegreen',linewidths=1)

    # inequality constraints c_i(x) >= 0 as green
    if plot_c_i and p.mi > 0:
        C_vals = np.zeros_like(X1)
        for k in range(p.mi):
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    x_ij = ca.DM([X1[i, j], X2[i, j]])
                    C_vals[i, j] = float(p.c_i(x_ij)[k])

            ax.contour(X1, X2, C_vals,levels=[0.0],colors='r',linestyles="--",linewidths=1)
            ax.contourf(X1,X2,(C_vals < 0).astype(float),levels=[0.5, 1.5],colors="r",alpha=0.25,)
            
    return ax, contours