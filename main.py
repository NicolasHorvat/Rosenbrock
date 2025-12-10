##########################################################################
##                    +++ Optimization Methods +++                      ##
##                                                                      ##
##                        Rosenbrock Function                           ##
##                                                                      ##
##  - Unconstrained vs. Equality Constrained vs Inequality Constrained  ##
##  - Search direction/ length: Line Search vs. Trust Region            ##
##  - Step Acceptance: Merit vs. Filter                                 ##
##  - SQP vs. Interior-Point Method                                     ##
##                                                                      ##
##########################################################################


## Problem (Rosenbrock Function).
##
## Unconstraint: min f(x) = (a-x1)^2 + b(x2-x1^2)^2
##
## Equality Coinstraints

## +++ Imports +++
import casadi as ca
import time

from utils import *
from algorithms import *

## +++ Problem +++
class problem:
    def __init__(self):
        a = 1
        b = 100

        self.n = 2
        self.x = ca.SX.sym('x',self.n)
        x1 = self.x[0]
        x2 = self.x[1]
        self.f_expr = (a - x1)**2 + b * (x2 - x1**2)**2 # <-- objective function
        self.f = ca.Function("rosenbrock", [self.x], [self.f_expr])

        # Equality Constraints (c_e(x) = 0 for all i):
        self.me = 2
        c_e_0 = x1**2 + x2**2 - 4
        c_e_1 = -x1+x2
        self.c_e_expr = ca.vertcat(c_e_0,c_e_1)
        self.c_e = ca.Function("c_e", [self.x], [self.c_e_expr])

        A_e_expr = ca.jacobian(self.c_e_expr, self.x)
        self.A_e = ca.Function("A", [self.x], [A_e_expr])

        # Inquality Constraints (c_i(x) >= 0 for all i):
        self.mi = 2
        c_i_0 = -100*x1**2 - x2**2 + 1000
        c_i_1 = -x1*x2 +1
        #c_i_2 = 10*(x1+1)**2 + x2 -5
        self.c_i_expr = ca.vertcat(c_i_0,c_i_1)#,c_i_2)
        self.c_i = ca.Function("c_i", [self.x], [self.c_i_expr])

        A_i_expr = ca.jacobian(self.c_i_expr, self.x)
        self.A_i = ca.Function("A", [self.x], [A_i_expr])


results = []

def run_solver(label, solve_fun, p, x0, show_contour=True, **solver_kwargs):
    """Run an optimization method, print stats, and plot its path."""
    print(f'\n{label}')
    start = time.perf_counter()
    x_path, x_opt = solve_fun(p, x0, **solver_kwargs)
    end = time.perf_counter()

    n_steps = len(x_path) - 1
    runtime = end - start

    print(f'optimum found at x = {x_opt}')
    print(f'number of steps: {n_steps}')
    print(f"runtime: {runtime:.6f} seconds")

    results.append({
        "label": label,
        "steps": n_steps,
        "time": runtime,
    })

    return x_path, x_opt


print('\nRosenbock Function:')
p = problem()
print("f(x):", p.f_expr)



## +++ Initial Condition +++
x0 = [-2,20]
print(f'\n--> Starting from x0 = {x0}')



## +++ Line Search Methods +++
print('\nUnconstrained Case:')

x_path_sd, x_opt_sd = run_solver('Steepest Descent',line_search,p, x0, method='sd')
x_path_sdwm, x_opt_sdwm = run_solver('Steepest Descent',line_search,p, x0,method='sd',use_momentum=True)
x_path_newton, x_opt_newton = run_solver("Newton",line_search,p, x0,method='newton')


## +++ Equality Constrained Case +++
print('\nEquality Constrained Case:')

x_path_newton_kkt_merit, x_opt_newton_kkt_merit = run_solver('Newton–KKT, merit phi, eq. const.',newton_kkt,p, x0, globalization="merit")
x_path_newton_kkt_filter, x_opt_newton_kkt_filter = run_solver('Newton–KKT, filter, eq. const.',newton_kkt,p, x0, globalization="filter")


## +++ Inequality Constrained Case (Penalty) +++
print('\nInequality Constrained Case:')

x_path_penalty, x_opt_penalty = run_solver('Penalty, ineq. const.',penalty_ineq,p,x0,method='newton', rho0=1.0,rho_mult=10.0,max_outer=3)
x_path_ip, x_opt_ip = run_solver('Interior-Point, ineq. const.',interior_point_ineq,p,x0,method='newton',mu0=10,mu_factor=0.2,max_outer=5,)

## +++ Inequality Constrained Case: Primal-Dual IP +++
print('\nInequality Constrained Case: primal-dual interior-point')

x_path_ip_pd, x_opt_ip_pd = run_solver('Primal-Dual IP, ineq. const.',interior_point_pd_ineq,p,x0,max_outer=8,max_inner=20,tol_mu=1e-6,sigma_mu=0.1)



# --- one figure with subplots ---
fig, axes = plt.subplots(3, 3, figsize=(10, 8), facecolor="lightgray")

contour_plot(p,x_path_sd,ax=axes[0, 0],title="Steepest Descent")
contour_plot(p,x_path_sdwm,ax=axes[0, 1],title="Steepest Descent")
contour_plot(p,x_path_newton,ax=axes[0, 2],title="Newton")
contour_plot(p,x_path_newton_kkt_merit,ax=axes[1, 0],title="Newton–KKT, merit phi, eq. const.",plot_c_e=True)
contour_plot(p,x_path_newton_kkt_filter,ax=axes[1, 1],title="Newton–KKT, filter, eq. const.",plot_c_e=True)
contour_plot(p,x_path_penalty,ax=axes[2, 0],title="Penalty, ineq. const.",plot_c_i=True)
contour_plot(p,x_path_ip,ax=axes[2, 1],title="Interior-Point, ineq. const.",plot_c_i=True)
contour_plot(p,x_path_ip_pd,ax=axes[2, 2],title="Primal-Dual IP, ineq. const.",plot_c_i=True)



plt.tight_layout()
plt.show()


## +++ Table +++
print("\n" + "#" * 70)
print("# Summary: methods comparison")
print("#" * 70)

# header
print(f"{'Method':40s} {'Steps':>8s} {'Time [s]':>12s}")
print("-" * 70)

for r in results:
    print(f"{r['label'][:40]:40s} {r['steps']:8d} {r['time']:12.6f}")