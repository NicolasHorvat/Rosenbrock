

###              +++ IPOPT - Algorithm +++

## +++ Imports +++
from myIPOPT_utils import *
from myIPOPT_algo import *

## +++ Problems +++
class problem:
    def __init__(self):

        # --- objective function ---
        a = 1
        b = 100
        self.n = 2
        self.x = ca.SX.sym('x',self.n)
        x1 = self.x[0]
        x2 = self.x[1]
        self.f_expr = (a - x1)**2 + b * (x2 - x1**2)**2
        hess_f_expr, grad_f_expr = ca.hessian(self.f_expr, self.x)
        self.f = ca.Function("rosenbrock", [self.x], [self.f_expr])
        self.grad_f = ca.Function("grad_f", [self.x], [grad_f_expr])
        self.hess_f = ca.Function("hess_f", [self.x], [hess_f_expr])

        # --- Equality Constraints (c_e(x) = 0 for all i) ---
        self.me = 2
        c_e_0 = x1**2 + x2**2 - 4
        c_e_1 = -x1+x2
        # self.me = 3
        # c_e_2 = 2*(-x1+x2) # redundant constraint
        # self.c_e_expr = ca.vertcat(c_e_0,c_e_1,c_e_2)
        self.c_e_expr = ca.vertcat(c_e_0,c_e_1)
        self.c_e = ca.Function("c_e", [self.x], [self.c_e_expr])

        A_e_expr = ca.jacobian(self.c_e_expr, self.x)
        self.A_e = ca.Function("A", [self.x], [A_e_expr])

        # --- Inquality Constraints (c_i(x) = 0 for all i) (only used for plotting)---
        self.mi = 2
        c_i_0 = -100*x1**2 - x2**2 + 1000
        c_i_1 = x1*x2 +1
        self.c_i_expr = ca.vertcat(c_i_0,c_i_1)
        self.c_i = ca.Function("c_i", [self.x], [self.c_i_expr])


        # --- Lagrangian ---
        lam = ca.SX.sym("lam", self.me)
        z   = ca.SX.sym("z", self.n)
        L_expr = self.f_expr + lam.T @ self.c_e_expr - z.T @ self.x  # L = f + λ^T c - z^T x
        hess_L_expr, grad_L_expr = ca.hessian(L_expr, self.x)
        self.grad_L = ca.Function("grad_L", [self.x, lam, z], [grad_L_expr])
        self.hess_L = ca.Function("hess_L", [self.x, lam, z], [hess_L_expr])

class problem_slack:
    def __init__(self):

        # --- objective function ---
        a = 1
        b = 100

        self.n = 2 + 2

        self.w = ca.SX.sym("w", self.n) # w = (x,s)

        x1 = self.w[0]
        x2 = self.w[1]

        s1 = self.w[2] # slack variables
        s2 = self.w[3]
        #s3 = self.w[4]

        #self.f_expr = (a - x1)**2 + b * (x2 - x1**2)**2   # rosenbrock function
        self.f_expr = x1*x2

        hess_f_expr, grad_f_expr = ca.hessian(self.f_expr, self.w)
        self.f = ca.Function("rosenbrock", [self.w], [self.f_expr])
        self.grad_f = ca.Function("grad_f", [self.w], [grad_f_expr])
        self.hess_f = ca.Function("hess_f", [self.w], [hess_f_expr])

        # --- Equality Constraints (c_e(x) = 0 for all i) ---
        self.me = 2 + 2
        self.mi = 2
        c_e_0 = x1**2 + x2**2 - 4
        c_e_1 = -x1+x2
        c_i_0 = -100*x1**2 - x2**2 + 1000 - s1
        c_i_1 = x1*x2 +1 - s2
        #c_i_2 = 10*(x1+1)**2 + x2 -5 - s3

        self.c_e_expr = ca.vertcat(c_e_0,c_e_1, c_i_0,c_i_1)
        self.c_e = ca.Function("c_e", [self.w], [self.c_e_expr])

        A_e_expr = ca.jacobian(self.c_e_expr, self.w)
        self.A_e = ca.Function("A", [self.w], [A_e_expr])

        # --- Lagrangian ---
        lam = ca.SX.sym("lam", self.me)
        z   = ca.SX.sym("z", self.n)
        L_expr = self.f_expr + lam.T @ self.c_e_expr - z.T @ self.w  # L = f + λ^T c - z^T x
        hess_L_expr, grad_L_expr = ca.hessian(L_expr, self.w)
        self.grad_L = ca.Function("grad_L", [self.w, lam, z], [grad_L_expr])
        self.hess_L = ca.Function("hess_L", [self.w, lam, z], [hess_L_expr])


def main():
    p = problem()
    ps = problem_slack()
    x01 = [0.5,10]
    x02 = [2,15]

    def get_s0(x0):
        return ca.vertcat(-100*x0[0]**2 - x0[1]**2 + 1000,-x0[0]*x0[1] + 1)

    s_min = 1e-3
    s_max = 1e3   # or 1e3, something moderate

    s01 = ca.fmax(s_min, ca.fmin(get_s0(x01), s_max))
    s02 = ca.fmax(s_min, ca.fmin(get_s0(x02), s_max))

    w01 = ca.vertcat(x01,s01)
    w02 = ca.vertcat(x02,s02)

    maxj=20
    maxk=50

    use_debug = False

    if use_debug == False:
        print("\n ++++++++++++++++++++ Start eq. only problem 1 ++++++++++++++++++++")
        x_path1, x_sol1, runtime1 = myIPOPT(p,x01,maxj,maxk)
        print("\n ++++++++++++++++++++ Start eq. only problem 2 ++++++++++++++++++++")
        x_path2, x_sol2, runtime2 = myIPOPT(p,x02,maxj,maxk)
        print("\n ++++++++++++++++++++ Start problem_slack 3 ++++++++++++++++++++")
        x_path3, x_sol3, runtime3 = myIPOPT(ps,w01,maxj,maxk)
        print("\n ++++++++++++++++++++ Start problem_slack 4 ++++++++++++++++++++")
        x_path4, x_sol4, runtime4 = myIPOPT(ps,w02,maxj,maxk)

    else:
        print("\n ++++++++++++++++++++ Start eq. only problem 1 ++++++++++++++++++++")
        x_path1, x_sol1, runtime1 = myIPOPT_debug(p,x01,maxj,maxk)
        print("\n ++++++++++++++++++++ Start eq. only problem 2 ++++++++++++++++++++")
        x_path2, x_sol2, runtime2 = myIPOPT_debug(p,x02,maxj,maxk)
        print("\n++++++++++++++++++++  Start problem_slack 3 ++++++++++++++++++++")
        x_path3, x_sol3, runtime3 = myIPOPT_debug(ps,w01,maxj,maxk)
        print("\n ++++++++++++++++++++ Start problem_slack 4 ++++++++++++++++++++")
        x_path4, x_sol4, runtime4 = myIPOPT_debug(ps,w02,maxj,maxk)

    print(f"\n(1) last x = {x_sol1}")
    print(f"this took {runtime1} sec")
    print(f"\n(2) last x = {x_sol2}")
    print(f"this took {runtime2} sec")
    print(f"\n(3) last x = {x_sol3}")
    print(f"this took {runtime3} sec")
    print(f"\n(4) last x = {x_sol4}")
    print(f"this took {runtime4} sec\n")


    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor="lightgray")

    # equality problem
    contour_plot(p, x_path1, ax=axes[0, 0], title="myIPOPT (eq only, x0=[1,10])", plot_c_e=True)
    contour_plot(p, x_path2, ax=axes[0, 1], title="myIPOPT (eq only, x0=[2,20])", plot_c_e=True)

    # slack problem – projected to (x1,x2) in the contour_plot implementation
    contour_plot(p, x_path3, ax=axes[1, 0], title="myIPOPT (slack, w0 from x0=[1,10])", plot_c_e=True, plot_c_i=True)
    contour_plot(p, x_path4, ax=axes[1, 1], title="myIPOPT (slack, w0 from x0=[2,20])", plot_c_e=True, plot_c_i=True)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()