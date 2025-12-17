import casadi as ca
import numpy as np


def line_search(p,x0,method='newton',use_momentum = False):
    '''
    Unconstrained Problem.
    Just steepest descend (with meomentum)
    or newton
    '''
    # gradient of f
    hess_f_expr, grad_f_expr = ca.hessian(p.f_expr,p.x)
    grad_f = ca.Function("grad_f", [p.x], [grad_f_expr])
    hess_f = ca.Function("hess_f", [p.x], [hess_f_expr])

    x_path = [ca.DM(x0)]
    xk = ca.DM(x0)

    max_it = 10000
    tol = 1e-6
    max_it_ls = 50
    c1 = 1e-3
    rho = 0.5
    beta = 0.9
    d_prev = None 
    for k in range(max_it):

        grad_f_of_xk = grad_f(xk)
        grad_norm = float(ca.norm_2(grad_f_of_xk))

        if grad_norm < tol:
            print(f"Converged: ||grad f|| = {grad_norm}")
            break

        alpha = 1

        # ---------- choose direction d ----------
        if method == "sd":
            if use_momentum and d_prev is not None:
                d = -grad_f_of_xk + beta * d_prev
            else:
                d = -grad_f_of_xk
        elif method == "newton":
            d = ca.solve(hess_f(xk), -grad_f_of_xk)
        else:
            raise ValueError(f"Unknown method '{method}'")

        f_of_xk = p.f(xk)
        grad_f_of_k_T_d = grad_f_of_xk.T@d
        for i in range(max_it_ls):
            f_trial = p.f(xk+alpha*d)
            if float(f_trial) <= float(f_of_xk + c1*alpha*grad_f_of_k_T_d):
                break
            else:
                alpha = rho*alpha

        d_prev = d
        xk = xk + alpha*d
        x_path.append(xk)

    return x_path, xk


def newton_kkt(p,x0,globalization="merit"):
    # lam: lambda
    # L(x,lambda) = f(x) - lam * c(x)
    # linear KKT System: [hess(L,x),A;A,0][p_x,p_lam]=[grad(L,x);c]

    # Lagrangian:
    lam = ca.SX.sym('lam',p.me)
    L_expr = p.f_expr + lam.T @ p.c_e_expr
    hess_L_expr, grad_L_expr = ca.hessian(L_expr,p.x)
    grad_L = ca.Function('grad_L',[p.x, lam],[grad_L_expr])
    hess_L = ca.Function('hess_L',[p.x, lam],[hess_L_expr])

    grad_f_expr = ca.gradient(p.f_expr, p.x)
    grad_f = ca.Function("grad_f", [p.x], [grad_f_expr])

    x_path = [ca.DM(x0)]
    xk = ca.DM(x0)
    lamk = ca.DM.zeros(p.me,1)

    max_it = 1000
    max_ls = 50
    tol = 1e-6
    c1 = 1e-4
    rho = 0.5
    gamma_theta = 1e-4
    gamma_phi = 1e-4

    # --- Initialize filter ---
    if globalization == "filter":
        theta_k = float(ca.norm_2(p.c_e(xk)))  # constraint violation
        phi_k = float(p.f(xk))                 # objective
        filter = [(theta_k, phi_k)]

    for k in range(max_it):

        grad_Lk = grad_L(xk, lamk)
        ck  = p.c_e(xk)
        Ak  = p.A_e(xk)
        hess_Lk  = hess_L(xk, lamk)
        zero_mm = ca.DM.zeros(p.me,p.me)

        KKT = ca.vertcat(ca.hcat([hess_Lk, Ak.T]), ca.hcat([Ak, zero_mm]))
        Fk = ca.vertcat(grad_Lk, ck)

        # Error Function
        E = float(ca.norm_2(Fk))
        #E = max(float(ca.norm_2(grad_Lk)),float(ca.norm_2(ck)))
        print(f"iter {k}: ||res|| = {E:.3e}")

        if E < tol:
            print("Converged (KKT residual small).")
            break

        d = ca.solve(KKT, -Fk)
        dx  = d[0:p.n]
        dlam = d[p.n:]


        alpha = 1.0

        if globalization == "merit":
            # ---- Line Search using a Merit Function ----
            merit_fk = 0.5 * float(ca.norm_2(Fk))**2
            grad_merit_fk = KKT.T @ Fk
            grad_merit_fk_d = float(grad_merit_fk.T @ d)

            for _ in range(max_ls):
                x_trial = xk + alpha * dx
                lam_trial = lamk + alpha * dlam

                grad_L_trial = grad_L(x_trial, lam_trial)
                c_trial = p.c_e(x_trial)
                F_trial = ca.vertcat(grad_L_trial, c_trial)
                merit_trial = 0.5 * float(ca.norm_2(F_trial))**2

                # Armijo condition on merit
                if merit_trial <= merit_fk + c1 * alpha * grad_merit_fk_d:
                    break

                alpha *= rho

        elif globalization == "filter":
            # ---- Line Search using a Filter ----
            theta_k = float(ca.norm_2(ck))
            phi_k = float(p.f(xk))

            for _ in range(max_ls):
                x_trial = xk + alpha * dx
                lam_trial = lamk + alpha * dlam

                c_trial = p.c_e(x_trial)
                theta_trial = float(ca.norm_2(c_trial))
                phi_trial = float(p.f(x_trial))

                # (1) Not dominated by filter
                dominated = False
                for (theta_i, phi_i) in filter:
                    if theta_trial >= theta_i and phi_trial >= phi_i:
                        dominated = True
                        break

                # (2) Sufficient improvement vs. current point
                constraint_reduction = theta_trial <= (1.0 - gamma_theta) * theta_k
                objective_reduction = phi_trial <= phi_k - gamma_phi * theta_k

                if (not dominated) and (constraint_reduction or objective_reduction):
                    # acceptable to the filter
                    break

                alpha *= rho

        # accept step
        xk   = x_trial
        lamk = lam_trial
        x_path.append(xk)

    return x_path, xk


def penalty_ineq(p, x0, method="newton",rho0=1.0, rho_mult=10.0,max_outer=3):
    """
    Quadratic penalty method for inequality constraints c_i(x) >= 0.

    Penalty function:
        theta(x) = f(x) + (rho/2) * ||min(0, c_i(x))||^2

    method: "sd" or "newton"
    rho0:   initial penalty parameter rho
    rho_mult: factor to increase rho between outer iterations
    max_outer: number of outer penalty updates
    """

    # --- build penalty function symbolically ---
    rho_sym = ca.SX.sym("rho")
    # negative part: violation only
    v_expr = ca.fmin(0, p.c_i_expr)        # elementwise min(0, g_i)
    phi_expr = p.f_expr + 0.5 * rho_sym * ca.dot(v_expr, v_expr)

    # gradient and Hessian
    hess_phi_expr, grad_phi_expr = ca.hessian(phi_expr, p.x)
    phi_fun = ca.Function("phi_fun", [p.x, rho_sym], [phi_expr])
    grad_phi = ca.Function("grad_phi", [p.x, rho_sym], [grad_phi_expr])
    hess_phi = ca.Function("hess_phi", [p.x, rho_sym], [hess_phi_expr])

    # for convenience: violation norm v(x) = ||min(0, c_i(x))||
    def theta(x):
        v = ca.fmin(0, p.c_i(x))
        return float(ca.norm_2(v))

    x_path = [ca.DM(x0)]
    xk = ca.DM(x0)

    # parameters
    max_it_inner = 50
    tol_grad = 1e-6
    tol_feas = 1e-6
    c1 = 1e-3
    beta_ls = 0.5  # backtracking factor

    rho_val = rho0

    for outer in range(max_outer):
        print(f"\n[Penalty outer iter {outer}, rho = {rho_val}]")

        for k in range(max_it_inner):
            # current gradient and violation
            grad_phik = grad_phi(xk, rho_val)
            grad_norm = float(ca.norm_2(grad_phik))
            theta_k = theta(xk)

            print(f"  inner {k}: ||grad phi|| = {grad_norm:.3e}, θ = {theta_k:.3e}")

            # stopping criteria: small gradient and small violation
            if grad_norm < tol_grad and theta_k < tol_feas:
                print("  -> Inner loop converged.")
                break

            # choose direction d
            if method == "sd":
                d = -grad_phik
            elif method == "newton":
                Hk = hess_phi(xk, rho_val)
                d = ca.solve(Hk, -grad_phik)

            # Armijo backtracking on rho
            phik = float(phi_fun(xk, rho_val))
            phik_T_d = float(grad_phik.T @ d)

            alpha = 1.0
            for _ in range(50):
                x_trial = xk + alpha * d
                phi_trial = float(phi_fun(x_trial, rho_val))
                if phi_trial <= phik + c1 * alpha * phik_T_d:
                    break
                alpha *= beta_ls

            xk = x_trial
            x_path.append(xk)

        # after inner loop, check feasibility
        theta_k = theta(xk)
        print(f"  end outer {outer}: θ = {theta_k:.3e}")

        if theta_k < tol_feas:
            print("Penalty method: feasibility achieved, stopping outer loop.")
            break

        # otherwise increase penalty and continue with same xk
        rho_val *= rho_mult

    return x_path, xk


def interior_point_ineq(p, x0, method="newton", mu0=1.0, mu_factor=0.2, max_outer=100):
    """
    Log-barrier interior-point method for inequality constraints c_i(x) >= 0.

        phi(x) = f(x) - mu * sum_i log( c_i(x) )

    """

    # phi(x), gradient and Hessian w.r.t. x
    mu_sym = ca.SX.sym("mu")
    phi_expr = p.f_expr + -mu_sym * ca.sum1(ca.log(p.c_i_expr))
    hess_phi_expr, grad_phi_expr = ca.hessian(phi_expr, p.x)
    phi_fun  = ca.Function("phi_ip",  [p.x, mu_sym], [phi_expr])
    grad_phi = ca.Function("grad_ip", [p.x, mu_sym], [grad_phi_expr])
    hess_phi = ca.Function("hess_ip", [p.x, mu_sym], [hess_phi_expr])

    # --- initial point ---
    xk = ca.DM(x0)
    gk = p.c_i(xk)
    if float(np.min(gk)) <= 0:
        raise ValueError(
            f"interior_point_ineq: x0 must be strictly feasible, "
            f"but min c_i(x0) = {float(np.min(gk))}"
        )

    x_path = [xk]

    # parameters
    max_inner = 50
    tol_grad  = 1e-6
    tol_mu    = 1e-3
    c1        = 1e-3
    beta_ls   = 0.5    # backtracking factor

    mu_val = mu0

    for outer in range(max_outer):
        print(f"\n[IP outer iter {outer}, mu = {mu_val}]")

        # --- inner loop: minimize phi(x) with Newton / SD ---
        for k in range(max_inner):
            grad_k = grad_phi(xk, mu_val)
            grad_norm = float(ca.norm_2(grad_k))
            print(f"  inner {k}: ||grad phi|| = {grad_norm:.3e}")

            if grad_norm < tol_grad:
                print("  -> Inner loop converged.")
                break


            # direction
            if method == "sd":
                d = -grad_k
            elif method == "newton":
                Hk = hess_phi(xk, mu_val)
                d = ca.solve(Hk, -grad_k)


            phi_k = float(phi_fun(xk, mu_val))
            dphi_k = float(grad_k.T @ d)

            # --- line search ---
            alpha = 1.0
            for _ in range(50):
                x_trial = xk + alpha * d
                g_trial = p.c_i(x_trial)

                # feasibility: all c_i(x_trial) > 0
                if float(np.min(g_trial)) <= 0:
                    alpha *= beta_ls
                    continue

                phi_trial = float(phi_fun(x_trial, mu_val))

                # Armijo
                if phi_trial <= phi_k + c1 * alpha * dphi_k:
                    break

                alpha *= beta_ls

            xk = x_trial
            x_path.append(xk)

        # --- barrier update ---
        if mu_val < tol_mu:
            print("Interior-point: μ small enough, stopping outer loop.")
            break

        mu_val *= mu_factor

    return x_path, xk


def interior_point_pd_ineq(p,x0,max_outer=10,max_inner=20,tol_mu=1e-6,sigma_mu=0.1):
    """
    Primal-dual interior-point method for inequality constraints c_i(x) >= 0
    - Requires a strictly feasible starting point: c_i(x0) > 0 for all i

    Problem:    min f(x)  s.t.  c_i(x) >= 0,  i=1..m

    (x, z),  z_i >= 0 (multipliers for inequalities),

    solves the perturbed KKT system:
        grad_f(x) - Ai(x)^T z       = 0
        diag(z) ci(x) - tau e       = 0

    with tau decreasing as tau = sigma_mu * mu,  mu = (ci^T z)/m.
    """

    # Lagrangian L(x,z) = f(x) - z^T g(x)
    z_sym = ca.SX.sym("z", p.mi)
    L_expr = p.f_expr - z_sym.T @ p.c_i_expr
    hess_L_expr, grad_L_expr = ca.hessian(L_expr, p.x)
    grad_L = ca.Function("grad_L", [p.x, z_sym], [grad_L_expr])
    hess_L = ca.Function("hess_L", [p.x, z_sym], [hess_L_expr])

    # A = dc/dx
    Ai_expr = ca.jacobian(p.c_i_expr, p.x)
    Ai = ca.Function("Jg", [p.x], [Ai_expr])

    # initial point: must be strictly feasible
    xk = ca.DM(x0)
    cik = p.c_i(xk)
    zk = ca.DM.ones(p.mi, 1)
    x_path = [xk]

    # complementarity measure mu = (c_i^T z) / m
    def mu_measure(x, z):
        return float((p.c_i(x).T @ z) / p.mi)

    # Parameters
    mu = mu_measure(xk, zk)
    beta_ls = 0.5
    c_ls = 1e-4
    gamma_theta = 1e-4
    gamma_phi = 1e-4

    for outer in range(max_outer):

        print(f"\n[PD-IP outer {outer}] mu = {mu:.3e}")

        if mu < tol_mu:
            print("Primal-dual IP: complementarity small, stopping.")
            break

        # target complementarity level for this outer iteration
        tau = sigma_mu * mu

        # initialize filter for this outer iteration
        theta_k = mu
        phi_k = float(p.f(xk))
        filter_list = [(theta_k, phi_k)]

        for k in range(max_inner):
            cik = p.c_i(xk)
            Aik = Ai(xk)
            grad_Lk = grad_L(xk, zk)
            hess_Lk = hess_L(xk, zk)

            # residuals
            r_dual = grad_Lk
            r_cent = ca.diag(zk) @ cik - tau * ca.DM.ones(p.mi, 1)
            r = ca.vertcat(r_dual, r_cent)
            res_norm = float(ca.norm_2(r))
            print(f"  inner {k}: ||res|| = {res_norm:.3e}")

            # local stopping for this tau
            if res_norm < tau:
                break

            # KKT matrix
            Zk = ca.diag(zk)
            Cik = ca.diag(cik)

            # [ H  -Ai^T ]
            # [ Z Ai   Ci ]
            KKT_top = ca.hcat([hess_Lk, -Aik.T])
            KKT_bottom = ca.hcat([Zk @ Aik, Cik])
            KKT = ca.vertcat(KKT_top, KKT_bottom)

            d = ca.solve(KKT, -r)
            dx = d[0:p.n]
            dz = d[p.n:]

            # --- Filter line search ---
            alpha = 1.0
            theta_k = mu_measure(xk, zk)
            phi_k = float(p.f(xk))

            for _ in range(50):
                x_trial = xk + alpha * dx
                z_trial = zk + alpha * dz
                c_i_trial = p.c_i(x_trial)

                # stay in the interior: c_i(x) > 0, z > 0
                if float(np.min(c_i_trial)) <= 0 or float(np.min(z_trial)) <= 0:
                    alpha *= beta_ls
                    continue

                theta_trial = mu_measure(x_trial, z_trial)
                phi_trial = float(p.f(x_trial))

                # (1) not dominated by current filter
                dominated = False
                for (theta_i, phi_i) in filter_list:
                    if theta_trial >= theta_i and phi_trial >= phi_i:
                        dominated = True
                        break

                # (2) sufficient improvement vs current point
                constraint_reduction = theta_trial <= (1.0 - gamma_theta) * theta_k
                objective_reduction  = phi_trial <= phi_k - gamma_phi * theta_k

                if (not dominated) and (constraint_reduction or objective_reduction):
                    break

                alpha *= beta_ls

            xk = x_trial
            zk = z_trial
            x_path.append(xk)

        # update mu based on current (xk,zk)
        mu = mu_measure(xk, zk)

    return x_path, xk
