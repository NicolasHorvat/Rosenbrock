import time
import casadi as ca
import numpy as np


def myIPOPT(p,x0, maxj, maxk):
    """
    IPOPT primal-dual interior-point filter line-search algorithm.
    Problem form:  min f(x)  s.t.  c(x) = 0,  x >= 0. (whith slack x =(x,s))
    """
    start = time.perf_counter() # timer start
    n = p.n
    m = p.me
    e = ca.DM.ones(n, 1)
    tol = 1e-8

    # ----------- helper functions ------------------------------------------------- #
    def barrier_phi(x, mu):
        """Barrier objective phi_mu(x) = f(x) - mu sum(ln(x_i))."""
        return float(p.f(x) - mu * ca.sum1(ca.log(x)))
    
    def grad_barrier_phi(x, mu):
        """grad of barrier objective grad_phi_mu(x) = grad_f(x) - mu (1/x_i)."""
        return p.grad_f(x) - mu * (1.0/x)

    def theta(x):
        """Constraint violation θ(x) = ||c(x)||. """  
        return float(ca.norm_1(p.c_e(x))) 

    def optimality_error(x, lam, z, mu, return_components=False):
        """
        E_mu(x,lam,z) = max(||grad(f(x))+A*lam-z||inf, ||c(x)||inf, ||XZe-mu*e||inf)
        (no scaling parameters for now)
        """
        r_dual = p.grad_f(x) + p.A_e(x).T @ lam - z
        r_pri  = p.c_e(x)
        r_comp = ca.diag(x) @ ca.diag(z) @ e - mu * e

        e_dual = float(ca.norm_inf(r_dual))
        e_pri  = float(ca.norm_inf(r_pri))
        e_comp = float(ca.norm_inf(r_comp))

        if return_components:
            return e_dual, e_pri, e_comp

        return max(e_dual, e_pri, e_comp)

    def update_mu(mu):
        """Barrier update rule: mu_j+1 = max(mu_tol, kappa_mu * mu_j)"""
        return max(tol/10, min(kappa_mu * mu,mu**theta_mu))
    
    def frac_to_bound_tau(mu):
        """ fraction-to-the-boundrary parameter (8)"""
        return max(tau_min,1-mu)

    # --- initialization ---------------------------

    mu          = 10    # not working for 0.1
    kappa_e     = 10
    kappa_mu    = 0.99  # should be 0.2 8but does not work with 0.2)
    theta_mu    = 1.5
    tau_min     = 0.99

    xk      = ca.DM(x0) #  x > 0
    lamk    = ca.DM.zeros(p.me)
    #zk      = ca.DM.ones(n, 1)
    zk    = mu / xk
    print(f"z = {zk}")

    # params for Line Search / Filter (Section 2.3)
    gamma_theta = 1e-5
    gamma_phi   = 1e-5
    gamma_alpha = 0.05

    alpha_min   = 1e-8
    beta_ls     = 0.5
    theta0      = theta(xk)              # <-- constraint violation
    phi0        = barrier_phi(xk, mu)    # <-- barrier function

    delta_bar_c         = 1e-8
    kappa_c             = 0.25
    delta_bar_w_0       = 1e-4
    delta_bar_w_min     = 1e-20
    delta_bar_w_max     = 1e40
    delta_w_last        = 0.0
    kappa_bar_w_plus    = 100.0
    kappa_w_plus        = 8.0
    kappa_w_minus       = 1.0/3.0

    theta_max = 1e4 * max(1.0, theta0)   # (from parameter list at end of section 2.5)
    theta_min = 1e-4 * max(1.0, theta0)

    s_phi = 2.3 # params for switching condition
    s_theta = 1.1
    delta = 1

    nu_phi = 1e-4
    kappa_soc = 0.99 # for second order correction (not implemented)
    p_max = 4


    # ======================== (A-1) =============================================

    tau = frac_to_bound_tau(mu)
    filter_list = [(theta_max, -np.inf)] # Initial filter F0 = { (theta, phi) : theta ≥ theta_max }

    x_path = [xk] # Path of iterates (for plotting)

    
    # ===================== Outer Loop (barrier problems - decreasing mu) =======================
    for j in range(maxj):

        e_dual, e_pri, e_comp = optimality_error(xk, lamk, zk, mu=0.0, return_components=True)
        print(f"j={j}: mu = {mu:.3e}, E_0 = {max(e_dual,e_pri,e_comp):.3e}, "
                f"dual={e_dual:.3e}, pri={e_pri:.3e}, comp={e_comp:.3e}, tol={tol:.3e}")
        
        # Check convergence for original problem: E_0 <= tol (A-2)
        if optimality_error(xk, lamk, zk, mu=0.0) <= tol:
            print("outer converged!!!!")
            end = time.perf_counter() # timer end
            runtime = end - start
            return x_path, xk, runtime


        # ===================== Inner Loop (solve barrier problem for current mu) =====================
        for k in range(maxk):

            e_dual, e_pri, e_comp = optimality_error(xk, lamk, zk, mu, return_components=True)
            print(f"k={k}: mu = {mu:.3e}, E_mu = {max(e_dual,e_pri,e_comp):.3e}, "
                f"dual={e_dual:.3e}, pri={e_pri:.3e}, comp={e_comp:.3e}, kappa_e*mu={kappa_e*mu:.3e}")

            # Check convergence for current barrier problem: E_mu <= kappa_e*muj (A-3)
            if optimality_error(xk, lamk, zk, mu) <= kappa_e*mu:
                print("inner converged!!!!\n")
                # -------- update mu and tau (A-3.1) ---------
                mu = update_mu(mu)
                tau = frac_to_bound_tau(mu)

                # -------- re-initialize filter (A-3.2) --------
                theta_k     = theta(xk)
                theta_max   = 1e4 * max(1.0, theta_k)
                theta_min   = 1e-4 * max(1.0, theta_k)
                filter_list = [(theta_max, -np.inf)]
                break
            
            # =============== Compute the search direction (A-4) ===============
            
            # build and solve KKT system (9)
            #           [ Hk   Ak^T   -I ] [ dx ]      [r_dual]
            #           [ Ak    0      0 ] [dlam]  =  -[r_pri ]
            #           [ Z     0      X ] [ dz ]      [r_cent]

            # eliminate dz, and solve the symmetric linear system (11):
            #           [ Hk + sigmak + delta_w*I      Ak^T   ] [ dx ] = -[grad_phi_mu(x) + A^T * lam]
            #           [            Ak            -delta_c*I ] [dlam]   -[           c(x)           ]
            #          
            # primal-dual barrier term Hessian approx: sigmak = X^-1 Z


            # ================= Inertia Correction (IC) =================
            delta_w = 0.0
            delta_c = 0.0

            def inertia_of_KKT(KKT_np):
                '''eigenvalues of symmetric matrix'''
                eigvals = np.linalg.eigvalsh(KKT_np)
                n_pos = np.sum(eigvals > tol)
                n_neg = np.sum(eigvals < -tol)
                n_zero = len(eigvals) - n_pos - n_neg
                return int(n_pos), int(n_neg), int(n_zero)

            # residuals of symmetric linear system
            r_top       = grad_barrier_phi(xk,mu) + p.A_e(xk).T @ lamk
            r_bottom    = p.c_e(xk)
            r           = -ca.vertcat(r_top,r_bottom)

            # Hessian, Sigma, A
            H = p.hess_L(xk, lamk, zk)
            sigmak = ca.diag(zk / xk)
            H_sigmak = H + sigmak
            A_e = p.A_e(xk)

            # build KKT Matrix
            KKT_top     = ca.hcat([H_sigmak + delta_w*ca.DM.eye(n), A_e.T])
            KKT_bottom  = ca.hcat([A_e, -delta_c*ca.DM.eye(m)])
            KKT         = ca.vertcat(KKT_top, KKT_bottom)
            KKT_np      = np.array(KKT)

            # ----------------------------- (IC-1) ---------------------------------
            n_pos, n_neg, n_zero = inertia_of_KKT(KKT_np)

            non_singular = True
            try:
                d = ca.solve(KKT, r)   # try to factorize
            except RuntimeError:
                non_singular = False

            if (non_singular) and (n_pos == n) and (n_neg == m) and (n_zero == 0):
                # good: step accepted (skip IC-2 - IC-6)
                dx  = d[0:n]
                dlam = d[n:]
            # ------------------------------------------------------------------------

            else:
                # ------------------------ (IC-2) ------------------------
                if n_zero > 0:
                    delta_c = delta_bar_c*(mu**kappa_c)
                else:
                    delta_c = 0.0
                # --------------------------------------------------------

                # ------------------------ (IC-3) ------------------------
                if delta_w_last == 0:
                    delta_w = delta_bar_w_0
                else:
                    delta_w = max(delta_bar_w_min, kappa_w_minus*delta_w_last)
                # --------------------------------------------------------

                # ------------------------ (IC-4) ------------------------
                while True:
                    KKT_top     = ca.hcat([H+sigmak + delta_w*ca.DM.eye(n), p.A_e(xk).T])
                    KKT_bottom  = ca.hcat([p.A_e(xk), -delta_c*ca.DM.eye(m)])
                    KKT         = ca.vertcat(KKT_top, KKT_bottom)
                    KKT_np      = np.array(KKT)

                    non_singular = True
                    try:
                        d = ca.solve(KKT, r)   # try to factorize
                    except RuntimeError:
                        non_singular = False

                    # inertia
                    n_pos, n_neg, n_zero = inertia_of_KKT(KKT_np)

                    if (non_singular) and (n_pos == n) and (n_neg == m) and (n_zero == 0):
                        # good: step accepted (skip IC-2 - IC-6)
                        delta_w_last = delta_w
                        dx  = d[0:n]
                        dlam = d[n:]
                        break
                    # --------------------------------------------------------

                    else:
                        # ------------------------ (IC-5) ------------------------
                        if delta_w_last == 0:
                            delta_w = kappa_bar_w_plus*delta_w
                        else:
                            delta_w = kappa_w_plus*delta_w
                        # --------------------------------------------------------

                        # ------------------------ (IC-6) ------------------------
                        if delta_w > delta_bar_w_max: # (IC-6)
                            print("delta_w > delta_bar_w_max => switching to restoration phase A-9 !")
                            raise RuntimeError("Restoration phase not implemented yet")
                        # --------------------------------------------------------

            # Recover dz (12)
            dz = mu * (1.0 / xk) - zk - sigmak @ dx


            # ---------------- Backtracking line-search (A-5) -------------
            # --- fraction-to-the-boundary rule for x, z (A-5.1 15a) ------------------
            alpha_max = 1.0
            alpha_z   = 1.0

            for i in range(n):
                while (xk[i]+alpha_max*dx[i] <= (1-tau)*xk[i]):
                    alpha_max = beta_ls*alpha_max

                while (zk[i]+alpha_z*dz[i] <= (1-tau)*zk[i]):
                    alpha_z = beta_ls*alpha_z

            alpha = alpha_max

            phi_k = barrier_phi(xk,mu)
            theta_k = theta(xk)

            # line search
            while True:

                # ---- compute new trial point (A-5.2) ----
                x_trial   = xk   + alpha * dx
                lam_trial = lamk + alpha * dlam
                z_trial   = zk   + alpha_z * dz

                # ---- check acceptability to the filter (A-5.3) ----
                theta_trial = theta(x_trial)
                phi_trial   = barrier_phi(x_trial, mu)

                # 1) filter acceptance: (theta_trial, phi_trial) not dominated by existing entries (A-5.3)
                dominated = False
                for (theta_i, phi_i) in filter_list:
                    if theta_trial >= theta_i and phi_trial >= phi_i:
                        dominated = True

                # 2) sufficient decrease vs current point (18)
                constraint_reduction = theta_trial <= (1.0 - gamma_theta) * theta_k
                objective_reduction  = phi_trial   <= phi_k - gamma_phi * theta_k

                if (dominated == False):

                    # ---- check sufficient decrease with respect to the current iterate (A-5.4) ----
                    theta_smaller_than_min = theta_trial <= theta_min
                    switching_condition_1 = grad_barrier_phi(xk,mu).T@dx < 0 # (19)
                    switching_condition_2 = alpha*(-grad_barrier_phi(xk,mu).T@dx)**s_phi > delta*theta(xk)**s_theta # (19)
                    armijo = barrier_phi(x_trial,mu) <= barrier_phi(xk,mu) +  nu_phi*alpha*grad_barrier_phi(xk,mu).T@dx # (20)
                    
                    # Case I:
                    if theta_smaller_than_min and switching_condition_1 and switching_condition_2:

                        if armijo: # (20)
                            # accept step (A-6)
                            xk   = x_trial
                            lamk = lam_trial
                            zk   = z_trial # rescaling (16) ignored
                            break

                        else:
                            # (A-5.5) init second-order correction
                            var = 0

                    # Case II:
                    elif (not theta_smaller_than_min) or (not switching_condition_1 or not switching_condition_2):
                        if constraint_reduction or objective_reduction: # (18)
                            # # accept step (A-6)
                            xk   = x_trial
                            lamk = lam_trial
                            zk   = z_trial # rescaling (16) ignored
                            break
                        else:
                            # (A-5.5) init second-order correction
                            soc = 0

                # --- new trial step size (A-5.10) ---
                alpha = 0.5*alpha
                if alpha <= alpha_min:
                    print("alpha <= alpha_min => switching to restoration phase A-9 !")
                    raise RuntimeError("Restoration phase not implemented yet")

            

            # ----- (A-7) -----
            phi_k = barrier_phi(xk,mu)
            theta_k = theta(xk)

            if (not armijo) or (not switching_condition_1) or (not switching_condition_2):
                # augment filter using (22)
                theta_new = (1.0 - gamma_theta) * theta_k
                phi_new   = phi_k - gamma_phi * theta_k
                filter_list.append((theta_new, phi_new))
            # else: leave filter_list unchanged

            x_path.append(xk)


        #                 finished inner loop for current mu 
        # ========================================================================

    print("max iterations reached")
    end = time.perf_counter() # timer end
    runtime = end - start
    return x_path, xk, runtime





def myIPOPT_debug(p,x0, maxj, maxk):
    """
    Same as myIPOPT but with more print statements for debuging (IC)

    todo: add print statements for the filter line search

    """
    start = time.perf_counter() # timer start
    n = p.n
    m = p.me
    e = ca.DM.ones(n, 1)
    tol = 1e-8
    # ----------- helper functions ------------------------------------------------- #
    def barrier_phi(x, mu):
        """Barrier objective phi_mu(x) = f(x) - mu sum(ln(x_i))."""
        return float(p.f(x) - mu * ca.sum1(ca.log(x)))
    
    def grad_barrier_phi(x, mu):
        """grad of barrier objective grad_phi_mu(x) = grad_f(x) - mu (1/x_i)."""
        return p.grad_f(x) - mu * (1.0/x)

    def theta(x):
        """Constraint violation θ(x) = ||c(x)||. """  
        return float(ca.norm_1(p.c_e(x))) 

    def optimality_error(x, lam, z, mu, return_components=False):
        """
        E_mu(x,lam,z) = max(||grad(f(x))+A*lam-z||inf, ||c(x)||inf, ||XZe-mu*e||inf)
        (no scaling parameters for now)
        """
        r_dual = p.grad_f(x) + p.A_e(x).T @ lam - z
        r_pri  = p.c_e(x)
        r_comp = ca.diag(x) @ ca.diag(z) @ e - mu * e

        e_dual = float(ca.norm_inf(r_dual))
        e_pri  = float(ca.norm_inf(r_pri))
        e_comp = float(ca.norm_inf(r_comp))

        if return_components:
            return e_dual, e_pri, e_comp

        return max(e_dual, e_pri, e_comp)

    def update_mu(mu):
        """Barrier update rule: mu_j+1 = max(mu_tol, kappa_mu * mu_j)"""
        return max(tol/10, min(kappa_mu * mu,mu**theta_mu))
    
    def frac_to_bound_tau(mu):
        """ fraction-to-the-boundrary parameter (8)"""
        return max(tau_min,1-mu)

    # ------------------- initialization -------------------

    mu          = 10    # not working for 0.1
    kappa_e     = 10
    kappa_mu    = 0.99  # should be 0.2 8but does not work with 0.2)
    theta_mu    = 1.5
    tau_min     = 0.99

    xk      = ca.DM(x0) #  x > 0
    lamk    = ca.DM.zeros(p.me)
    #zk      = ca.DM.ones(n, 1)
    zk    = mu / xk
    print(f"z = {zk}")

    # params for Line Search / Filter (Section 2.3)
    gamma_theta = 1e-5
    gamma_phi   = 1e-5
    gamma_alpha = 0.05

    alpha_min   = 1e-8
    beta_ls     = 0.5
    theta0      = theta(xk)              # <-- constraint violation
    phi0        = barrier_phi(xk, mu)    # <-- barrier function

    delta_bar_c         = 1e-8
    kappa_c             = 0.25
    delta_bar_w_0       = 1e-4
    delta_bar_w_min     = 1e-20
    delta_bar_w_max     = 1e40
    delta_w_last        = 0.0
    kappa_bar_w_plus    = 100.0
    kappa_w_plus        = 8.0
    kappa_w_minus       = 1.0/3.0

    theta_max = 1e4 * max(1.0, theta0)   # (from parameter list at end of section 2.5)
    theta_min = 1e-4 * max(1.0, theta0)

    s_phi = 2.3 # params for switching condition
    s_theta = 1.1
    delta = 1

    nu_phi = 1e-4
    kappa_soc = 0.99 # for second order correction (not implemented)
    p_max = 4


    # ======================== (A-1) =============================================

    tau = frac_to_bound_tau(mu)
    filter_list = [(theta_max, -np.inf)] # Initial filter F0 = { (theta, phi) : theta ≥ theta_max }

    x_path = [xk] # Path of iterates (for plotting)

    
    # ===================== Outer Loop (barrier problems - decreasing mu) =======================
    for j in range(maxj):

        e_dual, e_pri, e_comp = optimality_error(xk, lamk, zk, mu=0.0, return_components=True)
        print(f"j={j}: mu = {mu:.3e}, E_0 = {max(e_dual,e_pri,e_comp):.3e}, "
                f"dual={e_dual:.3e}, pri={e_pri:.3e}, comp={e_comp:.3e}, tol={tol:.3e}")
        
        # Check convergence for original problem: E_0 <= tol (A-2)
        if optimality_error(xk, lamk, zk, mu=0.0) <= tol:
            print("E_0 <= tol satisfied. Stop outer loop.")
            end = time.perf_counter() # timer end
            runtime = end - start
            return x_path, xk, runtime


        # ===================== Inner Loop (solve barrier problem for current mu) =====================
        for k in range(maxk):

            e_dual, e_pri, e_comp = optimality_error(xk, lamk, zk, mu, return_components=True)
            print(f"k={k}: mu = {mu:.3e}, E_mu = {max(e_dual,e_pri,e_comp):.3e}, "
                f"dual={e_dual:.3e}, pri={e_pri:.3e}, comp={e_comp:.3e}, kappa_e*mu={kappa_e*mu:.3e}")

            # Check convergence for current barrier problem: E_mu <= kappa_e*muj (A-3)
            if optimality_error(xk, lamk, zk, mu) <= kappa_e*mu:
                print("inner converged!!!!\n")
                # -------- update mu and tau (A-3.1) ---------
                mu = update_mu(mu)
                tau = frac_to_bound_tau(mu)

                # -------- re-initialize filter (A-3.2) --------
                theta_k     = theta(xk)
                theta_max   = 1e4 * max(1.0, theta_k)
                theta_min   = 1e-4 * max(1.0, theta_k)
                filter_list = [(theta_max, -np.inf)]
                break
            
            # =============== Compute the search direction (A-4) ===============
            
            # build and solve KKT system (9)
            #           [ Hk   Ak^T   -I ] [ dx ]      [r_dual]
            #           [ Ak    0      0 ] [dlam]  =  -[r_pri ]
            #           [ Z     0      X ] [ dz ]      [r_cent]

            # eliminate dz, and solve the symmetric linear system (11):
            #           [ Hk + sigmak + delta_w*I      Ak^T   ] [ dx ] = -[grad_phi_mu(x) + A^T * lam]
            #           [            Ak            -delta_c*I ] [dlam]   -[           c(x)           ]
            #          
            # primal-dual barrier term Hessian approx: sigmak = X^-1 Z


            # ================= Inertia Correction (IC) =================
            print("\n--- Inertia Correction (IC) ---")
            delta_w = 0.0
            delta_c = 0.0

            def inertia_of_KKT(KKT_np):
                '''eigenvalues of symmetric matrix'''
                eigvals = np.linalg.eigvalsh(KKT_np)
                n_pos = np.sum(eigvals > tol)
                n_neg = np.sum(eigvals < -tol)
                n_zero = len(eigvals) - n_pos - n_neg
                print(f"inertia = ({n_pos},{n_neg},{n_zero})")
                return int(n_pos), int(n_neg), int(n_zero)

            # residuals of symmetric linear system
            r_top       = grad_barrier_phi(xk,mu) + p.A_e(xk).T @ lamk
            r_bottom    = p.c_e(xk)
            r           = -ca.vertcat(r_top,r_bottom)
            print(f"\nr = {r}")

            # Hessian, Sigma, A
            H = p.hess_L(xk, lamk, zk)
            sigmak = ca.diag(zk / xk)
            H_sigmak = H + sigmak
            A_e = p.A_e(xk)
            print(f"\nH = {H}")
            print(f"\nsigmak = {sigmak}")
            print(f"\nH_sigmak = {H_sigmak}")
            print(f"\nA_e = {A_e}")

            # build KKT Matrix
            KKT_top     = ca.hcat([H_sigmak + delta_w*ca.DM.eye(n), A_e.T])
            KKT_bottom  = ca.hcat([A_e, -delta_c*ca.DM.eye(m)])
            KKT         = ca.vertcat(KKT_top, KKT_bottom)
            KKT_np      = np.array(KKT)
            print(f"\nKKT = {KKT}")

            # ----------------------------- (IC-1) ---------------------------------
            print("\n--> (IC-1)")
            n_pos, n_neg, n_zero = inertia_of_KKT(KKT_np)

            non_singular = True
            try:
                d = ca.solve(KKT, r)   # try to factorize
            except RuntimeError:
                non_singular = False
            print(f"non_singular = {non_singular}")

            if (non_singular) and (n_pos == n) and (n_neg == m) and (n_zero == 0):
                # good: step accepted (skip IC-2 - IC-6)
                print("\nstep accepted!")
                dx  = d[0:n]
                dlam = d[n:]
            # ------------------------------------------------------------------------

            else:
                print("\nstep rejected! -> continue Inertia Correction")

                # ------------------------ (IC-2) ------------------------
                print("\n--> (IC-2)")
                if n_zero > 0:
                    delta_c = delta_bar_c*(mu**kappa_c)
                else:
                    delta_c = 0.0
                print(f"delta_c = {delta_c}")
                # --------------------------------------------------------

                # ------------------------ (IC-3) ------------------------
                print("\n--> (IC-3)")
                if delta_w_last == 0:
                    delta_w = delta_bar_w_0
                else:
                    delta_w = max(delta_bar_w_min, kappa_w_minus*delta_w_last)
                print(f"delta_w_last = {delta_w_last}")
                print(f"delta_w = {delta_w}")
                # --------------------------------------------------------

                # ------------------------ (IC-4) ------------------------
                print("\n--> (IC-4)")
                while True:
                    KKT_top     = ca.hcat([H+sigmak + delta_w*ca.DM.eye(n), p.A_e(xk).T])
                    KKT_bottom  = ca.hcat([p.A_e(xk), -delta_c*ca.DM.eye(m)])
                    KKT         = ca.vertcat(KKT_top, KKT_bottom)
                    KKT_np      = np.array(KKT)
                    print(f"\nKKT Mat: {KKT}")

                    non_singular = True
                    try:
                        d = ca.solve(KKT, r)   # try to factorize
                    except RuntimeError:
                        non_singular = False
                    print(f"non_singular = {non_singular}")

                    # inertia
                    n_pos, n_neg, n_zero = inertia_of_KKT(KKT_np)

                    if (non_singular) and (n_pos == n) and (n_neg == m) and (n_zero == 0):
                        # good: step accepted (skip IC-2 - IC-6)
                        print("\nstep accepted!")
                        delta_w_last = delta_w
                        dx  = d[0:n]
                        dlam = d[n:]
                        break
                    # --------------------------------------------------------

                    else:
                        print("\nstep rejected! -> continue Inertia Correction")

                        # ------------------------ (IC-5) ------------------------
                        print("\n--> (IC-5)")
                        if delta_w_last == 0:
                            delta_w = kappa_bar_w_plus*delta_w
                        else:
                            delta_w = kappa_w_plus*delta_w
                        print(f"delta_w_last = {delta_w_last}")
                        print(f"delta_w = {delta_w}")
                        # --------------------------------------------------------

                        # ------------------------ (IC-6) ------------------------
                        print("\n--> (IC-6)")
                        if delta_w > delta_bar_w_max: # (IC-6)
                            print("delta_w > delta_bar_w_max => switching to restoration phase A-9 !")
                            raise RuntimeError("Restoration phase not implemented yet")
                        # --------------------------------------------------------

            # Recover dz (12)
            dz = mu * (1.0 / xk) - zk - sigmak @ dx

            # ---------------- Backtracking line-search (A-5) -------------
            # --- fraction-to-the-boundary rule for x, z (A-5.1 15a) ------------------
            alpha_max = 1.0
            alpha_z   = 1.0

            for i in range(n):
                while (xk[i]+alpha_max*dx[i] <= (1-tau)*xk[i]):
                    alpha_max = beta_ls*alpha_max

                while (zk[i]+alpha_z*dz[i] <= (1-tau)*zk[i]):
                    alpha_z = beta_ls*alpha_z

            alpha = alpha_max

            phi_k = barrier_phi(xk,mu)
            theta_k = theta(xk)

            # line search
            while True:

                # ---- compute new trial point (A-5.2) ----
                x_trial   = xk   + alpha * dx
                lam_trial = lamk + alpha * dlam
                z_trial   = zk   + alpha_z * dz

                # ---- check acceptability to the filter (A-5.3) ----
                theta_trial = theta(x_trial)
                phi_trial   = barrier_phi(x_trial, mu)

                # 1) filter acceptance: (theta_trial, phi_trial) not dominated by existing entries (A-5.3)
                dominated = False
                for (theta_i, phi_i) in filter_list:
                    if theta_trial >= theta_i and phi_trial >= phi_i:
                        dominated = True

                # 2) sufficient decrease vs current point (18)
                constraint_reduction = theta_trial <= (1.0 - gamma_theta) * theta_k
                objective_reduction  = phi_trial   <= phi_k - gamma_phi * theta_k

                if (dominated == False):

                    # ---- check sufficient decrease with respect to the current iterate (A-5.4) ----
                    theta_smaller_than_min = theta_trial <= theta_min
                    switching_condition_1 = grad_barrier_phi(xk,mu).T@dx < 0 # (19)
                    switching_condition_2 = alpha*(-grad_barrier_phi(xk,mu).T@dx)**s_phi > delta*theta(xk)**s_theta # (19)
                    armijo = barrier_phi(x_trial,mu) <= barrier_phi(xk,mu) +  nu_phi*alpha*grad_barrier_phi(xk,mu).T@dx # (20)
                    
                    # Case I:
                    if theta_smaller_than_min and switching_condition_1 and switching_condition_2:

                        if armijo: # (20)
                            # accept step (A-6)
                            xk   = x_trial
                            lamk = lam_trial
                            zk   = z_trial # rescaling (16) ignored
                            break

                        else:
                            # (A-5.5) init second-order correction
                            var = 0

                    # Case II:
                    elif (not theta_smaller_than_min) or (not switching_condition_1 or not switching_condition_2):
                        if constraint_reduction or objective_reduction: # (18)
                            # # accept step (A-6)
                            xk   = x_trial
                            lamk = lam_trial
                            zk   = z_trial # rescaling (16) ignored
                            break
                        else:
                            # (A-5.5) init second-order correction
                            soc = 0

                # --- new trial step size (A-5.10) ---
                alpha = 0.5*alpha
                if alpha <= alpha_min:
                    print("alpha <= alpha_min => switching to restoration phase A-9 !")
                    raise RuntimeError("Restoration phase not implemented yet")

            

            # ----- (A-7) -----
            phi_k = barrier_phi(xk,mu)
            theta_k = theta(xk)

            if (not armijo) or (not switching_condition_1) or (not switching_condition_2):
                # augment filter using (22)
                theta_new = (1.0 - gamma_theta) * theta_k
                phi_new   = phi_k - gamma_phi * theta_k
                filter_list.append((theta_new, phi_new))
            # else: leave filter_list unchanged

            x_path.append(xk)


        #                 finished inner loop for current mu 
        # ========================================================================

    print("max iterations reached")
    end = time.perf_counter() # timer end
    runtime = end - start
    return x_path, xk, runtime