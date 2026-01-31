import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import fsolve

def plot_nullclines():
    alpha1 = alpha2 = 10
    beta = gamma = 2
    
    u1, u2 = sm.symbols('u1 u2', negative=False, real=True)
    du1 = alpha1 / (1+u2**beta) - u1
    du2 = alpha2 / (1+u1**gamma) - u2
    
    du1_eq_0 = sm.Eq(du1, 0)
    du2_eq_0 = sm.Eq(du2, 0)
    
    # get steady-states
    ss = sm.solve([du1_eq_0, du2_eq_0], [u1, u2], dict=True)
    print(f'There are {len(ss)} steady states possible for this system.')
    
    du1_nullcline = sm.solve(du1_eq_0, u1)[0]
    du2_nullcline = sm.solve(du2_eq_0, u1)[0]
    
    # vectorize
    f_null1 = sm.lambdify(u2, du1_nullcline, 'numpy')
    f_null2 = sm.lambdify(u2, du2_nullcline, 'numpy')
    f_du1 = sm.lambdify((u1, u2), du1, 'numpy')
    f_du2 = sm.lambdify((u1, u2), du2, 'numpy')
    
    u2_vec = np.linspace(0.01, 12, 200)
    u1_from_null1 = f_null1(u2_vec)
    u1_from_null2 = np.real(f_null2(u2_vec.astype(complex))) 
    
    # to draw arrows
    U2_m, U1_m = np.meshgrid(np.linspace(0, 12, 25), np.linspace(0, 12, 25))
    dU1_m = f_du1(U1_m, U2_m)
    dU2_m = f_du2(U1_m, U2_m)
    
    # plotting
    plt.figure(figsize=(8, 6))
    plt.plot(u2_vec, u1_from_null1, label='u1 nullcline', lw=2)
    plt.plot(u2_vec, u1_from_null2, label='u2 nullcline', lw=2)
    
    # steady-states 
    ss_u1 = []
    ss_u2 = []
    
    for point in ss:
        if point[u1].is_real and point[u2].is_real:
            ss_u1.append(float(point[u1]))
            ss_u2.append(float(point[u2]))
    
    # plot
    plt.plot(ss_u2, ss_u1, 'go', markersize=8, label="steady state")
    plt.quiver(U2_m, U1_m, dU2_m, dU1_m, color='gray', alpha=0.4)
    plt.xlabel('u2')
    plt.ylabel('u1')
    plt.ylim(0, 12)
    plt.xlim(0, 12)
    plt.legend()
    plt.title('Bistable System: Nullclines and Steady States')
    plt.savefig('nullclines.png')
    plt.close()

def plot_bistability():
    # setup symbols
    u1, u2 = sm.symbols('u1 u2', negative=False, real=True)
    a1, a2, b, g = sm.symbols('alpha1 alpha2 beta gamma', real=True)
    
    # ODEs and J
    du1 = a1 / (1 + u2**b) - u1
    du2 = a2 / (1 + u1**g) - u2
    J_sym = sm.Matrix([[sm.diff(du1, u1), sm.diff(du1, u2)],
                       [sm.diff(du2, u1), sm.diff(du2, u2)]])
    
    # vectorize
    f_ode = sm.lambdify((u1, u2, a1, a2, b, g), [du1, du2], 'numpy')
    f_jac = sm.lambdify((u1, u2, a1, a2, b, g), J_sym, 'numpy')
    
    # get is stable or bistable
    def get_bistability_status(a1, a2, beta_val=2, gamma_val=2):
    
        # just give ode output
        def system_ode(u):
            return f_ode(u[0], u[1], a1, a2, beta_val, gamma_val)
    
        # get ss
        found_ss = []
        starts = np.arange(0, 21, 5) 
        for i in starts:
            for j in starts:
                res, _, ier, _ = fsolve(system_ode, [i, j], full_output=True)
                if ier == 1 and np.all(res >= -1e-6):
                    found_ss.append(np.round(np.maximum(res, 0), 3))
        
        if not found_ss: return 0
        
        # unique steady states
        unique_ss = np.unique(found_ss, axis=0)
        
        stable_count = 0
        eps = 1e-9
    
        for ss in unique_ss:
            jac_mat = np.array(f_jac(ss[0] + eps, ss[1] + eps, a1, a2, beta_val, gamma_val))
            if np.all(np.isfinite(jac_mat)):
                eigs = np.linalg.eigvals(jac_mat)
                # point is stable if all eigenvalue real parts are negative
                if np.all(np.real(eigs) < 0):
                    stable_count += 1
                
        # here: bistability is defined by having 2 stable steady states
        return 1 if stable_count == 2 else 0
    
    # plot
    alpha_values = np.arange(0, 21, 5) 
    N = len(alpha_values)
    SWITCH = np.zeros((N, N))
    
    for i, a1_val in enumerate(alpha_values):
        for j, a2_val in enumerate(alpha_values):
            SWITCH[i, j] = get_bistability_status(a1_val, a2_val)
    
    plt.figure(figsize=(7, 6))
    img = plt.imshow(SWITCH, origin='lower', extent=[0, 20, 0, 20], interpolation='nearest')
    
    patches = [
        mpatches.Patch(color=img.cmap(img.norm(0)), label='Without Switch dynamics'),
        mpatches.Patch(color=img.cmap(img.norm(1)), label='With Switch dynamics')
    ]
    
    plt.legend(handles=patches, loc='upper right')
    plt.xlabel('alpha2')
    plt.ylabel('alpha1')
    plt.title('System dynamics given different alpha1 & alpha2 values')
    plt.savefig("bistability.png")
    plt.close()

plot_nullclines()
plot_bistability()


