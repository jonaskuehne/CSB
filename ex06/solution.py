import numpy as np
import sympy as sm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

index_reaction_dict = {
    0: 'K1',
    1: 'K1p',
}

N = np.array([
    [-1, -1,  1],
    [ 1,  1, -1],
])

k_K1 = k_K2 = 30
k_P = 3
k_P2 = 0.75 * 1e8
K_mK1 = K_mK2 = 1e-6
K_mP = 2.5 * 1e-9
c_K1_tot = 5 * 1e-8
c_P = 5 * 1e-9

# no negative concentrations
cK1, cK2, cK1P = sm.symbols('cK1 cK2 cK1P', negative=False)

# kinase 1
r_K1 = k_K1 * cK1P * cK1/(K_mK1 + cK1)
# kinase 2
r_K2 = k_K2 * cK2 * cK1/(K_mK2 + cK1)
# michaelis-menten dynamics
r_K1_P1 = k_P * c_P * cK1P/(K_mP + cK1P)
# mass action dynamics
r_K1_P2 = k_P2 * c_P * cK1P

def plot_rates(ma=False):
    # conservation relation: c_K1_tot = c_K1 + c_K1_P
    r_K1_func = sm.lambdify(cK1P, r_K1.subs(cK1, c_K1_tot - cK1P), 'numpy')
    if ma:
        # michaelis-menten dynamics 
        r_P_mm_func = sm.lambdify(cK1P, r_K1_P1, 'numpy')
    else:
        r_P_mm_func = sm.lambdify(cK1P, r_K1_P2, 'numpy')

    # input
    c_K1_P_input = np.linspace(0, c_K1_tot, 20)
    # calc
    r_K1_output = r_K1_func(c_K1_P_input)
    r_P_output = r_P_mm_func(c_K1_P_input)

    # plot
    plt.xlabel('K1_P concentration (M)')
    plt.ylabel('Rates (M/s)')
    plt.plot(c_K1_P_input, r_K1_output, label='K1 rate')
    plt.plot(c_K1_P_input, r_P_output, label='K1_P rate')
    plt.legend(loc="best")
    if ma:
        plt.savefig("rates_ma.png")
    else:
        plt.savefig("rates_mm.png")
    plt.close()

def get_ss():
    # concentration change reactions assuming michaelis-menten dynamics
    dc_K1 = -r_K1 + r_K1_P1
    dc_K1_P = r_K1 - r_K1_P1
    
    dc_K1_eq_0 = sm.Eq(dc_K1, 0)
    dc_K1_P_eq_0 = sm.Eq(dc_K1_P, 0)
    
    # again conservation
    dc_K1_eq_0 = dc_K1_eq_0.subs(cK1, c_K1_tot - cK1P)
    dc_K1_P_eq_0 = dc_K1_P_eq_0.subs(cK1, c_K1_tot - cK1P)
    
    # solve 
    c_K1_nullcline = sm.solve(dc_K1_eq_0, cK1P)
    c_K1_P_nullcline = sm.solve(dc_K1_P_eq_0, cK1P)
    
    print(f"c_K1 nullcline => c_K1_P = {c_K1_nullcline}")
    print(f"c_K1_P nullcline => c_K1_P = {c_K1_P_nullcline}")

# vectorize
calc_r_K1 = sm.lambdify((cK1, cK1P), r_K1, 'numpy')
calc_r_K2 = sm.lambdify((cK1, cK2), r_K2, 'numpy')
calc_r_P_mm = sm.lambdify(cK1P, r_K1_P1, 'numpy')
calc_r_P_ma = sm.lambdify(cK1P, r_K1_P2, 'numpy')

# rate laws
def get_rates(c, c_K2, type='mm'):
    v_k1 = calc_r_K1(c[0], c[1])
    v_k2 = calc_r_K2(c[0], c_K2)
    v_p  = calc_r_P_mm(c[1]) if type == 'mm' else calc_r_P_ma(c[1])
    return np.array([v_k1, v_k2, v_p])

# model def
def model(_, c, c_K2, type):
    rates = get_rates(c, c_K2, type)
    return N @ rates

def run_segment(y0, t_start, duration, c_K2, type):
    t_span = (t_start, t_start + duration)
    sol = solve_ivp(model, t_span, y0, args=(c_K2, type), 
                    method="Radau", atol=1e-10)
    return sol.t, sol.y

def sim_switch():
    K2_signals = np.linspace(0, 7e-9, 10)
    duration = 20
    results = {'mm': {'t': [], 'K1': [], 'KP': [], 'K2': []}, 
               'ma': {'t': [], 'K1': [], 'KP': [], 'K2': []}}
    
    # initial conditions
    y0_dict = {'mm': np.array([c_K1_tot, 0.0]), 'ma': np.array([c_K1_tot, 0.0])}
    t_current = 0
    
    # split signals
    for sig in K2_signals:
        for mode in ['mm', 'ma']:
            # signal on
            t_on, y_on = run_segment(y0_dict[mode], t_current, duration, sig, mode)
            
            # signal off -> start with last point in on
            t_off, y_off = run_segment(y_on[:, -1], t_on[-1], duration, 0, mode)
    
            # store
            results[mode]['t'].extend(np.concatenate([t_on, t_off]))
            results[mode]['K1'].extend(np.concatenate([y_on[0], y_off[0]]))
            results[mode]['KP'].extend(np.concatenate([y_on[1], y_off[1]]))
            results[mode]['K2'].extend([sig]*len(t_on) + [0]*len(t_off))
            
            # update init cond
            y0_dict[mode] = y_off[:, -1]
            
        t_current = results['mm']['t'][-1]
    
    # plot
    _, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    titles = ['Michaelis-Menten Dynamics', 'Mass Action Dynamics']
    
    for ax, mode, title in zip(axes, ['mm', 'ma'], titles):
        res = results[mode]
        ax.plot(res['t'], res['K1'], label='$c_{K1}$')
        ax.plot(res['t'], res['KP'], label='$c_{KP}$')
        ax.step(res['t'], res['K2'], where='post', label='$c_{K2}$ (Input)', alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel('Concentration (M)')
        ax.legend()
    
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig("switch_sim.png")
    plt.close()

plot_rates(ma=False)
get_ss()
plot_rates(ma=True)
sim_switch()




