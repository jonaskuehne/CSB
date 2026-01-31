import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

index_reaction_dict = {
    0: 'MKKK',
    1: 'MKKKp',
    2: 'MKK',
    3: 'MKKp',
    4: 'MKKpp',
    5: 'MAPK',
    6: 'MAPKp',
    7: 'MAPKpp',
}

N = np.array([
    [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0,-1, 0, 0, 1, 0, 0, 0, 0],
    [ 0, 0, 1,-1, 1,-1, 0, 0, 0, 0],
    [ 0, 0, 0, 1,-1, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0,-1, 0, 0, 1],
    [ 0, 0, 0, 0, 0, 0, 1,-1, 1,-1],
    [ 0, 0, 0, 0, 0, 0, 0, 1,-1, 0],
])

V1=2.5
V2=0.25
V5=V6=0.75
V9=V10=0.5
k3=k4=k7=k8=0.025
K1=10
K2=8
K3=K4=K5=K6=K7=K8=K9=K10=15
KI=9

initial_c = [100, 0, 300, 0, 0, 300, 0, 0]

def calculate_rates(c, V1=V1):
    return [
        V1*c[0]/(K1+c[0]),
        V2*c[1]/(K2+c[1]),
        k3*c[1]*c[2]/(K3+c[2]),
        k4*c[1]*c[3]/(K4+c[3]),
        V5*c[4]/(K5+c[4]),
        V6*c[3]/(K6+c[3]),
        k7*c[4]*c[5]/(K7+c[5]),
        k8*c[4]*c[6]/(K8+c[6]),
        V9*c[7]/(K9+c[7]),
        V10*c[6]/(K10+c[6]),
    ]

def calculate_rates_I(c):
    return [
        # change this one for inhibition
        V1*c[0]/(K1+c[0])*(KI/(KI+c[7])),
        V2*c[1]/(K2+c[1]),
        k3*c[1]*c[2]/(K3+c[2]),
        k4*c[1]*c[3]/(K4+c[3]),
        V5*c[4]/(K5+c[4]),
        V6*c[3]/(K6+c[3]),
        k7*c[4]*c[5]/(K7+c[5]),
        k8*c[4]*c[6]/(K8+c[6]),
        V9*c[7]/(K9+c[7]),
        V10*c[6]/(K10+c[6]),
    ]

def model(_, c, V1=V1, use_I=False):
    if use_I:
        rates = calculate_rates_I(c)
    else:
        rates = calculate_rates(c, V1)

    return N @ rates

def terminal_event(func):
    func.terminal = True
    return func

@terminal_event
def steady_state_event(t, c, V1=V1):
    # check total change
    return np.sum(np.abs(model(t, c, V1))) - 1e-3

def std_sim():
    sol = solve_ivp(
            model, 
            t_span=(0, 1000), 
            y0=initial_c, 
        )
    plt.xlabel('time (s)')
    plt.ylabel('concentration (nM)')
    
    for i in range(len(N)):
        plt.plot(sol.t, sol.y[i], label=index_reaction_dict[i])
    
    plt.legend(loc="right")
    plt.savefig("sim.png")
    plt.close()

def ss_sim():
    V_input = np.linspace(0, 0.5, 20)
    MAPKpp = []
    
    for v1 in V_input:
        sol = solve_ivp(
            model, 
            t_span=(0, 100000), 
            y0=initial_c, 
            args=(v1,), 
            events=steady_state_event,
        ) 
        c_MAPKpp = sol.y[7][-1]
        MAPKpp.append(c_MAPKpp)
    
    plt.xlabel('V1 (nM/s)')
    plt.ylabel('MAPKpp (nM)')
    plt.plot(V_input, MAPKpp)
    plt.savefig("ss.png")
    plt.close()

def inhib_sim():
    sol = solve_ivp(
            model, 
            t_span=(0, 6000), 
            y0=initial_c, 
            args=(V1, True), 
        )
    plt.xlabel('time (s)')
    plt.ylabel('concentration (nM)')
    
    for i in range(len(N)):
        plt.plot(sol.t, sol.y[i], label=index_reaction_dict[i])
    
    plt.legend(loc="right")
    plt.savefig("sim_I.png")
    plt.close()

std_sim()
ss_sim()
inhib_sim()
