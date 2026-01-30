import numpy as np
import sympy
from scipy.optimize import linprog

stoichiometry = np.array([[1, 0, 0, 0, -1, -1, -1, 0, 0, 0], 
                          [0, 1, 0, 0, 1, 0, 0, -1, -1, 0], 
                          [0, 0, 0, 0, 0, 1, 0, 1, 0, -1], 
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, -1], 
                          [0, 0, 0, -1, 0, 0, 0, 0, 0, 1], 
                          [0, 0, -1, 0, 0, 0, 0, 0, 1, 1]])

basis = sympy.Matrix(stoichiometry).nullspace()

# max r3 -> min -r3
c = np.array([0,0,-1,0,0,0,0,0,0,0])
A_ub = -1 * np.identity(10)
b_ub = np.zeros(10)
# R2 and R8 are reversible
A_ub[1,1] = 0
A_ub[7,7] = 0
# equality constraints
A_eq = np.concatenate((stoichiometry, np.array([[1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0]])), axis = 0)
b_eq = np.array([0,0,0,0,0,0,1,0])

opt_flux = linprog(c, A_ub, b_ub, A_eq, b_eq)['x']
print(opt_flux)

init_1 = np.array([1, 0, 0.5, 0.5, 0, 0.5, 0.5, 0, 0, 0.5])
init_2 = np.array([1, 0, 0.7, 0.3, 0.3, 0.4, 0.3, -0.1, 0.4, 0.3])

opt_flux = linprog(c, A_ub, b_ub, A_eq, b_eq, x0=init_1, method='revised simplex')['x']
print(opt_flux)

opt_flux = linprog(c, A_ub, b_ub, A_eq, b_eq, x0=init_2, method='revised simplex')['x']
print(opt_flux)

opt_results = []
new_b_eq = np.concatenate((b_eq, [0]))
for ko in range(10):
    new_A_eq = np.concatenate((A_eq, np.array([[i==ko for i in range(10)]], dtype = 'float')), axis = 0)
    opt_flux = linprog(c, A_ub, b_ub, new_A_eq, new_b_eq)['x']
    if not opt_flux is None:
        print(f"gene: {ko+1} | yield: {opt_flux[2]} | byproduct: {opt_flux[3]}")
