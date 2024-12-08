import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from Data import investor_generation_data, DA_prices_3d, Wind_PF_data, Investment_data

# Set up data
N = 24  # Number of nodes
T = 24  # Number of time periods
Omega = 3  # Number of scenarios

# Parameters
phi = {w: 1 / Omega for w in range(Omega)}  # Scenario probabilities

lambda_nwt = {
    (n, w, t): DA_prices_3d[str(w)][t, n]
    for w in range(Omega)
    for t in range(T)
    for n in range(N)
}

CE = Wind_PF_data.iloc[:, 1].to_dict()  
QW_values = Wind_PF_data.iloc[:, 1].to_dict()
QW_max = {(t): QW_values[t] for t in range(T)}
PE_max = investor_generation_data.iloc[:, 2].to_dict()  
XW_max = 150  
Kmax = 1.6e8  ###### Change investment 
KW = Investment_data.iloc[2, 1] 

# Model
model = gp.Model("Optimization_Model")

# Variables
xW = model.addVars(N, name="xW", lb=0, ub=XW_max)
pW = model.addVars(N, Omega, T, name="pW", lb=0)
pE = model.addVars(N, Omega, T, name="pE", lb=0,
                   ub={(n): PE_max[n] for n in range(N)})

# Constraints to force xW and pW to only be nonzero for node 14
model.addConstrs((xW[n] == 0 for n in range(N) if n != 13), name="Fix_xW_others")
model.addConstrs((pW[n, w, t] == 0 for n in range(N) if n != 13 for w in range(Omega) for t in range(T)), name="Fix_pW_others")



model.addConstrs((xW[n] <= XW_max for n in range(N)), name="Wind_Capacity")

model.addConstr(gp.quicksum(KW * xW[n] for n in range(N)) <= Kmax, name="Total_Capacity")

model.addConstrs(
    (pW[n, w, t] <= QW_max[t] * xW[n] for n in range(N) for w in range(Omega) for t in range(T)),
    name="Wind_Production"
)

model.addConstrs(
    (pE[n, w, t] <= PE_max[n] for n in range(N) for w in range(Omega) for t in range(T)),
    name="Existing_Production"
)



# Objective function
objective =- 20*365*gp.quicksum(
    phi[w] * gp.quicksum(
        gp.quicksum(
            lambda_nwt[n, w, t] * (pW[n, w, t] + pE[n, w, t]) - pE[n, w, t] * CE[n]
            for n in range(N)
        )
        for t in range(T)
    )
    for w in range(Omega)
) + gp.quicksum(KW * xW[n] for n in range(N))

model.setObjective(objective, GRB.MINIMIZE)





# Optimize the model
model.optimize()

# Results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    # Since the objective function is currently minimized, a negative value indicates profit.
    # We'll treat the negative of the objective value as profit in dollars.
    profit = -model.objVal
    print(f"Profit: {profit} $")

    # xW is measured in MW
    for n in range(N):
        print(f"xW[{n}] = {xW[n].x} MW")
else:
    print("No optimal solution found.")
