import gurobipy as gp
from gurobipy import GRB
from Data import investor_generation_data, DA_prices_3d, Wind_PF_data, Investment_data
import matplotlib.pyplot as plt

# Set up data
N = 24  # Number of nodes
T = 24  # Number of time periods
Omega = 3  # Number of scenarios

# Parameters
phi = {w: 1 / Omega for w in range(Omega)}  # Scenario probabilities

lambda_nwt = {
    (n, w, t): 1 * DA_prices_3d[str(w)][t, n]
    for w in range(Omega)
    for t in range(T)
    for n in range(N)
}

CE = {n: 1 * Wind_PF_data.iloc[n, 1] for n in Wind_PF_data.index}

QW_values = Wind_PF_data.iloc[:, 1].to_dict()
QW_max = {t: QW_values[t] for t in range(T)}
PE_max = investor_generation_data.iloc[:, 2].to_dict()
XW_max = 150
Kmax = 1.6e8
KW = Investment_data.iloc[2, 1]

# Resultatlagring
lower_bounds = []
upper_bounds = []
xW_13_values = []  # Store xW[13] values across iterations

# **Master Problem**
master = gp.Model("Master Problem")
xW = master.addVars(N, lb=0, ub=XW_max, name="xW")
gamma = master.addVar(lb=-GRB.INFINITY, name="gamma")
epsilon = 0.001

# Initialize xW to start at 0
for n in range(N):
    xW[n].start = 0  # Explicitly set initial guess to 0

master.addConstrs((xW[n] >= epsilon for n in range(N) if n == 13), name="Positive_xW")
master.addConstrs((xW[n] == 0 for n in range(N) if n != 13), name="Fix_xW_others")
master.addConstr(gp.quicksum(KW * xW[n] for n in range(N)) <= Kmax, name="Total_Capacity")

master.addConstr(gamma >= 0, name="GammaLowerBound")
master.addConstr(gamma <= 1e6, name="GammaUpperBound")
master.setObjective(
    gp.quicksum(KW * xW[n] for n in range(N)) + gamma,
    GRB.MAXIMIZE
)

def solve_subproblem(xW_values):
    z_w = []  # Objective values for each scenario
    rho_w = []  # Dual values for each scenario

    for w in range(Omega):  # Loop over scenarios
        subproblem = gp.Model(f"Subproblem_{w}")
        subproblem.setParam('OutputFlag', 0)

        # Variables: Include scenario (Omega), node (N), and time (T)
        pW = subproblem.addVars(N, Omega, T, lb=0, name="pW")  # Wind production
        pE = subproblem.addVars(N, Omega, T, lb=0,
                                ub={(n, w, t): PE_max[n] for n in range(N) for w in range(Omega) for t in range(T)},
                                name="pE")  # Existing production

        # Objective
        subproblem.setObjective(
            20 * 365 * gp.quicksum(
                lambda_nwt[n, w, t] * (pW[n, w, t] + pE[n, w, t]) - pE[n, w, t] * CE[n]
                for n in range(N) for t in range(T)
            ),
            GRB.MAXIMIZE
        )

        # Constraints
        subproblem.addConstrs(
            (pW[n, w, t] == 0 for n in range(N) for t in range(T) if n != 13),
            name=f"Fix_pW_others"
        )
        subproblem.addConstrs(
            (pW[n, w, t] <= QW_max[t] * xW_values[n] for n in range(N) for t in range(T)),
            name=f"Wind_Production"
        )
        subproblem.addConstrs(
            (pE[n, w, t] <= PE_max[n] for n in range(N) for t in range(T)),
            name=f"Existing_Production"
        )

        # Solve subproblem
        subproblem.optimize()

        if subproblem.status != GRB.OPTIMAL:
            subproblem.write(f"subproblem_{w}.lp")
            if subproblem.status == GRB.INFEASIBLE:
                subproblem.computeIIS()
                subproblem.write(f"subproblem_{w}_iis.ilp")
            raise Exception(f"Subproblem {w} did not solve to optimality!")

        # Store objective value
        z_w.append(subproblem.ObjVal)

        # Dual sensitivities
        dual_values = {}
        for n in range(N):
            for t in range(T):
                constr_name = f"Wind_Production[{n},{w},{t}]"
                constr = subproblem.getConstrByName(constr_name)
                if constr is not None:
                    dual_values[(n, t)] = constr.Pi
                else:
                    print(f"Warning: Constraint {constr_name} not found! Setting Pi = 0.")
                    dual_values[(n, t)] = 0  # Default value if constraint not found
        rho_w.append(dual_values)

    return z_w, rho_w






max_iters = 100
epsilon = 1e-4

for iteration in range(max_iters):
    print(f"Starting iteration {iteration + 1}...")

    # Solve master problem
    master.optimize()
    if master.status == GRB.OPTIMAL:
        xW_values = [xW[n].x for n in range(N)]
        print(f"Master Objective: {master.ObjVal:.4f}")
        print(f"xW values: {xW_values}")

        # Store the value of xW[13] for this iteration
        xW_13_values.append(xW[13].x)
    else:
        break

    # Solve subproblems
    z_w, rho_w = solve_subproblem(xW_values)
    print(f"Subproblem Objectives (z_w): {z_w}")

    # Compute upper bound
    upper_bound_expr = gp.quicksum(KW * xW_values[n] for n in range(N)) + sum(phi[w] * z_w[w] for w in range(Omega))
    upper_bound_value = upper_bound_expr.getValue() if isinstance(upper_bound_expr, gp.LinExpr) else upper_bound_expr

    # Store bounds
    lower_bounds.append(master.ObjVal)
    upper_bounds.append(upper_bound_value)
    print(f"Lower Bound: {master.ObjVal:.4f}, Upper Bound: {upper_bound_value:.4f}")

    # Check for convergence
    if abs(master.ObjVal - upper_bound_value) < epsilon:
        print("Convergence achieved!")
        break

    # Add Benders cut
    gamma_cut = gp.quicksum(
        phi[w] * (
            z_w[w] + gp.quicksum(rho_w[w][(n, t)] * (xW[n] - xW_values[n]) for n in range(N) for t in range(T))
        )
        for w in range(Omega)
    )
    master.addConstr(gamma <= gamma_cut, name=f"BendersCut_{iteration}")

# Final results
print(f"Number of iterations: {iteration + 1}")
print(f"Final xW[13] value: {xW[13].x}")



# Plot xW[13] values over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(xW_13_values) + 1), xW_13_values, marker="o", color="blue", label="xW[13]")
plt.xlabel("Iteration")
plt.ylabel("xW[13] Value")
plt.title("Convergence of xW[13] Over Iterations")
plt.legend()
plt.grid()
plt.show()
