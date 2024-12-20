import gurobipy as gb
import pandas as pd
import matplotlib.pyplot as plt
import timeit

# Load required data
from data_benders import (
    investor_generation_data_d,
    pv_PF,
    investment_data,
    DA_prices
)

# Parameters
SCENARIOS = list(range(20))  # PV production scenarios
probability_scenario = [0.05] * 20

invested_node = 18
max_investment_cap = 250
budget = 2.00e8
discount_rate = 0.05
lifetime_years = 20

# Data
generation_existing_cost = investor_generation_data_d["Bid price [$/MWh]"]
generation_capacity = investor_generation_data_d["Pmax [MW]"]
PF_PV = pv_PF.iloc[:, 1]


############ Solve and print results with fixed budget ##############


# Model
model = gb.Model("original_model")

# Decision variables
xP = model.addVar(lb=0, ub=max_investment_cap, name="Investment_Capacity_PV")
pP = model.addVars(24, 24, lb=0, name="PV_Production")
pE = model.addVars(
    range(24), range(24), 
    lb=0, 
    ub={n: generation_capacity[n] for n in range(24)},  # Node-specific upper bounds
    name="Existing_Production"
)

# Objective
objective = investment_data.iloc[1,1]*xP-( gb.quicksum(
    (1 / (1 + discount_rate) ** t) * (
        365 * gb.quicksum(
            gb.quicksum(
                DA_prices[scenario, h, n] * (pP[n, h] + pE[n, h])
                for n in range(24)
            )
            for h in range(24)
        ) -
        365 * gb.quicksum(
            gb.quicksum(
                pE[n, h] * generation_existing_cost[n]
                for n in range(24)
            )
            for h in range(24)
        )
    )
    for t in range(1, lifetime_years + 1)
    for scenario in SCENARIOS
))
model.setObjective(objective, gb.GRB.MINIMIZE)

# Constraints
# Investment capacity constraint
model.addConstr(xP <= max_investment_cap, name="Max_Investment_Capacity")

# Budget constraint
model.addConstr(investment_data.iloc[1, 1] * xP <= budget, name="Budget_Constraint")

# PV production constraint
model.addConstrs(
    (pP[n, t] == 0 for n in range(24) if n != invested_node for t in range(24)),
    name="Zero_Production_Outside_Invested_Node"
)
model.addConstrs(
    (pP[n, t] <= xP * PF_PV[t] for n in range(24) for t in range(24)),
    name="PV_Production_Limit"
)


# Existing production constraint
model.addConstrs(
    (pE[n, t] <= generation_capacity[n] for n in range(24) for t in range(24)),
    name="Existing_Production_Limit"
)

# Solve
start = timeit.default_timer()
model.optimize()
end = timeit.default_timer()

# Results
if model.status == gb.GRB.OPTIMAL:
    print(f"Optimal Investment Capacity (MW): {xP.x}")
    print(f"Profit: {abs(model.ObjVal) / 1e6} mill. $")
else:
    print("No optimal solution found.")




##################  Results when varying budget #########################


# # Define the range of budget values
# K_values = [ 1.50e6, 1.00e7, 1.50e7, 1.00e8, 2.00e8, 3.00e8, 4.00e8, 5.00e8, 6.00e8, 7.00e8, 8e8]

# # Results storage
# results = []

# for budget in K_values:
#     # Update the budget
#     Budget = budget
#     print(f"Running optimization for Budget: {Budget} $")

#     try:
#         # Model
#         model = gb.Model("original_model")

#         # Decision variables
#         xP = model.addVar(lb=0, ub=max_investment_cap, name="Investment_Capacity_PV")
#         pP = model.addVars(24, 24, lb=0, name="PV_Production")
#         pE = model.addVars(
#             range(24), range(24), 
#             lb=0, 
#             ub={n: generation_capacity[n] for n in range(24)},  # Node-specific upper bounds
#             name="Existing_Production"
#         )

#         # Objective
#         objective = investment_data.iloc[1,1]*xP-gb.quicksum(
#             (1 / (1 + discount_rate) ** t) * (
#                 365 * gb.quicksum(
#                     gb.quicksum(
#                         DA_prices[scenario, h, n] * (pP[n, h] + pE[n, h])
#                         for n in range(24)
#                     )
#                     for h in range(24)
#                 ) -
#                 365 * gb.quicksum(
#                     gb.quicksum(
#                         pE[n, h] * generation_existing_cost[n]
#                         for n in range(24)
#                     )
#                     for h in range(24)
#                 )
#             )
#             for t in range(1, lifetime_years + 1)
#             for scenario in SCENARIOS
#         )
#         model.setObjective(objective, gb.GRB.MINIMIZE)

#         # Constraints
#         # Investment capacity constraint
#         model.addConstr(xP <= max_investment_cap, name="Max_Investment_Capacity")

#         # Budget constraint
#         model.addConstr(investment_data.iloc[1, 1] * xP <= Budget, name="Budget_Constraint")

#         # PV production constraint
#         model.addConstrs(
#             (pP[n, t] == 0 for n in range(24) if n != invested_node for t in range(24)),
#             name="Zero_Production_Outside_Invested_Node"
#         )
#         model.addConstrs(
#             (pP[n, t] <= xP * PF_PV[t] for n in range(24) for t in range(24)),
#             name="PV_Production_Limit"
#         )

#         # Existing production constraint
#         model.addConstrs(
#             (pE[n, t] <= generation_capacity[n] for n in range(24) for t in range(24)),
#             name="Existing_Production_Limit"
#         )

#         # Solve
#         start = timeit.default_timer()
#         model.optimize()
#         end = timeit.default_timer()

#         # Results storage
#         if model.status == gb.GRB.OPTIMAL:
#             results.append({
#                 "Budget": Budget,
#                 "Optimal Cost": model.ObjVal,
#                 "Investment Capacity (MW)": xP.x,
#                 "Solving Time (s)": end - start
#             })
#         else:
#             results.append({
#                 "Budget": Budget,
#                 "Optimal Cost": None,
#                 "Investment Capacity (MW)": None,
#                 "Solving Time (s)": end - start
#             })

#     except Exception as e:
#         print(f"Optimization failed for Budget: {Budget} with error: {e}")
#         results.append({
#             "Budget": Budget,
#             "Optimal Cost": None,
#             "Investment Capacity (MW)": None,
#             "Solving Time (s)": None,
#             "Error": str(e)
#         })

# # Convert results to DataFrame
# results_df = pd.DataFrame(results)

# # Save the DataFrame to an Excel file
# results_df.to_excel("results_without_decomposition.xlsx", index=False)



