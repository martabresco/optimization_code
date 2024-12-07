# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:26:40 2024

@author: anjal
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from Data import investor_generation_data, DA_prices_3d, Wind_PF_data, Investment_data

# Basic parameters
N = 24   # Number of nodes
T = 24   # Number of time periods
Omega = 3  # Number of scenarios

# Scenario probabilities (assuming equal)
phi = {w: 1 / Omega for w in range(Omega)} 

# Extract data from provided DataFrames
generation_existing_cost = investor_generation_data["Bid price [$/MWh]"]
generation_capacity = investor_generation_data["Pmax [MW]"]
CE = Wind_PF_data.iloc[:, 1].to_dict()
KW = Investment_data.iloc[2, 1]

# Example: DA_prices_3d is a dictionary of scenario: numpy array (T x N)
# lambda_nwt indexing
lambda_nwt = {
    (n, w, t): DA_prices_3d[str(w)][t, n]
    for w in range(Omega)
    for t in range(T)
    for n in range(N)
}

#DA_prices_3d=lambda_nwt

def build_subproblem_variables(m, DA_prices_3d):
    variables = {}
    # Investment capacity variable
    variables['inv_cap_wind'] = m.addVar(lb=0, ub=250, name='Investment_Capacity_Wind')

    # Wind production variables
    variables['wind_production'] = {
        (w, t): m.addVar(lb=0, ub=250, name=f'wind_production_w{w}_t{t}')
        for w in range(Omega)
        for t in range(T)
    }

    # Existing production variables
    variables['existing_production'] = {
        (n, w, t): m.addVar(lb=0, ub=generation_capacity[n], 
                            name=f'existing_production_n{n}_w{w}_t{t}')
        for n in range(N)
        for w in range(Omega)
        for t in range(T)
    }

    m.update()
    return variables

def build_subproblem_objective(m, variables, DA_prices_3d, generation_existing_cost):
    # Objective: minimize costs
    obj = 20 * 365 * gp.quicksum(
        gp.quicksum(
            gp.quicksum(
                (DA_prices_3d[str(w)][t, n] * ((variables['wind_production'][w, t] if n == 13 else 0)
                 + variables['existing_production'][n, w, t]) 
                - variables['existing_production'][n, w, t] * generation_existing_cost[n])
                for n in range(N)
            )
            for t in range(T)
        )
        for w in range(Omega)
    )
    m.setObjective(obj, GRB.MINIMIZE)
    m.update()


def build_subproblem_constraints(m, variables, DA_prices_3d):
    constraints = {}
    # Wind upper limit
    constraints['wind_upper_limit'] = m.addConstrs(
        (variables['wind_production'][w, t] <= 250 
         for w in range(Omega) for t in range(T)),
        name='wind_upper_limit'
    )

    # Existing production upper limit
    constraints['existing_upper_limit'] = m.addConstrs(
        (variables['existing_production'][n, w, t] <= generation_capacity[n]
         for n in range(N) for w in range(Omega) for t in range(T)),
        name='existing_upper_limit'
    )
    m.update()
    return constraints

def build_master_variables(m, benders_type, DA_prices_3d):
    variables = {}
    # Investment variable in master
    variables['inv_cap_wind'] = m.addVar(lb=0, ub=250, name='Inv_Cap_Wind')

    if benders_type == 'uni-cut':
        variables['gamma'] = m.addVar(lb=-1000, name='gamma')
    elif benders_type == 'multi-cut':
        variables['gamma'] = {
            scenario: m.addVar(lb=-1000, name=f'gamma_{scenario}')
            for scenario in range(Omega)
        }

    m.update()
    return variables

def build_master_objective(m, variables, benders_type, DA_prices_3d):
    # Objective for master problem: Investment cost minus approximation of subproblem costs
    if benders_type == 'uni-cut':
        objective = Investment_data.iloc[2, 1] * variables['inv_cap_wind'] - variables['gamma']
    else:  # multi-cut
        # probability_scenario might be defined, assuming equal probabilities here
        probability_scenario = phi  # If different, adjust accordingly
        objective = (Investment_data.iloc[2, 1] * variables['inv_cap_wind'] 
                     - gp.quicksum(probability_scenario[w] * variables['gamma'][w]
                                   for w in range(Omega)))
    m.setObjective(objective, GRB.MINIMIZE)
    m.update()

def build_master_constraints(m, variables):
    K = 5e10
    constraints = {}
    constraints['DA_balance_constraint'] = m.addConstr(
        Investment_data.iloc[2, 1] * variables['inv_cap_wind'] <= K, name='max_investment'
    )
    constraints['master_cuts'] = {}
    m.update()
    return constraints

def save_master_data(master):
    # Placeholder for saving master data
    pass

def save_subproblem_data(subproblem, w):
    # Placeholder for saving subproblem data
    pass

def update_upper_lower_bounds(master, subproblems, DA_prices_3d, probability_scenario):
    # Placeholder for updating bounds
    pass

def solve_benders_step(master, subproblems, DA_prices_3d, probability_scenario, epsilon, max_iters):
    m = master['model']
    m.setParam('Presolve', 2)
    m.setParam('Cuts', 2)

    master['iteration'] += 1
    m.optimize()
    if m.status == GRB.OPTIMAL:
        print(f"Master problem solved at iteration {master['iteration']}. Inv_Cap_Wind: {master['variables']['inv_cap_wind'].x}")
    elif m.status == GRB.INFEASIBLE:
        print("Master problem infeasible.")
    else:
        print("Master problem unbounded or another issue occurred.")

    save_master_data(master)

    # Solve subproblems
    for w in range(Omega):
        subproblem = subproblems[w]
        subproblem['model'].optimize()
        save_subproblem_data(subproblem, w)

    update_upper_lower_bounds(master, subproblems, DA_prices_3d, probability_scenario)


# Main execution
benders_type = 'multi-cut'  # For example, choose multi-cut
master = {}
master['model'] = gp.Model(name='master')
master['iteration'] = 0
master['variables'] = build_master_variables(master['model'], benders_type=benders_type, DA_prices_3d=DA_prices_3d)
master['constraints'] = build_master_constraints(master['model'], master['variables'])
build_master_objective(master['model'], master['variables'], benders_type, DA_prices_3d)

subproblems = {}
for w in range(Omega):
    subproblem = {}
    subproblem['model'] = gp.Model(name=f'subproblem_{w}')
    subproblem['variables'] = build_subproblem_variables(subproblem['model'], DA_prices_3d)
    build_subproblem_objective(subproblem['model'], subproblem['variables'], DA_prices_3d, generation_existing_cost)
    subproblem['constraints'] = build_subproblem_constraints(subproblem['model'], subproblem['variables'], DA_prices_3d)
    subproblems[w] = subproblem

epsilon = 1e-6
max_iters = 100

solve_benders_step(master, subproblems, DA_prices_3d, phi, epsilon=epsilon, max_iters=max_iters)
