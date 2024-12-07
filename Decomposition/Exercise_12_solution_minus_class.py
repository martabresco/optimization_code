
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:29:44 2023

@author: lesiamitridati
"""


import gurobipy as gb
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
import timeit
from Data import investor_generation_data
from Data import DA_prices_3d, probability_scenario, Wind_PF_data, Investment_data


# Set plot parameters
sb.set_style('ticks')
size_pp = 15
font = {'family': 'times new roman',
        'color':  'black',
        'weight': 'normal',
        'size': size_pp,
        }


# Define ranges and hyperparameters  
BENDERS_TYPES = ['unit-cut', 'multi-cut']  # types of Benders algorithm


# Set values of input parameters
generation_existing_cost = investor_generation_data["Bid price [$/MWh]"] 
generation_capacity =  investor_generation_data["Pmax [MW]"]  # Generators capacity (Q^G_i)


# Helper functions for the model

def build_subproblem_variables(m, DA_prices_3d):
    """Builds subproblem variables"""
    variables = {}

    # complicating variables
    variables['inv_cap_wind'] = m.addVar(lb=0, ub=250, name='Investment Capacity Wind')

    # Subproblem variables
    variables['wind_production'] = {
        (w, t): m.addVar(lb=0, ub=250, name=f'wind production for scenario {w} in hour {t}')
        for w in range(len(DA_prices_3d))
        for t in range(24)  # 0 to 23 inclusive, so use range(24)
    }

    variables['existing_production'] = {
        (n, w, t): m.addVar(lb=0, ub=generation_capacity[n], name=f'existing production for node {n}, scenario {w} in hour {t}')
        for n in range(24)  # Assuming there are 24 nodes
        for w in range(len(DA_prices_3d))
        for t in range(24)  # Loop over hours
    }

    m.update()
    return variables


def build_subproblem_objective(m, variables, DA_prices_3d, generation_existing_cost):
    

    """Builds the objective function for the subproblem."""
    objective = 20 * 365 * gb.quicksum(
        gb.quicksum(
            gb.quicksum(
                (DA_prices_3d[w][t, n] * ((variables['wind_production'][w, t] if n == 13 else 0)
                    + variables['existing_production'][n, w, t]
                ) - variables['existing_production'][n, w, t] * generation_existing_cost[n])
                for n in range(24)  # Assuming there are 24 nodes
            )
            for t in range(24)  # Assuming there are 24 hours
        )
        for w in range(len(DA_prices_3d))  # Looping over all scenarios w
    )

    m.setObjective(objective, gb.GRB.MINIMIZE)
    m.update()


def build_subproblem_constraints(m, variables, DA_prices_3d):
    """Builds constraints for the subproblem."""
    constraints = {}

    # Wind production upper limit
    constraints['wind_upper_limit'] = m.addConstrs(
        (
            variables['wind_production'][w, t]
            <= 250
            # <= Wind_PF_data[t] * variables['inv_cap_wind'].x[w, t]
            for w in range(len(DA_prices_3d))
            for t in range(24)
        ),
        name='set upper bound for wind production'
    )

    # Existing production upper limit
    constraints['existing_upper_limit'] = m.addConstrs(
        (
            variables['existing_production'][n, w, t]
            <= generation_capacity[n]
            for n in range(24)
            for w in range(len(DA_prices_3d))
            for t in range(24)
        ),
        name='set upper bound for existing production'
    )

    m.update()
    return constraints


def build_master_variables(m, benders_type, DA_prices_3d):
    """Builds master problem variables"""
    variables = {}

    # complicating variables
    variables['inv_cap_wind'] = m.addVar(lb=0, ub=250, name='Investment Capacity Wind')

    # Gamma = approximator of subproblems' objective value
    if benders_type == 'uni-cut':  # One new cut per iteration
        variables['gamma'] = m.addVar(lb=-1000, name='gamma')

    if benders_type == 'multi-cut':  # One new cut per subproblem and per iteration
        variables['gamma'] = {scenario: m.addVar(lb=-1000, name=f'gamma_{scenario}') for scenario in range(len(DA_prices_3d))}

    m.update()
    return variables


def build_master_objective(m, variables, benders_type, DA_prices_3d):
    """Builds the objective function for the master problem."""
    if benders_type == 'uni-cut':
        objective = Investment_data.iloc[2, 1] * variables['inv_cap_wind'] - variables['gamma']

    if benders_type == 'multi-cut':
        objective = Investment_data.iloc[2, 1] * variables['inv_cap_wind'] - gb.quicksum(
            probability_scenario[w] * variables['gamma'][w] for w in range(len(DA_prices_3d)))

    m.setObjective(objective, gb.GRB.MINIMIZE)
    m.update()


def build_master_constraints(m, variables):
    """Builds the master problem constraints."""
    K = 5e5

    constraints = {}

    # Add constraints related to complicating constraints
    constraints['DA_balance_constraint'] = m.addConstr(
        Investment_data.iloc[2, 1] * variables['inv_cap_wind'],
        gb.GRB.LESS_EQUAL,
        K, name='maximum investment')  # day-ahead balance equation

    # Initialize master problem cuts (empty)
    constraints['master_cuts'] = {}

    m.update()
    return constraints


def solve_benders_step(master, subproblems, DA_prices_3d, probability_scenario, epsilon, max_iters):
    """Executes one Benders step (solve both master and subproblems)."""
    m = master['model']
    m.setParam('Presolve', 2)  # Enables presolve with bound tightening
    m.setParam('Cuts', 2)  # Enable cutting planes for improved bounds

    master['iteration'] += 1  # Go to next iteration
    m.optimize()  # Optimize master problem
    if m.status == GRB.OPTIMAL:
        print(f"Optimal value of inv_cap_wind: {master['variables']['inv_cap_wind'].x}")
    elif m.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    else:
        print("Model is unbounded or another issue occurred.")

    # Save master problem data
    save_master_data(master)

    # Solve subproblems
    for w in range(len(DA_prices_3d)):
        subproblem = subproblems[w]
        subproblem['model'].optimize()
        save_subproblem_data(subproblem, w)

    # Update upper and lower bounds for convergence check
    update_upper_lower_bounds(master, subproblems, DA_prices_3d, probability_scenario)


def save_master_data(master):
    """Saves results from the master problem."""
    # Save master data
    pass


def save_subproblem_data(subproblem, w):
    """Saves results from subproblems."""
    # Save subproblem data
    pass


def update_upper_lower_bounds(master, subproblems, DA_prices_3d, probability_scenario):
    """Updates the upper and lower bounds for convergence check."""
    pass


#%% Main function to run the optimization

start = timeit.timeit()  # Define start time

# Build master problem
master = {}
master['model'] = gb.Model(name='master')
master['variables'] = build_master_variables(master['model'], benders_type='multi-cut', DA_prices_3d=DA_prices_3d)
master['constraints'] = build_master_constraints(master['model'], master['variables'])

# Build subproblems
subproblems = {}
for w in range(len(DA_prices_3d)):
    subproblem = {}
    subproblem['model'] = gb.Model(name=f'subproblem_{w}')
    subproblem['variables'] = build_subproblem_variables(subproblem['model'], DA_prices_3d)
    subproblem['objective'] = build_subproblem_objective(subproblem['model'], subproblem['variables'], DA_prices_3d, generation_existing_cost)
    subproblem['constraints'] = build_subproblem_constraints(subproblem['model'], subproblem['variables'], DA_prices_3d)
    subproblems[w] = subproblem

# Perform Benders iteration
solve_benders_step(master, subproblems, DA_prices_3d, probability_scenario, epsilon=1e-6, max_iters=100)

end = timeit.timeit()  # End time
print(f"Optimization completed in {end - start:.2f} seconds")
