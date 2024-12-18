
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


inv_cap_wind_init=0

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
    variables['inv_cap_wind'] = m.addVar(lb=0, ub=150, name='Investment Capacity Wind')

    # Subproblem variables
    variables['wind_production'] = {
        (w, t): m.addVar(lb=0, ub=150, name=f'wind production for scenario {w} in hour {t}')
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
    objective = 20 * 365 * gb.quicksum(
        gb.quicksum(
            gb.quicksum(
                (DA_prices_3d[str(w)][t, n] * ((variables['wind_production'][w, t] if n == 13 else 0)
                    + variables['existing_production'][n, w, t])
                 - variables['existing_production'][n, w, t] * generation_existing_cost[n])
                for n in range(24)
            )
            for t in range(24)
        )
        for w in range(len(DA_prices_3d))
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
            <=Wind_PF_data.iloc[t,1] * variables['inv_cap_wind']
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
    
    # Add fix_wind_cap constraint
    constraints['fix_wind_cap'] = m.addConstr(
        variables['inv_cap_wind'],
        gb.GRB.EQUAL,
        inv_cap_wind_init,  # Placeholder for initialization; this will be updated later
        name='fix inv wind cap variable'
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


def _update_master_cut(master, subproblems, probability_scenario, DA_prices_3d):
    """
    Adds cuts to the master problem for the current iteration.
    Handles both uni-cut and multi-cut Benders decomposition.
    """
    # Shortcut for the model and iteration
    m = master['model']
    iteration = master['iteration']
    
    # Ensure the data storage for constraints exists
    if 'constraints' not in master:
        master['constraints'] = {'master_cuts': {}}

    print(f"Adding cuts at iteration {iteration}...")

    if master['benders_type'] == 'uni-cut':
        # Add a single cut for all subproblems
        master['constraints']['master_cuts'][iteration] = m.addConstr(
            master['variables']['gamma'],
            gb.GRB.GREATER_EQUAL,
            gb.quicksum(
                probability_scenario[w] * (
                    master['data']['subproblem_objectives'][iteration - 1][w]
                    + master['data']['sensitivities'][iteration - 1][(w, 'inv_cap_wind')] * (
                        master['variables']['inv_cap_wind'] - master['data']['inv_cap_wind_values'][iteration - 1]
                    )
                )
                for w in range(len(DA_prices_3d))
            ),
            name=f'new_uni_cut_iteration_{iteration}'
        )

    elif master['benders_type'] == 'multi-cut':
        # Add separate cuts for each subproblem
        for w in range(len(DA_prices_3d)):
            master['constraints']['master_cuts'][(iteration, w)] = m.addConstr(
                master['variables']['gamma'][w],
                gb.GRB.GREATER_EQUAL,
                master['data']['subproblem_objectives'][iteration - 1][w]
                + master['data']['sensitivities'][iteration - 1][(w, 'inv_cap_wind')] * (
                    master['variables']['inv_cap_wind'] - master['data']['inv_cap_wind_values'][iteration - 1]
                ),
                name=f'new_multi_cut_subproblem_{w}_iteration_{iteration}'
            )

    m.update()



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
    _save_master_data(master, probability_scenario)

    # Solve subproblems
    for w in range(len(DA_prices_3d)):
        subproblem = subproblems[w]
        subproblem['model'].optimize()
        _save_subproblems_data(master, subproblems, probability_scenario)

    # Update upper and lower bounds for convergence check
    update_upper_lower_bounds(master, subproblems, DA_prices_3d, probability_scenario)


def _save_master_data(master, probability_scenario):
    """
    Saves results of master problem optimization at each iteration.
    This includes complicating variables, objective value, and lower bound value.
    """
    # Shortcut for the model and iteration
    m = master['model']
    iteration = master['iteration']
    
    # Ensure the data storage exists
    if 'data' not in master:
        master['data'] = {
            'inv_cap_wind_values': {},  # Store investment capacity values
            'gamma_values': {},         # Store gamma values
            'lower_bounds': {},         # Store lower bound values
            'master_objectives': {}     # Store master problem objective values
        }
    
    # Save complicating variable (investment capacity wind)
    master['data']['inv_cap_wind_values'][iteration] = master['variables']['inv_cap_wind'].x

    # Save gamma value
    if 'gamma' in master['variables']:  # For uni-cut
        master['data']['gamma_values'][iteration] = master['variables']['gamma'].x
    else:  # For multi-cut
        master['data']['gamma_values'][iteration] = {
            s: master['variables']['gamma'][s].x for s in probability_scenario.keys()
        }
    
    # Save lower bound value (master problem objective value)
    master['data']['lower_bounds'][iteration] = m.ObjVal

    # Save adjusted master problem objective value
    if 'gamma' in master['variables']:  # For uni-cut
        master['data']['master_objectives'][iteration] = m.ObjVal - master['variables']['gamma'].x
    else:  # For multi-cut
        master['data']['master_objectives'][iteration] = m.ObjVal - sum(
            probability_scenario[s] * master['variables']['gamma'][s].x for s in probability_scenario.keys()
        )

    m.update()




def _save_subproblems_data(master, subproblems, probability_scenario):
    """
    Saves results of subproblems optimization at each iteration.
    This includes sensitivities, objective values, and the upper bound value.
    """
    # Shortcut for iteration
    iteration = master['iteration']
    
    # Ensure the data storage exists
    if 'data' not in master:
        master['data'] = {
            'sensitivities': {},  # Store dual sensitivities
            'subproblem_objectives': {},  # Store subproblem objectives
            'upper_bounds': {}  # Store upper bounds
        }
    
    # Save sensitivities (dual values for complicating variables in subproblems)
    master['data']['sensitivities'][iteration] = {
        (w, 'inv_cap_wind'): subproblems[w]['constraints']['fix_wind_cap'].Pi
        for w in range(len(subproblems))
    }

    # Save subproblems' objective values
    master['data']['subproblem_objectives'][iteration] = {
        w: subproblems[w]['model'].ObjVal for w in range(len(subproblems))
    }

    # Save upper bound value
    master['data']['upper_bounds'][iteration] = master['data']['master_objectives'][iteration] + sum(
        probability_scenario[w] * subproblems[w]['model'].ObjVal for w in range(len(subproblems))
    )

    # No need to call model.update() as we're saving data and not modifying constraints or variables.



def update_upper_lower_bounds(master, subproblems, DA_prices_3d, probability_scenario):
    """Updates the upper and lower bounds for convergence check."""
    # Retrieve the master problem's objective value as the lower bound
    if master['model'].status == gb.GRB.OPTIMAL:
        lower_bound = master['model'].objVal
    else:
        lower_bound = -float('inf')  # If the master problem is infeasible or unbounded
    
    # Compute the upper bound by combining the subproblem objectives
    upper_bound = 0
    for w, subproblem in subproblems.items():
        model = subproblem['model']
        if model.status == gb.GRB.OPTIMAL:
            subproblem_obj_val = model.objVal
            upper_bound += probability_scenario[w] * subproblem_obj_val
        else:
            upper_bound = float('inf')  # If any subproblem is infeasible or unbounded
    
    # Store bounds for debugging/logging
    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    return lower_bound, upper_bound


#%% Main function to run the optimization

start = timeit.timeit()  # Define start time

# Build master problem
master = {}
master['iteration'] = 0
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


