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
BENDERS_TYPES = ['unit-cut','multi-cut'] # types of benders algorithm
#CONTROLLABLE_GENERATORS = ['G1','G3'] #range of controllable generators
#WIND_GENERATORS = ['G2'] #range of wind generators
# GENERATORS = ['G1','G2','G3'] #range of all generators
# LOADS = ['D1'] #range of Loads
#SCENARIOS = ['S1','S2','S3','S4'] # range of wind production scenarios


# Set values of input parameters
generation_existing_cost = investor_generation_data["Bid price [$/MWh]"] 
#adjustment_cost_up = {'G1':dispatch_cost['G1']+2,'G2':dispatch_cost['G2']+2,'G3':dispatch_cost['G3']+2} # costs for upward adjustments in real time (c^up_i) in DKK/MWh
#adjustment_cost_down = {'G1':dispatch_cost['G1']-1,'G2':dispatch_cost['G2']-1,'G3':dispatch_cost['G3']-1} # costs for downward adjustments in real time (c^dw_i) in DKK/MWh
generation_capacity =  investor_generation_data["Pmax [MW]"]  # Generators capacity (Q^G_i) in MW
#adjustment_capacity_up = {'G1':10,'G2':150,'G3':50} # upward adjustment capacity (Q^up_i) in MW
#adjustment_capacity_down = {'G1':10,'G2':150,'G3':50} # downward adjustment capacity (Q^dw_i) in MW
#wind_availability_scenario = {('G2','S1'):0.6,('G2','S2'):0.7,('G2','S3'):0.75,('G2','S4'):0.85} # scenarios of available wind production -
#scenario_probability = DA_prices_3d # probability of scenarios of available wind production -
#load_capacity = {'D1':200} # inflexible load consumption

#%%


class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class benders_subproblem: # Class representing the subproblems for each scenario

    def __init__(self,master,scenario,inv_cap_wind_init): # initialize class
        self.data = expando() # define data attributes
        self.variables = expando() # define variable attributes
        self.constraints = expando() # define constraints attributes
        self.master = master # define master problem to which subproblem is attached
        self._init_data(scenario,inv_cap_wind_init) # initialize data
        self._build_model() # build gurobi model
    
        
    def _init_data(self,scenario,inv_cap_wind_init): # initialize data
        self.data.scenario = scenario # add scenario #
        self.data.inv_cap_wind_init = inv_cap_wind_init # add initial value of complicating variables

    def _build_model(self): # build gurobi model
        
        self.model = gb.Model(name='subproblem') # build model
        self._build_variables() # add variables
        self._build_objective() # add objective
        self._build_constraints() # add constraints
        self.model.update() 


    def _build_variables(self): # build variables

        #index shortcut 
        m = self.model
        
        # complicating variables
        self.variables.inv_cap_wind = m.addVar(lb=0,ub=250,name='Investment Capacity Wind') # electricity production of generators (x^G_i)
    
        # subproblem variables
        self.variables.wind_production = {
            (w, t): m.addVar(lb=0, ub=250, name=f'wind production for scenario {w} in hour {t}')
            for w in range(len(DA_prices_3d))
            for t in range(24)  # 0 to 23 inclusive, so use range(24)
        }
        
        
        self.variables.existing_production = {
            (n, w, t): m.addVar(lb=0,ub=investor_generation_data.loc[n,"Pmax [MW]"],name=f'wind production for scenario {w} in hour {t}')
            for n in range(24)  # 0 to 23 inclusive, so use range(24)
            for w in range(len(DA_prices_3d))
            for t in range(24)  # 0 to 23 inclusive, so use range(24)
        }
        
        m.update() # update model
    

    def _build_objective(self): # define objective function
 
        #index shortcut 
        m = self.model

                                                            
        subproblem_objective = 20 * 365 * gb.quicksum(
            gb.quicksum(
                gb.quicksum(
                    (DA_prices_3d[str(w)][t, n] * (
                        (self.variables.wind_production[w, t] if n == 13 else 0)
                        + self.variables.existing_production[n, w, t]
                    ) - self.variables.existing_production[n, w, t] * generation_existing_cost[n])
                    for n in range(len(investor_generation_data))  # Loop over all nodes
                )
                for t in range(24)  # Loop over all hours
            )
            for w in range(len(DA_prices_3d))  # Loop over all scenarios
        )

        m.setObjective(subproblem_objective, gb.GRB.MINIMIZE) #minimize cost

        m.update() 
        

    def _build_constraints(self):

        #index shortcut 
        m = self.model

        self.constraints.wind_upper_limit_constraint = m.addConstrs(
            (
                self.variables.wind_production[w, t]
                <= Wind_PF_data.iloc[t,1] * self.variables.inv_cap_wind
                for w in range(len(DA_prices_3d))  # Loop over scenarios
                for t in range(24)  # Loop over hours
            ),
            name='set upper bound for wind production'
        )
        
        
        self.constraints.existing_upper_limit_constraint = m.addConstrs(
            (
                self.variables.existing_production[n, w, t]
                <= generation_capacity[n]
                for n in range(24)
                for w in range(len(DA_prices_3d))  # Loop over scenarios
                for t in range(24)  # Loop over hours
            ),
            name='set upper bound for existing production'
        )
        
        self.constraints.fix_wind_cap = m.addConstr(
            self.variables.inv_cap_wind,
            gb.GRB.EQUAL,
            self.data.inv_cap_wind_init,
            name='fix inv wind cap variable')
    
        m.update()


    def _update_complicating_variables(self): # function that updates the value of the complicating variables in the right-hand-side of self.constraints.fix_generator_dispatch

        # index shortcut
        m = self.model
        
        # Update the right-hand side of the constraint
        self.constraints.fix_wind_cap.rhs = self.master.variables.inv_cap_wind.x#-------------------------------------------------------------------------------------------------------------------------------
    
        m.update()


#%%     define class of master problem taking as inputs the benders_type: uni-cut or multi-cut, epsilon: convergence criteria parameter, and max_iters: maximum number of terations   

class benders_master: # class of master problem
    
    def __init__(self,benders_type,epsilon,max_iters): # initialize class
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build sontraint attributes
        self._init_data(benders_type,epsilon,max_iters) # initialize data
        self._build_model() # build gurobi model
    
        
    def _init_data(self,benders_type,epsilon,max_iters): # initialize data

        self.data.benders_type = benders_type # add type of benders problem (unit-cut or multi-cut)
        self.data.epsilon = epsilon # add value of convergence criteria
        self.data.max_iters = max_iters # add max number of iterations
        self.data.iteration = 1 # initialize value of iteration count
        self.data.upper_bounds = {} # initialize list of upper-bound values
        self.data.lower_bounds = {} # initialize list of lower-bound values
        self.data.inv_cap_wind_sensitivity = {} # initialize list of sensitivities values
        self.data.inv_cap_wind_values = {} # initialize list of complicating variables values
        self.data.gamma_values = {} # initialize list of gamma values
        self.data.subproblem_objectives = {} # initialize list of subproblems objective values
        self.data.master_objectives = {} # initialize list of master problem objective values

    def _build_model(self): # build gurobi model
        
        self.model = gb.Model(name='master') # build model
        self._build_variables() # add variables
        self._build_objective() # add objective
        self._build_constraints() # add constraints
        self.model.update()


    def _build_variables(self): # build variables

        #index shortcut 
        m = self.model
        
        # complicating variables
        self.variables.inv_cap_wind = m.addVar(lb=0,ub=250,name='Investment Capacity Wind') # electricity production of generators (x^G_i)
        # Set an initial starting value for inv_cap_wind
        self.variables.inv_cap_wind.start = 10  # Starting value for the first iteration


        # gamma = approximator of subproblems' objective value
        if self.data.benders_type == 'uni-cut': # one new cut per iteration
            self.variables.gamma = m.addVar(lb=-GRB.INFINITY,ub=10000, name='gamma')
            
        if self.data.benders_type == 'multi-cut': # one new cut per subproblem and per iteration
            self.variables.gamma = {scenario:m.addVar(lb=-GRB.INFINITY,ub=10000,name='gamma') for scenario in range(len(DA_prices_3d))}
            
            # Set an initial starting value for inv_cap_wind
            self.variables.gamma.start = 10  # Starting value for the first iteration

        
        print("Gamma", len(DA_prices_3d))
        m.update()
    

    def _build_objective(self): # build objective
 
        #index shortcut 
        m = self.model

        # Set the objective function for the master problem
        if self.data.benders_type == 'uni-cut':
            master_objective = Investment_data.iloc[2,1]*self.variables.inv_cap_wind - self.variables.gamma # expected electricity production cost (z)
            
        if self.data.benders_type == 'multi-cut':
            master_objective =Investment_data.iloc[2,1]*self.variables.inv_cap_wind - gb.quicksum(probability_scenario[w]*self.variables.gamma[w] for w in range(len(DA_prices_3d))) # expected electricity production cost (z)   
        m.setObjective(master_objective, gb.GRB.MINIMIZE) #minimize cost

        m.update() 

    def _build_constraints(self): # build constraints

        K=5e5
        #index shortcut 
        m = self.model
            
        # add constraints related to complicating constraints
        self.constraints.DA_balance_constraint = m.addConstr(
                Investment_data.iloc[2,1]*self.variables.inv_cap_wind,
                gb.GRB.LESS_EQUAL,
                K,name='maximum investment') # day-ahead balance equation

        # initialize master problem cuts (empty)
        self.constraints.master_cuts = {}
        
        m.update()


    def _build_subproblems(self): # function that builds subproblems
        
        self.subproblem = {w:benders_subproblem(self,scenario=w,inv_cap_wind_init=self.variables.inv_cap_wind.x) for w in range(len(DA_prices_3d))}


    # def _update_master_cut(self): # fucntion tat adds cuts to master problem
        
    #     # index shortcut
    #     m = self.model

    #     if self.data.benders_type == 'uni-cut':           
    #         self.constraints.master_cuts[self.data.iteration] = m.addConstr(
    #             self.variables.gamma,
    #             gb.GRB.GREATER_EQUAL,
    #             gb.quicksum(probability_scenario[w]*(self.data.subproblem_objectives[self.data.iteration-1][w] 
    #                                                  + self.data.inv_cap_wind_sensitivity[self.data.iteration-1][w]*(self.variables.inv_cap_wind-self.data.inv_cap_wind_values[self.data.iteration-1]))  
    #                                                  for w in range(len(DA_prices_3d))),
    #             name='new (uni)-cut at iteration {0}'.format(self.data.iteration))

    #     if self.data.benders_type == 'multi-cut': 
    #         for w in range(len(DA_prices_3d)):
    #             self.constraints.master_cuts[self.data.iteration,w] = m.addConstr(
    #                 self.variables.gamma[w],
    #                 gb.GRB.GREATER_EQUAL,
    #                 self.data.subproblem_objectives[self.data.iteration-1][w] 
    #                 + gb.quicksum(self.data.inv_cap_wind_sensitivity[self.data.iteration-1][w]*(self.variables.inv_cap_wind-self.data.inv_cap_wind_values[self.data.iteration-1]) ),
    #                 name='new (multi)-cut for subproblem {0} at iteration {1}'.format(w,self.data.iteration))

    #     m.update()
    
    def _update_master_cut(self): 
        """Function that adds cuts to master problem for iterations >= 2."""
        
        # Only execute for iterations >= 2
        if self.data.iteration < 2:
            return  # Exit the function for iteration < 2
    
        # Shortcut to the model
        m = self.model
    
        if self.data.benders_type == 'uni-cut':           
            self.constraints.master_cuts[self.data.iteration] = m.addConstr(
                self.variables.gamma,
                gb.GRB.GREATER_EQUAL,
                gb.quicksum(
                    probability_scenario[w] * (
                        self.data.subproblem_objectives[self.data.iteration - 1][w] 
                        + self.data.inv_cap_wind_sensitivity[self.data.iteration - 1][w] * (
                            self.variables.inv_cap_wind - self.data.inv_cap_wind_values[self.data.iteration - 1]
                        )
                    )  
                    for w in range(len(DA_prices_3d))
                ),
                name='new (uni)-cut at iteration {0}'.format(self.data.iteration)
            )
    
        if self.data.benders_type == 'multi-cut': 
            for w in range(len(DA_prices_3d)):
                self.constraints.master_cuts[self.data.iteration, w] = m.addConstr(
                    self.variables.gamma[w],
                    gb.GRB.GREATER_EQUAL,
                    self.data.subproblem_objectives[self.data.iteration - 1][w] 
                    + self.data.inv_cap_wind_sensitivity[self.data.iteration - 1][w] * (
                        self.variables.inv_cap_wind - self.data.inv_cap_wind_values[self.data.iteration - 1]
                    ),
                    name='new (multi)-cut for subproblem {0} at iteration {1}'.format(w, self.data.iteration)
                )
    
        m.update()

    
    
    # def _save_master_data(self): # function that saves results of master problem optimization at each iteration (complicating variables, objective value, lower bound value)
        
    #     # index shortcut
    #     m = self.model
        
       
        
    #     # save complicating variables value
    #     self.data.inv_cap_wind_values[self.data.iteration] = self.variables.inv_cap_wind.x
        
    #     # save gamma value
    #     if self.data.benders_type == 'uni-cut':
    #         self.data.gamma_values[self.data.iteration] = self.variables.gamma.x
    #     if self.data.benders_type == 'multi-cut':
    #         self.data.gamma_values[self.data.iteration] = {w:self.variables.gamma[w].x for w in range(len(DA_prices_3d))}           
        
    #     # save lower bound value
    #     self.data.lower_bounds[self.data.iteration] = m.ObjVal

    #     # save master problem objective value
    #     if self.data.benders_type == 'uni-cut':
    #         self.data.master_objectives[self.data.iteration] = m.ObjVal - self.variables.gamma.x
    #     if self.data.benders_type == 'multi-cut':
    #         self.data.master_objectives[self.data.iteration] = m.ObjVal -sum(probability_scenario[w]*self.variables.gamma[w].x for w in range(len(DA_prices_3d)))           

        
    #     m.update()
    
    def _save_master_data(self): 
        """Function that saves results of master problem optimization at each iteration."""
    
        # Shortcut to the model
        m = self.model
        
        # Check if the model is solved optimally
        if m.status == GRB.OPTIMAL:
            # Save complicating variable values
            self.data.inv_cap_wind_values[self.data.iteration] = self.variables.inv_cap_wind.x
    
            # Save gamma values
            if self.data.benders_type == 'uni-cut':
                self.data.gamma_values[self.data.iteration] = self.variables.gamma.x
            if self.data.benders_type == 'multi-cut':
                self.data.gamma_values[self.data.iteration] = {w: self.variables.gamma[w].x for w in range(len(DA_prices_3d))}
            
            # Save lower bound value
            self.data.lower_bounds[self.data.iteration] = m.ObjVal
    
            # Save master problem objective value
            if self.data.benders_type == 'uni-cut':
                self.data.master_objectives[self.data.iteration] = m.ObjVal - self.variables.gamma.x
            if self.data.benders_type == 'multi-cut':
                self.data.master_objectives[self.data.iteration] = m.ObjVal - sum(
                    probability_scenario[w] * self.variables.gamma[w].x for w in range(len(DA_prices_3d))
                )
        else:
            # Handle non-optimal cases: infeasibility, unboundedness, etc.
            print(f"Master problem optimization failed at iteration {self.data.iteration}. Status: {m.status}")
            raise RuntimeError("Master problem did not solve to optimality.")
        
        m.update()


    def _save_subproblems_data(self): # function that saves results of subproblems optimization at each iteration (sensitivities, objective value, upper bound value)
        
        # index shortcut
        m = self.model

        # save sensitivities (for each complicating variables in each subproblem)
        self.data.inv_cap_wind_sensitivity[self.data.iteration] = {(w):self.subproblem[w].constraints.fix_wind_cap.Pi  for w in range(len(DA_prices_3d))}
        
        # save subproblems objective values
        self.data.subproblem_objectives[self.data.iteration] = {w:self.subproblem[w].model.ObjVal for w in range(len(DA_prices_3d))}             
        
        # save upper bound value
        self.data.upper_bounds[self.data.iteration] = self.data.master_objectives[self.data.iteration] + sum(probability_scenario[w]*self.subproblem[w].model.ObjVal for w in range(len(DA_prices_3d))) ##################################
                      
        m.update()

    def _do_benders_step(self): # function that does one benders step
        
        # index shortcut
        m = self.model
        
        m.setParam('Presolve', 2)  # Enables presolve with bound tightening
        m.setParam('Cuts', 2)  # Enable cutting planes for improved bounds
    
        self.data.iteration += 1 # go to next iteration        
        self._update_master_cut() # add cut
        m.optimize() # optimize master problem
        if m.status == GRB.OPTIMAL:
            print(f"Optimal value of inv_cap_wind: {self.variables.inv_cap_wind.x}")
        elif m.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        else:
            print("Model is unbounded or another issue occurred.")
        
        self._save_master_data() # save master problem optimization results
        for w in range(len(DA_prices_3d)): 
            self.subproblem[w]._update_complicating_variables() # update value of complicating constraints in subproblems
            self.subproblem[w].model.optimize() # solve subproblems
        self._save_subproblems_data() # save subproblems optimization results


               
    def _benders_iterate(self): # function that solves iteratively the benders algorithm
        
        # index shortcut            
        m = self.model
        #m.setParam('Presolve', 2)  # Enables presolve with bound tightening
        #m.setParam('Cuts', 2)  # Enable cutting planes for improved bounds
        
        # initial iteration: 
        m.optimize() # solve master problem (1st iteration)
        if m.status == GRB.OPTIMAL:
            print(f"Optimal value of inv_cap_wind: {self.variables.inv_cap_wind.x}")
        elif m.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        else:
            print("Model is unbounded or another issue occurred.")
        
        self._save_master_data() # save results of master problem and lower bound
        self._build_subproblems() # build subproblems (1st iteration)
        for w in range(len(DA_prices_3d)): 
            self.subproblem[w].model.optimize() # solve subproblems
        self._save_subproblems_data() # save results of subproblems and upper bound
    
        # do benders steps until convergence
        while (
            (abs(self.data.upper_bounds[self.data.iteration] - self.data.lower_bounds[self.data.iteration])>self.data.epsilon and
                self.data.iteration < self.data.max_iters)):
            self._do_benders_step()


#%% solve and print results for uni-cut

# start = timeit.timeit() # define start time

# DA_model = benders_master(benders_type='uni-cut',epsilon=0.1,max_iters=100)
# DA_model._benders_iterate()

# end = timeit.timeit() # define end time

# print('uni-cut solving time',end-start) # print solving time

# print('uni-cut optimal cost',DA_model.data.upper_bounds[DA_model.data.iteration]) # print optimal cost (last upper-bound)

# f, ax=plt.subplots(figsize=(10,10)) # print upper and lower bounds evolution at each iteration
# ax.plot(range(1,DA_model.data.iteration),[DA_model.data.upper_bounds[it] for it in range(1,DA_model.data.iteration)],label='upper-bound',linewidth=2,marker='o',color='red') # upper bounds at each iteration
# ax.plot(range(1,DA_model.data.iteration),[DA_model.data.lower_bounds[it] for it in range(1,DA_model.data.iteration)],label='lower-bound',linewidth=2,marker='o',color='blue') # lower bounds at each iteration
# ax.set_ylabel('Bounds (DKK)',fontsize=size_pp+5) 
# ax.set_xlabel('Iterations',fontsize=size_pp+5) 
# ax.legend(bbox_to_anchor=(0.75,1),bbox_transform=plt.gcf().transFigure,ncol=2,fontsize=size_pp+5)


#%% solve and print results for multi-cut

start = timeit.timeit() # define start time

DA_model = benders_master(benders_type='multi-cut',epsilon=0.1,max_iters=100)
DA_model._benders_iterate()

end = timeit.timeit() # define end time

print('multi-cut solving time',end-start) # print solving time

print('multi-cut optimal cost',DA_model.data.upper_bounds[DA_model.data.iteration]) # print optimal cost (last upper-bound)
print('end iteration',DA_model.data.iteration)

f, ax=plt.subplots(figsize=(10,10)) # print upper and lower bounds evolution at each iteration
ax.plot(range(1,DA_model.data.iteration),[DA_model.data.upper_bounds[it] for it in range(1,DA_model.data.iteration)],label='upper-bound',linewidth=2,marker='o',color='red') # upper bounds at each iteration
ax.plot(range(1,DA_model.data.iteration),[DA_model.data.lower_bounds[it] for it in range(1,DA_model.data.iteration)],label='lower-bound',linewidth=2,marker='o',color='blue') # lower bounds at each iteration
ax.set_ylabel('Bounds (DKK)',fontsize=size_pp+5) 
ax.set_xlabel('Iterations',fontsize=size_pp+5) 
ax.legend(bbox_to_anchor=(0.75,1),bbox_transform=plt.gcf().transFigure,ncol=2,fontsize=size_pp+5)

