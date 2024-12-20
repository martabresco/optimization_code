

import gurobipy as gb
import matplotlib.pyplot as plt
import seaborn as sb
import timeit


from data_benders import (
    investor_generation_data_d,
    pv_PF,
    investment_data,
    DA_prices
)

# Set plot parameters
sb.set_style('ticks')
size_pp = 15
font = {'family': 'times new roman',
        'color':  'black',
        'weight': 'normal',
        'size': size_pp,
        }


SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] # range of PV production scenarios
probability_scenario = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]


# Define ranges and hyperparameters  
BENDERS_TYPES = ['unit-cut','multi-cut'] # types of benders algorithm

# Set values of input parameters
invested_node=18
max_Investment_cap=250
Budget= 2.00e8
discount_rate=0.05
lifetime_years=20

generation_existing_cost = investor_generation_data_d["Bid price [$/MWh]"] 
generation_capacity =  investor_generation_data_d["Pmax [MW]"]
PF_PV= pv_PF.iloc[:, 1] 



#%%


class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class benders_subproblem: # Class representing the subproblems for each scenario

    def __init__(self,master,scenario,inv_cap_PV_init): # initialize class
        self.data = expando() # define data attributes
        self.variables = expando() # define variable attributes
        self.constraints = expando() # define constraints attributes
        self.master = master # define master problem to which subproblem is attached
        self._init_data(scenario,inv_cap_PV_init) # initialize data
        self._build_model() # build gurobi model
    
        
    def _init_data(self,scenario,inv_cap_PV_init): # initialize data
        self.data.scenario = scenario # add scenario #
        self.data.inv_cap_PV_init = inv_cap_PV_init # add initial value of complicating variables

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
        self.variables.inv_cap_PV = m.addVar(lb=0,ub=max_Investment_cap,name='Investment Capacity PV') # electricity production of generators (x^G_i)
    
        # subproblem variables
        self.variables.PV_production = {
            (n,t): m.addVar(lb=0, name=f'PV production at node{n} in hour {t}')
            for n in range(24)  # 0 to 23 inclusive, so use range(24)
            for t in range(24)  # 0 to 23 inclusive, so use range(24)
        }
        
        
        self.variables.existing_production = {
            (n, t): m.addVar(lb=0,ub=generation_capacity[n],name=f'existing production at node{n} in hour {t}')
            for n in range(24)  # 0 to 23 inclusive, so use range(24)
            for t in range(24)  # 0 to 23 inclusive, so use range(24)
        }
        
        m.update() # update model
    
    
    
        

    def _build_objective(self): # define objective function
 
        #index shortcut 
        m = self.model



        subproblem_objective = (
            gb.quicksum(  # Sum over the lifetime of the project
                (1 / ((1 + discount_rate) ** t)) *  # Discount factor for year t
                (
                   - 365 * (
                        gb.quicksum(
                            gb.quicksum(
                                DA_prices[self.data.scenario, h, n] *
                                (self.variables.PV_production[n, h] + self.variables.existing_production[n, h])
                                for n in range(24)
                            )
                            for h in range(24)
                        )
                    ) +
                    365 * (
                        gb.quicksum(
                            gb.quicksum(
                                self.variables.existing_production[n, h] * generation_existing_cost[n]
                                for n in range(24)
                            )
                            for h in range(24)
                        )
                    )
                )
                for t in range(1, lifetime_years + 1)  # Iterate over project lifetime
            )
        )



        m.setObjective(subproblem_objective, gb.GRB.MINIMIZE) #minimize cost

        m.update() 
        

    def _build_constraints(self):

        #index shortcut 
        m = self.model
        
        self.constraints.PV_set_0_if_n = m.addConstrs(
            (
                self.variables.PV_production[n,t]==0
                for n in range(24) if n!= invested_node
                for t in range(24)
            ),
            name="PV production set to zero, if not investent node"
        )
        
        
        self.constraints.PV_upper_limit_constraint = m.addConstrs(
            (
                self.variables.PV_production[n,t]<= self.variables.inv_cap_PV
                #self.variables.PV_production[n,t]<= self.variables.inv_cap_PV*PF_PV[t]
                for n in range(24)
                for t in range(24)
            ),
            name="PV_upper_limit_constraint"
        )
        
        
        self.constraints.existing_upper_limit_constraint = m.addConstrs(
            (
                self.variables.existing_production[n, t]
                <= generation_capacity[n]
                for n in range(24)
                for t in range(24)  # Loop over hours
            ),
            name='set upper bound for existing production'
        )
        
        self.constraints.fix_PV_cap = m.addConstr(
            self.variables.inv_cap_PV,
            gb.GRB.EQUAL,
            self.data.inv_cap_PV_init,
            name='fix inv PV cap variable')
        
        m.update()


    def _update_complicating_variables(self): # function that updates the value of the complicating variables in the right-hand-side of self.constraints.fix_generator_dispatch

        # index shortcut
        m = self.model
        
        # Update the right-hand side of the constraint
        self.constraints.fix_PV_cap.rhs = self.master.variables.inv_cap_PV.x
    
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
        self.data.inv_cap_PV_sensitivity = {} # initialize list of sensitivities values
        self.data.inv_cap_PV_values = {} # initialize list of complicating variables values
        self.data.gamma_values = {} # initialize list of gamma values
        self.data.subproblem_objectives = {} # initialize list of subproblems objective values
        self.data.master_objectives = {} # initialize list of master problem objective values

    def _build_model(self): # build gurobi model
        
        self.model = gb.Model(name='naster') # build model
        self._build_variables() # add variables
        self._build_objective() # add objective
        self._build_constraints() # add constraints
        self.model.update()


    def _build_variables(self): # build variables

       
        
        #index shortcut 
        m = self.model
        
        # complicating variables
        self.variables.inv_cap_PV = m.addVar(lb=0,ub=max_Investment_cap,name='Investment Capacity PV') # electricity production of generators (x^G_i)


        # gamma = approximator of subproblems' objective value
        if self.data.benders_type == 'uni-cut': # one new cut per iteration
            self.variables.gamma = m.addVar(lb=-10e10, name='gamma')
            
        if self.data.benders_type == 'multi-cut': # one new cut per subproblem and per iteration
            self.variables.gamma = {scenario:m.addVar(lb=-10e10,name='gamma') for scenario in SCENARIOS}


        m.update()
    

    def _build_objective(self): # build objective
        
        #index shortcut 
        m = self.model

        #Set the objective function for the master problem
        if self.data.benders_type == 'uni-cut':
            master_objective = investment_data.iloc[1,1]*self.variables.inv_cap_PV + self.variables.gamma # expected electricity production cost (z)
            
        if self.data.benders_type == 'multi-cut':
            master_objective = investment_data.iloc[1,1]*self.variables.inv_cap_PV + gb.quicksum(probability_scenario[s]*self.variables.gamma[s] for s in SCENARIOS) # expected electricity production cost (z)   
        m.setObjective(master_objective, gb.GRB.MINIMIZE) #minimize cost

            
        m.setObjective(master_objective, gb.GRB.MINIMIZE) #minimize cost

        m.update() 

    def _build_constraints(self): # build constraints
        
        #index shortcut 
        m = self.model
            
        # add constraints related to complicating constraints
        self.constraints.max_budget_constraint = m.addConstr(
                investment_data.iloc[1,1]*self.variables.inv_cap_PV,
                gb.GRB.LESS_EQUAL,
                Budget,name='maximum investment') # day-ahead balance equation

        # initialize master problem cuts (empty)
        self.constraints.master_cuts = {}
        
        m.update()


    def _build_subproblems(self): # function that builds subproblems
        
        self.subproblem = {s:benders_subproblem(self,scenario=s,inv_cap_PV_init=self.variables.inv_cap_PV.x) for s in SCENARIOS}


    def _update_master_cut(self): # fucntion tat adds cuts to master problem
        
        #Only execute for iterations >= 2
        #if self.data.iteration < 2:
                #return  # Exit the function for iteration < 2


        # index shortcut
        m = self.model

        if self.data.benders_type == 'uni-cut':           
            self.constraints.master_cuts[self.data.iteration] = m.addConstr(
                self.variables.gamma,
                gb.GRB.GREATER_EQUAL,
                gb.quicksum(probability_scenario[s]*(self.data.subproblem_objectives[self.data.iteration-1][s] + self.data.inv_cap_PV_sensitivity[self.data.iteration-1][s]*(self.variables.inv_cap_PV-self.data.inv_cap_PV_values[self.data.iteration-1])) for s in SCENARIOS),
                name='new (uni)-cut at iteration {0}'.format(self.data.iteration))

        if self.data.benders_type == 'multi-cut': 
            for s in SCENARIOS:
                self.constraints.master_cuts[self.data.iteration,s] = m.addConstr(
                    self.variables.gamma[s],
                    gb.GRB.GREATER_EQUAL,
                    self.data.subproblem_objectives[self.data.iteration-1][s] + self.data.inv_cap_PV_sensitivity[self.data.iteration-1][s]*(self.variables.inv_cap_PV-self.data.inv_cap_PV_values[self.data.iteration-1]),
                    name='new (multi)-cut for subproblem {0} at iteration {1}'.format(s,self.data.iteration))

        m.update()
        
    
    
    def _save_master_data(self): # function that saves results of master problem optimization at each iteration (complicating variables, objective value, lower bound value)
        
        # index shortcut
        m = self.model

        # Save complicating variable values
        self.data.inv_cap_PV_values[self.data.iteration] = self.variables.inv_cap_PV.x
        
        # save gamma value
        if self.data.benders_type == 'uni-cut':
            self.data.gamma_values[self.data.iteration] = self.variables.gamma.x

        # if self.data.benders_type == 'multi-cut':
        #     self.data.gamma_values[self.data.iteration] = {s:self.variables.gamma[s].x for s in SCENARIOS}           
        
        # save lower bound value
        self.data.lower_bounds[self.data.iteration] = m.ObjVal #self.variables.gamma.x 

        # save master problem objective value
        if self.data.benders_type == 'uni-cut':
            self.data.master_objectives[self.data.iteration] = m.ObjVal - self.variables.gamma.x
        if self.data.benders_type == 'multi-cut':
            self.data.master_objectives[self.data.iteration] = m.ObjVal -sum(probability_scenario[s]*self.variables.gamma[s].x for s in SCENARIOS)           

        
        m.update()
        
        

    def _save_subproblems_data(self): # function that saves results of subproblems optimization at each iteration (sensitivities, objective value, upper bound value)
        
        # index shortcut
        m = self.model

        # save sensitivities (for each complicating variables in each subproblem)
        self.data.inv_cap_PV_sensitivity[self.data.iteration] = {(s):self.subproblem[s].constraints.fix_PV_cap.Pi for s in SCENARIOS}
        
        # save subproblems objective values
        self.data.subproblem_objectives[self.data.iteration] = {s:self.subproblem[s].model.ObjVal for s in SCENARIOS}             
        
        # save upper bound value
        self.data.upper_bounds[self.data.iteration] = self.data.master_objectives[self.data.iteration] + sum(probability_scenario[s]*self.subproblem[s].model.ObjVal for s in SCENARIOS)#sum(probability_scenario[s]*self.subproblem[s].model.ObjVal for s in SCENARIOS)
    
        
        print("only master value", self.data.master_objectives[self.data.iteration])
        print("upper bound", self.data.upper_bounds[self.data.iteration])
        print("lower bound", self.data.lower_bounds[self.data.iteration])
        
        #print("subproblem existing production", self.subproblem.variables.existing_production)
        m.update()
        

    def _do_benders_step(self): # function that does one benders step
        
        # index shortcut
        m = self.model

        self.data.iteration += 1 # go to next iteration        
        self._update_master_cut() # add cut
        m.optimize() # optimize master problem
        self._save_master_data() # save master problem optimization results
        for s in SCENARIOS: 
            self.subproblem[s]._update_complicating_variables() # update value of complicating constraints in subproblems
            self.subproblem[s].model.optimize() # solve subproblems
        self._save_subproblems_data() # save subproblems optimization results
        
               
    def _benders_iterate(self): # function that solves iteratively the benders algorithm

        # index shortcut            
        m = self.model
        print("Iteration", self.data.iteration)
        # initial iteration: 
        m.optimize() #   solve master problem (1st iteration)
        
        print("only master value", m.ObjVal)
        
        self._save_master_data() # save results of master problem and lower bound
        self._build_subproblems() # build subproblems (1st iteration)
        for s in SCENARIOS: 
            self.subproblem[s].model.optimize() # solve subproblems
        self._save_subproblems_data() # save results of subproblems and upper bound


        print("x", self.data.upper_bounds[self.data.iteration])
        print("x", self.data.lower_bounds[self.data.iteration])
        
        # do benders steps until convergence
        while (
            (abs(self.data.upper_bounds[self.data.iteration] - self.data.lower_bounds[self.data.iteration])>self.data.epsilon and
                self.data.iteration < self.data.max_iters)):
            self._do_benders_step()





################ solve and print results for uni-cut ##################################

start = timeit.timeit() # define start time

DA_model = benders_master(benders_type='uni-cut',epsilon=0.1,max_iters=100)
DA_model._benders_iterate()

end = timeit.timeit() # define end time

print('uni-cut solving time',end-start) # print solving time


print('Iterations', DA_model.data.iteration)

print('upper bounds', DA_model.data.upper_bounds)
print('lower bounds', DA_model.data.lower_bounds)

print('uni-cut optimal profit',abs(DA_model.data.upper_bounds[DA_model.data.iteration])*10**(-6), 'mill.$') # print optimal cost (last upper-bound)

print('uni-cut investment x',DA_model.variables.inv_cap_PV.x, 'MW') # print optimal cost (last upper-bound)



f, ax=plt.subplots(figsize=(10,10)) # print upper and lower bounds evolution at each iteration
ax.plot(range(1,DA_model.data.iteration+1),[DA_model.data.upper_bounds[it] for it in range(1,DA_model.data.iteration+1)],label='upper-bound',linewidth=2,marker='o',color='red') # upper bounds at each iteration
ax.plot(range(1,DA_model.data.iteration+1),[DA_model.data.lower_bounds[it] for it in range(1,DA_model.data.iteration+1)],label='lower-bound',linewidth=2,marker='o',color='blue') # lower bounds at each iteration
ax.set_ylabel('Bounds ($)',fontsize=size_pp+5) 
ax.set_xlabel('Iterations',fontsize=size_pp+5) 
ax.legend(bbox_to_anchor=(0.75,1),bbox_transform=plt.gcf().transFigure,ncol=2,fontsize=size_pp+5)


################# solve and print results for multi-cut  #################################

# start = timeit.timeit() # define start time

# DA_model = benders_master(benders_type='multi-cut',epsilon=0.1,max_iters=100)
# DA_model._benders_iterate()

# end = timeit.timeit() # define end time

# print('multi-cut solving time',end-start) # print solving time

# print('multi-cut optimal profit',abs(DA_model.data.upper_bounds[DA_model.data.iteration])*10**(-6)'mill.$') # print optimal cost (last upper-bound)

# print('multi-cut investment x',DA_model.variables.inv_cap_PV.x, 'MW') # print optimal cost (last upper-bound)


# f, ax=plt.subplots(figsize=(10,10)) # print upper and lower bounds evolution at each iteration
# ax.plot(range(1,DA_model.data.iteration+1),[DA_model.data.upper_bounds[it] for it in range(1,DA_model.data.iteration+1)],label='upper-bound',linewidth=2,marker='o',color='red') # upper bounds at each iteration
# ax.plot(range(1,DA_model.data.iteration+1),[DA_model.data.lower_bounds[it] for it in range(1,DA_model.data.iteration+1)],label='lower-bound',linewidth=2,marker='o',color='blue') # lower bounds at each iteration
# ax.set_ylabel('Bounds (DKK)',fontsize=size_pp+5) 
# ax.set_xlabel('Iterations',fontsize=size_pp+5) 
# ax.legend(bbox_to_anchor=(0.75,1),bbox_transform=plt.gcf().transFigure,ncol=2,fontsize=size_pp+5)


###################   Results when varying budget   ########################################

# import pandas as pd


# # Define the range of budget values
# K_values = [ 1.50e6, 1.00e7, 1.50e7, 1.00e8, 2.00e8, 3.00e8, 4.00e8, 5.00e8, 6.00e8, 7.00e8, 8e8]

# # Results storage
# results = []

# for budget in K_values:
#     # Update the budget
#     Budget = budget
#     print(f"Running optimization for Budget: {Budget} DKK")

#     try:
#         # Solve the problem for uni-cut
#         start = timeit.default_timer()
#         DA_model_uni = benders_master(benders_type='uni-cut', epsilon=0.1, max_iters=100)
#         DA_model_uni.data.budget = Budget  # Dynamically set the budget
#         DA_model_uni._benders_iterate()
#         end = timeit.default_timer()

#         uni_cut_cost = DA_model_uni.data.upper_bounds.get(DA_model_uni.data.iteration, float('inf'))
#         uni_cut_investment = DA_model_uni.variables.inv_cap_PV.x if DA_model_uni.variables.inv_cap_PV else None
#         uni_cut_iterations = DA_model_uni.data.iteration

#         # Store uni-cut results
#         results.append({
#             "Budget": Budget,
#             "Benders Type": "uni-cut",
#             "Optimal Cost": uni_cut_cost,
#             "Investment Capacity (MW)": uni_cut_investment,
#             "Iterations": uni_cut_iterations,
#             "Solving Time (s)": end - start
#         })

#         # Solve the problem for multi-cut
#         start = timeit.default_timer()
#         DA_model_multi = benders_master(benders_type='multi-cut', epsilon=0.1, max_iters=100)
#         DA_model_multi.data.budget = Budget  # Dynamically set the budget
#         DA_model_multi._benders_iterate()
#         end = timeit.default_timer()

#         multi_cut_cost = DA_model_multi.data.upper_bounds.get(DA_model_multi.data.iteration, float('inf'))
#         multi_cut_investment = DA_model_multi.variables.inv_cap_PV.x if DA_model_multi.variables.inv_cap_PV else None
#         multi_cut_iterations = DA_model_multi.data.iteration

#         # Store multi-cut results
#         results.append({
#             "Budget": Budget,
#             "Benders Type": "multi-cut",
#             "Optimal Cost": multi_cut_cost,
#             "Investment Capacity (MW)": multi_cut_investment,
#             "Iterations": multi_cut_iterations,
#             "Solving Time (s)": end - start
#         })

#     except Exception as e:
#         print(f"Optimization failed for Budget: {Budget} with error: {e}")
#         results.append({
#             "Budget": Budget,
#             "Benders Type": "Error",
#             "Optimal Cost": None,
#             "Investment Capacity (MW)": None,
#             "Iterations": None,
#             "Solving Time (s)": None,
#             "Error": str(e)
#         })

# # Convert results to DataFrame
# results_df = pd.DataFrame(results)

# # Save the DataFrame to an Excel file
# results_df.to_excel("k_results_decomposition.xlsx", index=False)

