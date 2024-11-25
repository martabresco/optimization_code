import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math as m

from Data import investor_generation_data
from Data import rival_generation_data
from Data import lines_data
from Data import Wind_PF_data
from Data import PV_PF_data
from Data import Demand_profile
from Data import Demand_distribution
from Data import Demand_prices
from Data import Investment_data
from Data import Rival_scenarios
from Data import Demand_scenarios

nodes = list(range(1, 25))
K = 1.6e9  #in dollars, max investment budget

class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class InputData: #Idea: create one class for Input variables and one class for the optimization itself
    
    def __init__(
        self, 
        #existing investor generators data
        existing_investor_generator_id: list, #list with the number of each unit
        existing_investor_generator_node:dict[str,int],
        existing_investor_generator_capacity: dict[str, int],
        existing_investor_generator_cost: dict[str, int],     
        
        #existing rival generators data
        existing_rival_generator_id: list, #list with the number of each unit
        existing_rival_generator_node:dict[str,int],
        existing_rival_generator_capacity: dict[str, int],
        existing_rival_generator_cost: dict[str, int],

        #line data
        line_id:list,
        line_from:dict[str,int],
        line_to:dict[str,int],
        line_X:dict[str,int],
        line_capacity:dict[str,int],

        #Wind power factor
        hour:list,
        wind_PF: dict[str,int],
        
        #PV power factor
        PV_PF: dict[str,int],
        
        #Demand distribution
        demand_id: list,
        demand_node:dict[str,int],
        fraction_system_load:dict[str,int],
        
        #Demand price
        demand_price: dict[str,int],
        #Demand profile
        system_demand:dict[str,int],
        
        #Investment data
        technology_type:list,
        investment_cost:dict[str,int],
        max_investment_capacity:dict[str,int]
        
        #Doubt: do I also need to import demand scenarios and Rival scenarios as an attribute?
        

    ):
        # List of existing generators 
        self.existing_investor_generator_id = existing_investor_generator_id
        #Dictionary with connection node of each existing generator
        self.existing_investor_generator_node = existing_investor_generator_node
        #Dictionary with capacity of each existing generator
        self.existing_investor_generator_capacity = existing_investor_generator_capacity
        # Dictionary with each cost of existing geenrator
        self.existing_investor_generator_cost = existing_investor_generator_cost 
        self.existing_rival_generator_id = existing_rival_generator_id
        #Dictionary with connection node of each existing generator
        self.existing_rival_generator_node = existing_rival_generator_node
        #Dictionary with capacity of each existing generator
        self.existing_rival_generator_capacity = existing_rival_generator_capacity
        # Dictionary with each cost of existing geenrator
        self.existing_rival_generator_cost = existing_rival_generator_cost      
        self.line_id=line_id
        self.line_from=line_from
        self.line_to=line_to
        self.line_X=line_X
        self.line_capacity=line_capacity
        self.hour=hour
        self.PV_PF=PV_PF
        self.demand_id=demand_id
        self.demand_node=demand_node
        self.fraction_system_load=fraction_system_load
        self.demand_price=demand_price
        self.system_demand=system_demand
        self.technology_type=technology_type
        self.investment_cost=investment_cost
        self.max_investment_capacity=max_investment_capacity


class Optimal_Investment():

    def __init__(self, input_data: InputData, complementarity_method: str = 'Big M'): # initialize class
        self.data = input_data # define data attributes
        self.complementarity_method = complementarity_method # define method for complementarity conditions
        self.variables = Expando() # define variable attributes
        self.constraints = Expando() # define constraints attributes
        self.results = Expando() # define results attributes
        self._build_model() # build gurobi model
     
    def _build_variables(self):
    # lower-level primal variables
        self.variables.prod_new_conv_unit = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,  
                name=f'Electricity production of candidate  conventional unit  at node {n}, scenario {w} and hour {h}'
                )
            for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
            for h in self.data.hour        # Iterating over hours
            for n in self.data.nodes       # Iterating over nodes
            }
        self.variables.prod_PV ={
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,  # Upper bound is taken from Investment_data for this example
                name=f'Electricity production of candidate  PV unit  at node {n}, scenario {w} and hour {h}'
                )
            for w in self.data.Rival_scenarios.shape[1] # Assuming you have a list of generators
            for h in self.data.hour        # Iterating over hours
            for n in self.data.nodes       # Iterating over nodes
            } #production from new pv unit located in n, under scenario w and at time h
        self.variables.prod_wind ={
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,  # Upper bound is taken from Investment_data for this example
                name=f'Electricity production of candidate  wind unit  at node {n}, scenario {w} and hour {h}'
                )
            for w in self.data.Rival_scenarios.shape[1] # Assuming you have a list of generators
            for h in self.data.hour        # Iterating over hours
            for n in self.data.nodes       # Iterating over nodes
            }
        self.variables.prod_existing_conv ={
            (w, h, n,u): self.model.addVar(
                lb=0, 
                ub=investor_generation_data.iloc[u-1, 2], #maybe there is an error here
                name=f'Electricity production of existing conventional investor from unit {u} at node {n}, scenario {w} and hour {h}'
                )
            for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
            for h in self.data.hour        # Iterating over hours
            for n in self.data.nodes       # Iterating over nodes
            for u in self.data.existing_investor_generator_id
            }
        self.variables.prod_existing_rival={
            (w,h,n,u): self.model.addVar(
                lb=0,
                ub=rival_generation_data.iloc[u-1, 2],
                name = f'Electricity production of existing conventional from rival unit{u}, at node {n}, scenario {w} and hour {h}'
                )#maybe there is an error here)
            for w in self.data.Rival_scenarios.shape[1] # Assuming you have a list of generators
            for h in self.data.hour        # Iterating over hours
            for n in self.data.nodes       # Iterating over nodes
            for u in self.data.existing_rival_generator_id #iterating over units 
            }
        self.variables.prod_new_conv_rival={
            (w,h,n): self.model.addVar(
                lb=0,
                ub=Rival_scenarios.iloc[0,w], #maybe this is wrong
                name = f'Electricity production of new conventional from rival at node {n}, scenario {w} and hour {h}'
                )#maybe there is an error here)
            for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
            for h in self.data.hour        # Iterating over hours
            for n in self.data.nodes       # Iterating over nodes
            }
        self.variables.demand_consumed={
            (w,h,n): self.model.addVar(
                lb=0,
                ub=Demand_scenarios.iloc[h,w], #maybe this is wrong
                name = f'Electricity consumed by demand at node {n}, scenario {w} and hour {h}'
                )#maybe there is an error here)
            for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
            for h in self.data.hour        # Iterating over hours
            for n in self.data.nodes       # Iterating over nodes
            }
        self.variables.voltage_angle={
            (w,h,n): self.model.addVar(
                lb=-math.pi,
                ub=math.pi,
                name = f'Voltage angle at node {n}, scenario {w} and hour {h}'
                )
            }

        ## upper-level variables
        self.variables.cap_invest_conv = {
            n: self.model.addVar(
            lb=0,
            ub=GRB.INFINITY,
            name = f'Capacity investment in conventionals in node {n}')
        for n in self.data.nodes       # Iterating over nodes
        }
        self.variables.cap_invest_PV = {
            n: self.model.addVar(
            lb=0,
            ub=GRB.INFINITY,
            name = f'Capacity investment in PV in node {n}')
        for n in self.data.nodes      # Iterating over nodes
        }
        self.variables.cap_invest_Wind = {
            n: self.model.addVar(
            lb=0,
            ub=GRB.INFINITY,
            name = f'Capacity investment in wind in node {n}')
        for n in self.data.nodes       # Iterating over nodes
        }
        self.variables.conv_invest_bin={
            n: self.model.addVar(
                vtype=grb.GRB.BINARY, 
                name = f'Binary var, 1 if conventional investment in node {n}'
                )}
        self.variables.PV_invest_bin={
            n: self.model.addVar(
                vtype=grb.GRB.BINARY, 
                name = f'Binary var, 1 if PV investment in node {n}'
                )}
        self.variables.wind_invest_bin={
            n: self.model.addVar(
                vtype=grb.GRB.BINARY, 
                name = f'Binary var, 1 if wind investment in node {n}'
                )}
        self.variables.node_bin={
            n: self.model.addVar(
                vtype=grb.GRB.BINARY, 
                name = f'Binary var, 1 if any investment in node {n}'
                )}
        
        ##lower-level dual variables (1 for each constraint of the lower level problem)
        self.variables.lambda_dual ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'Lambda for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                } 
        self.variables.min_mu_conv_inv ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'Min mu for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.max_mu_conv_inv ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'Max mu for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1] # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.min_sigma_PV ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'Min sigma PV for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.max_sigma_PV ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'Max sigma PV for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.min_sigma_wind ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'Min sigma wind for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.max_sigma_wind ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'Max sigma wind for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1] # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.min_mu_existing ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'Min mu existing generators for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.max_mu_existing ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'Max mu existing generators for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.min_mu_rival ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'Min mu rival generators for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.max_mu_rival ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'max mu rival generators for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.min_mu_rival_new ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'min mu rival new generators for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.max_mu_rival_new ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'max mu rival new generators for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.min_sigma_demand ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'min sigma demand for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.gamma_f ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'line flow constraint dual variable gamma')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.min_epsilon_theta ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'min volatge angle dual for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.max_epsilon_theta ={
            (w,h,n):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'max volatge angle dual for node {n}, scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                for n in self.data.nodes       # Iterating over nodes
                }
        self.variables.ref_epsilon ={
            (w,h):self.model.addVar(
                lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                name = f'reference angle dual variable for scenario{w} and hour {h}')
                for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
                for h in self.data.hour        # Iterating over hours
                }


    



    def _build_upper_level_constraint(self):
        self.constraints.upper_level_max_inv_conv = {
            n: self.model.addConstr(
                self.variables.cap_invest_conv[n] == self.variables.node_bin * data.Investment_data.iloc[0,2],
                name = 'Max capacity investment for conventionals in node {n}'.format(n))
            for n in self.data.nodes
            }
        self.constraints.upper_level_max_num_investments_per_node={
            n: self.model.addConstr(
                self.variables.conv_invest_bin[n]+ self.variables.PV_invest_bin[n] + self.variables.wind_invest_bin[n]<= 3*self.variables.node_bin[n],
                name = 'We can only invest in one tech per node {n}'.format(n)
                )
            for n in self.data.nodes
            }
        self.constraint.upper_level_only_invest_one_node= self.model.addConstr(
            gp.quicksum(self.variables.node_bin[n] for n in self.data.nodes) <= 1,
            name = 'Only invest in one node of the system')
        
        self.constraint_upper_level_max_inv_PV = {
            n: self.model.addContr(
                self.variables.cap_invest_PV[n] <= self.variables.PV_invest_bin[n] * data.Investment_data.iloc[1,2],
                name = 'Max capacity investment for PV in node {n}'.format(n))
            for n in self.data.nodes
                }
        self.constraint_upper_level_max_inv_wind = {
            n: self.model.addContr(
                self.variables.cap_invest_Wind[n] <= self.variables.wind_invest_bin[n] * data.Investment_data.iloc[2,2],
                name = 'Max capacity investment for wind in node {n}'.format(n))
            for n in self.data.nodes
                }
        self.constraint.upper_level_max_investment_budget= self.model.addConstr(
            gp.quicksum(Investment_data.iloc[0,1]*self.variables.cap_invest_conv[n]+
                        Investment_data.iloc[1,1]*self.variables.cap_invest_PV[n]+
                        Investment_data.iloc[2,1]*self.variables.cap_invest_wind[n]  for n in self.data.nodes) <= K,
            name = 'Budget limit')
        




    def _build_kkt_primal_constraints(self):
        # build balance constraint
        # constraints from the 2nd level problem
    
    def _build_kkt_first_order_constraints(self):
        #lagrangian
        self.constraints.first_order_condition_generator_production = {
            g: self.model.addLConstr(
                self.data.generator_cost[g] - self.variables.balance_dual - self.variables.min_production_dual[g] + self.variables.max_production_dual[g],
                GRB.EQUAL,
                0,
                name='1st order condition - wrt to production {0}'.format(g)
            ) for g in self.data.GENERATORS
        } 
        self.constraints.first_order_condition_load_consumption = {
            d: self.model.addLConstr(
                - self.data.load_utility[d] + self.variables.balance_dual - self.variables.min_consumption_dual[d] + self.variables.max_consumption_dual[d],
                GRB.EQUAL,
                0,
                name='1st order condition - wrt to consumption {0}'.format(d)
            ) for d in self.data.LOADS
        }

    def _build_big_m_complementarity_conditions(self):
        # create auxiliary variables
        self.variables.complementarity_min_production_auxiliary = {
            g: self.model.addVar(
                vtype=GRB.BINARY, name='Auxiliary variable for complementarity condition on min. production constraint of generator {0}'.format(g)
            ) for g in self.data.GENERATORS
        }
        self.variables.complementarity_max_production_auxiliary = {
            g: self.model.addVar(
                vtype=GRB.BINARY, name='Auxiliary variable for complementarity condition on max. production constraint of generator {0}'.format(g)
            ) for g in self.data.GENERATORS
        } 
        self.variables.complementarity_min_consumption_auxiliary = {
            d: self.model.addVar(
                vtype=GRB.BINARY, name='Auxiliary variable for complementarity condition on min. consumption constraint of load {0}'.format(d)
            ) for d in self.data.LOADS
        } 
        self.variables.complementarity_max_consumption_auxiliary = {
            d: self.model.addVar(
                vtype=GRB.BINARY, name='Auxiliary variable for complementarity condition on max. consumption constraint of load {0}'.format(d)
            ) for d in self.data.LOADS
        }

        big_M = 10000

        # complementarity conditions related to production as constraints
        self.constraints.complementarity_max_production_mu = {
            g: self.model.addLConstr(
                self.variables.max_production_dual[g], 
                GRB.LESS_EQUAL,
                big_M * self.variables.complementarity_max_production_auxiliary[g]
            ) for g in self.data.GENERATORS
        }
        self.constraints.complementarity_max_production_gx = {
            g: self.model.addLConstr(
                self.data.generator_capacity[g] - self.variables.generator_production[g],
                GRB.LESS_EQUAL,
                big_M * (1 - self.variables.complementarity_max_production_auxiliary[g]),
            ) for g in self.data.GENERATORS if g != 'G1'
        }
        self.constraints.complementarity_max_production_g1 = self.model.addLConstr(
            self.variables.g1_production_DA - self.variables.generator_production['G1'],
            GRB.LESS_EQUAL,
            big_M * (1 - self.variables.complementarity_max_production_auxiliary['G1']),
        )
        self.constraints.complementarity_min_production_mu = {
            g: self.model.addLConstr(
                self.variables.min_production_dual[g], 
                GRB.LESS_EQUAL,
                big_M * self.variables.complementarity_min_production_auxiliary[g],
            ) for g in self.data.GENERATORS
        }
        self.constraints.complementarity_min_production_gx = {
            g: self.model.addLConstr(
                self.variables.generator_production[g],
                GRB.LESS_EQUAL,
                big_M * (1 - self.variables.complementarity_min_production_auxiliary[g])
            ) for g in self.data.GENERATORS
        }

        # complementarity conditions related to consumption as constraints
        self.constraints.complementarity_max_consumption_sigma = {
            d: self.model.addLConstr(
                self.variables.max_consumption_dual[d], 
                GRB.LESS_EQUAL,
                big_M * self.variables.complementarity_max_consumption_auxiliary[d]
            ) for d in self.data.LOADS
        }
        self.constraints.complementarity_max_consumption_lx = {
            d: self.model.addLConstr(
                self.data.load_capacity[d] - self.variables.load_consumption[d],
                GRB.LESS_EQUAL,
                big_M * (1 - self.variables.complementarity_max_consumption_auxiliary[d])
            ) for d in self.data.LOADS
        }
        self.constraints.complementarity_min_consumption_sigma = {
            d: self.model.addLConstr(
                self.variables.min_consumption_dual[d], 
                GRB.LESS_EQUAL,
                big_M * self.variables.complementarity_min_consumption_auxiliary[d]
            ) for d in self.data.LOADS
        }
        self.constraints.complementarity_min_consumption_lx = {
            d: self.model.addLConstr(
                self.variables.load_consumption[d],
                GRB.LESS_EQUAL,
                big_M * (1 - self.variables.complementarity_min_consumption_auxiliary[d])
            ) for d in self.data.LOADS
        }

    def _build_sos1_complementarity_conditions(self):
        # auxiliary variables (1 associated with each primal inequality constraint)
        self.variables.complementarity_max_production_auxiliary = {
            g: self.model.addVar(
                vtype=GRB.CONTINUOUS, name='Auxiliary variable for complementarity condition on max. production constraint of generator {0}'.format(g)
            ) for g in self.data.GENERATORS
        }  
        self.variables.complementarity_max_consumption_auxiliary = {
            d: self.model.addVar(
                vtype=GRB.CONTINUOUS, name='Auxiliary variable for complementarity condition on max. consumption constraint of load {0}'.format(d)
            ) for d in self.data.LOADS
        }

        # equality constraints setting the auxilliary variables equal to lhs of constraints. 
        self.constraints.complementarity_max_production_constraints = {
            g: self.model.addLConstr(
                self.variables.complementarity_max_production_auxiliary[g], 
                GRB.EQUAL,
                self.data.generator_capacity[g] - self.variables.generator_production[g],
            ) for g in self.data.GENERATORS if g != 'G1'
        }
        self.constraints.complementarity_max_production_constraints['G1'] = self.model.addLConstr(
            self.variables.complementarity_max_production_auxiliary['G1'], 
            GRB.EQUAL,
            self.variables.g1_production_DA - self.variables.generator_production['G1'],
        )
        self.constraints.complementarity_max_consumption_constraints = {
            d: self.model.addLConstr(
                self.variables.complementarity_max_consumption_auxiliary[d],
                GRB.EQUAL,
                self.data.load_capacity[d] - self.variables.load_consumption[d],
            ) for d in self.data.LOADS
        }

        # create SOS1 conditions 
        self.constraints.sos1_min_production = {
            g: self.model.addSOS(
                GRB.SOS_TYPE1, [self.variables.min_production_dual[g], self.variables.generator_production[g]]
            ) for g in self.data.GENERATORS
        }
        self.constraints.sos1_max_production = {
            g: self.model.addSOS(
                GRB.SOS_TYPE1, [self.variables.max_production_dual[g], self.variables.complementarity_max_production_auxiliary[g]]
            ) for g in self.data.GENERATORS
        }
        self.constraints.sos1_min_consumption = {
            d: self.model.addSOS(
                GRB.SOS_TYPE1, [self.variables.min_consumption_dual[d], self.variables.load_consumption[d]]
            ) for d in self.data.LOADS
        }
        self.constraints.sos1_max_consumption = {
            d: self.model.addSOS(
                GRB.SOS_TYPE1, [self.variables.max_consumption_dual[d], self.variables.complementarity_max_consumption_auxiliary[d]]
            ) for d in self.data.LOADS
        }

    def _build_kkt_complementarity_conditions(self):
        if self.complementarity_method == 'Big M':
            self._build_big_m_complementarity_conditions()
        elif self.complementarity_method == 'SOS1':
            self._build_sos1_complementarity_conditions()
        else: 
            raise NotImplementedError(
                "The complementarity_method has to be either 'Big M' (default) or 'SOS1'."
            )

    def _build_objective_function(self):
        objective = (
            - self.data.generator_cost['G1'] * self.variables.g1_production_DA 
            - gp.quicksum(
                self.data.generator_cost[g] * self.variables.generator_production[g] 
                + self.data.generator_capacity[g] * self.variables.max_production_dual[g]
                for g in self.data.GENERATORS if g != 'G1'
            )
            + gp.quicksum(
                self.data.load_utility[d] * self.variables.load_consumption[d]
                - self.data.load_capacity[d] * self.variables.max_consumption_dual[d]
                for d in self.data.LOADS
            )
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)

    def _build_model(self):
        self.model = gp.Model(name='Bilevel offering strategy')
        self._build_variables()
        self._build_upper_level_constraint()
        self._build_kkt_primal_constraints()
        self._build_kkt_first_order_constraints()
        self._build_kkt_complementarity_conditions()
        self._build_objective_function()
        self.model.update()
    
    def _save_results(self):
        # save objective value
        self.results.objective_value = self.model.ObjVal
        # save generator dispatch values
        self.results.generator_production = {
            g: self.variables.generator_production[g].x for g in self.data.GENERATORS
        }
        # save load consumption values
        self.results.load_consumption = {
            d: self.variables.load_consumption[d].x for d in self.data.LOADS
        }
        # save price (i.e., dual variable of balance constraint)
        self.results.price = self.variables.balance_dual.x
        # save strategic day-ahead offer of generator G1
        self.results.g1_offer = self.variables.g1_production_DA.x

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"optimization of {model.ModelName} was not successful")
    
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Optimal energy production cost:")
        print(self.results.objective_value)
        print("Optimal generator dispatches:")
        print(self.results.generator_production)
        print("Price at optimality:")
        print(self.results.price)
        print("Strategic offer by generator G1")
        print(self.results.g1_offer)







if __name__ == "__main__":
    input_data = InputData(
        existing_investor_generator_id=investor_generation_data.iloc[:,0].tolist(),
        existing_investor_generator_node=dict(zip(investor_generation_data.iloc[:,0].tolist(),investor_generation_data.iloc[:,1].tolist())),
        existing_investor_generator_cost=dict(zip(investor_generation_data.iloc[:,0].tolist(),investor_generation_data.iloc[:,3].tolist())),
        existing_investor_generator_capacity=dict(zip(investor_generation_data.iloc[:,0].tolist(),investor_generation_data.iloc[:,2].tolist())),
        
        existing_rival_generator_id=rival_generation_data.iloc[:,0].tolist(),
        existing_rival_generator_node=dict(zip(rival_generation_data.iloc[:,0].tolist(),rival_generation_data.iloc[:,1].tolist())),
        existing_rival_generator_cost=dict(zip(rival_generation_data.iloc[:,0].tolist(),rival_generation_data.iloc[:,3].tolist())),
        existing_rival_generator_capacity=dict(zip(rival_generation_data.iloc[:,0].tolist(),rival_generation_data.iloc[:,2].tolist())),
        
        line_id = lines_data["Line id"].tolist(),
        line_from=dict(zip(lines_data["Line id"].tolist(),lines_data["From"].tolist())),
        line_to=dict(zip(lines_data["Line id"].tolist(),lines_data["To"].tolist())),
        line_X=dict(zip(lines_data["Line id"].tolist(),lines_data["X (pu)"].tolist())),
        line_capacity=dict(zip(lines_data["Line id"].tolist(),lines_data["Capacity (MVA)"].tolist())),
        hour=Wind_PF_data["Hour"].tolist(),
        wind_PF=dict(zip(Wind_PF_data["Hour"].tolist(),Wind_PF_data["Onshore Wind"].tolist())),
        PV_PF=dict(zip(Wind_PF_data["Hour"].tolist(),PV_PF_data["PV"].tolist())),
        
        demand_id=Demand_distribution["Load #"].tolist(),
        demand_node=dict(zip(Demand_distribution["Load #"].tolist(),Demand_distribution["Node"].tolist())),
        fraction_system_load=dict(zip(Demand_distribution["Load #"].tolist(),Demand_distribution["% of system load"].tolist())),
        demand_price=dict(zip(Wind_PF_data["Hour"].tolist(),Demand_prices["$/MWh"].tolist())),    
        system_demand=dict(zip(Wind_PF_data["Hour"].tolist(),Demand_profile["System demand (MW)"].tolist())),  
        technology_type=Investment_data["Technology"].tolist(),
        investment_cost=dict(zip(Investment_data["Technology"].tolist(),Investment_data["Inv. Cost ($/MW)"].tolist())),
        max_investment_capacity=dict(zip(Investment_data["Technology"].tolist(),Investment_data["Max Inv. Capacity (MW)"].tolist()))
        
        
        
        
        
        
        
        
    )


