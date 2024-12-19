import gurobipy as grb
from gurobipy import GRB
#import pandas as pd
import math as math
from gurobipy import quicksum

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
#from Data import Omega_n_sets
from Data import matrix_B
from Data import capacity_matrix
from Data import DA_prices_data

nodes = list(range(1, 25))
K = 1.6e9  #in dollars, max investment budget
cand_Conv_cost=7.24;

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
        max_investment_capacity:dict[str,int],
        
        #Day-ahead price
        DA_price_data: dict[int,list]
        
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
        self.DA_price_data = DA_price_data
        

class Optimal_Investment():

    def __init__(self, input_data: InputData, complementarity_method: str = 'SOS1'): # initialize class
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
        self.variables.prod_existing_conv = { 
            (w, h, n): self.model.addVar(
           lb=0, 
           ub=GRB.INFINITY,
           name=f'Electricity production of existing conventional investor at node {n}, scenario {w} and hour {h}'
           )
            for w in range(self.data.Rival_scenarios.shape[1])  # Iterating over scenarios
            for h in self.data.hour        # Iterating over hours
            for n in self.data.nodes       # Iterating over nodes
            }
        self.variables.prod_existing_rival={
            (w,h,n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name = f'Electricity production of existing conventional from rival unit at node {n}, scenario {w} and hour {h}'
                )#maybe there is an error here)
            for w in self.data.Rival_scenarios.shape[1] # Assuming you have a list of generators
            for h in self.data.hour        # Iterating over hours
            for n in self.data.nodes       # Iterating over nodes
            }
        self.variables.prod_new_conv_rival={
            (w,h,n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY, 
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
            for w in self.data.Rival_scenarios.shape[1]  # Assuming you have a list of generators
            for h in self.data.hour        # Iterating over hours
            for n in self.data.nodes       # Iterating over nodes
            }

        
        # from upper level
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
                )
            for n in self.data.nodes  }
        self.variables.PV_invest_bin={
            n: self.model.addVar(
                vtype=grb.GRB.BINARY, 
                name = f'Binary var, 1 if PV investment in node {n}'
                )
            for n in self.data.nodes  }
        self.variables.wind_invest_bin={
            n: self.model.addVar(
                vtype=grb.GRB.BINARY, 
                name = f'Binary var, 1 if wind investment in node {n}'
                )
            for n in self.data.nodes  }
        self.variables.node_bin={
            n: self.model.addVar(
                vtype=grb.GRB.BINARY, 
                name = f'Binary var, 1 if any investment in node {n}'
                )
            for n in self.data.nodes  }
        
        


    



    def _build_upper_level_constraint(self):
        self.constraints.upper_level_max_inv_conv = {
            n: self.model.addConstr(
                self.variables.cap_invest_conv[n] == self.variables.node_bin * Investment_data.iloc[0,2],
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
            grb.quicksum(self.variables.node_bin[n] for n in self.data.nodes) <= 1,
            name = 'Only invest in one node of the system')
        
        self.constraint_upper_level_max_inv_PV = {
            n: self.model.addContr(
                self.variables.cap_invest_PV[n] <= self.variables.PV_invest_bin[n] * Investment_data.iloc[1,2],
                name = 'Max capacity investment for PV in node {n}'.format(n))
            for n in self.data.nodes
                }
        self.constraint_upper_level_max_inv_wind = {
            n: self.model.addContr(
                self.variables.cap_invest_Wind[n] <= self.variables.wind_invest_bin[n] * Investment_data.iloc[2,2],
                name = 'Max capacity investment for wind in node {n}'.format(n))
            for n in self.data.nodes
                }
        self.constraint.upper_level_max_investment_budget= self.model.addConstr(
            grb.quicksum(Investment_data.iloc[0,1]*self.variables.cap_invest_conv[n]+
                        Investment_data.iloc[1,1]*self.variables.cap_invest_PV[n]+
                        Investment_data.iloc[2,1]*self.variables.cap_invest_wind[n]  for n in self.data.nodes) <= K,
            name = 'Budget limit')
        
        
        
        self.constraints.power_balance = {
            (w, h, n): self.model.addConstr(
                    self.variables.demand_consumed[w, h, n] +
                    quicksum(matrix_B[n, m] * (self.variables.voltage_angle[w, h, n] - self.variables.voltage_angle[w, h, m])
                             for m in self.data.nodes if m != n) 
                    - self.variables.prod_new_conv_unit[w, h, n] 
                    - self.variables.prod_PV[w, h, n] 
                    - self.variables.prod_wind[w, h, n] 
                    - self.variables.prod_existing_conv[w, h, n] 
                    - self.variables.prod_existing_rival[w, h, n] 
                    - self.variables.prod_new_conv_rival[w, h, n] == 0,
                    name='Power balance at node {n} for scenario {w} at time {h}'.format(w=w, h=h, n=n)
                    )
            for w in range(self.data.Rival_scenarios.shape[1])  # Assuming you have a list of generators
            for h in range(self.data.hour)        # Iterating over hours
            for n in self.data.nodes             # Iterating over nodes
                }
            
        
        self.constraints.production_limits_con = {
            (w, h, n): self.model.addConstr(
                0<=self.variables.prod_new_conv_unit[w,h,n]<=self.variables.cap_invest_conv[n],
                name = 'Production limits new conventional unit in node {n} scenario {w} at hour {h}'.format(n,w,h))
            for w in range(self.data.Rival_scenarios.shape[1])  # Assuming you have a list of generators
            for h in range(self.data.hour)        # Iterating over hours
            for n in self.data.nodes             # Iterating over nodes
            }

        self.constraints.production_limits_PV = {
            (w, h, n): self.model.addConstr(
                0<=self.variables.prod_PV[w,h,n]<=PV_PF_data.iloc[:, 1]*self.variables.cap_invest_PV[n],
                name = 'Production limits new PV unit in node {n}, scenario {w}, at hour {h}'.format(n,w,h))
            for w in range(self.data.Rival_scenarios.shape[1])  # Assuming you have a list of generators
            for h in range(self.data.hour)        # Iterating over hours
            for n in self.data.nodes             # Iterating over nodes
            }
        
        self.constraints.production_limits_wind = {
            (w, h, n): self.model.addConstr(
                0<=self.variables.prod_wind[w,h,n]<=Wind_PF_data.iloc[:, 1]*self.variables.cap_invest_wind[n],
                name = 'Production limits new wind unit in node {n}, scenario {w}, at hour {h}'.format(n,w,h))
            for w in range(self.data.Rival_scenarios.shape[1])  # Assuming you have a list of generators
            for h in range(self.data.hour)        # Iterating over hours
            for n in self.data.nodes             # Iterating over nodes
            }
        
        
        
       
        self.constraints.production_limits_existing_con = {
            (w, h, n): self.model.addConstr(
                0 <= self.variables.prod_existing_conv[w, h, n] <= investor_generation_data.iloc[
               investor_generation_data.iloc[:, 1] == n, 2].values[0],  # Accessing the value in column 2
                name='Production limits existing conventional unit in node {n} scenario {w} at hour {h}'.format(n=n, w=w, h=h)
                )
            for w in range(self.data.Rival_scenarios.shape[1])  # Iterating over scenarios
            for h in range(self.data.hour)  # Iterating over hours
            for n in self.data.nodes  # Iterating over nodes
            }
     

        self.constraints.production_limits_existing_rival = {
            (w, h, n): self.model.addConstr(
                0<=self.variables.prod_existing_rival[w,h,n]<=rival_generation_data.iloc[rival_generation_data.iloc[:,1]==n,2],
                name = 'Production limits existing rival unit  in node {n} scenario {w} at hour {h}'.format(n,w,h))
            for w in range(self.data.Rival_scenarios.shape[1])  # Assuming you have a list of generators
            for h in range(self.data.hour)        # Iterating over hours
            for n in self.data.nodes             # Iterating over nodes
            }
        
        self.constraints.production_limits_new_rival = {
            (w, h, n): self.model.addConstr(
                0 <= self.variables.prod_new_conv_riva[w, h, n] <= Rival_scenarios.iloc[0, :],
                name='Production limits new new unit in node {n}, scenario {w}, at hour {h}'.format(n=n, w=w, h=h))
            for w in range(self.data.Rival_scenarios.shape[1])  # Iterating over scenarios
            for h in range(self.data.hour)        # Iterating over hours
            for n in self.data.nodes             # Iterating over nodes
            if n == 23  
            }
        
        
        
        
        self.constraints.node_limits_new_rival = {
            (w, h, n): self.model.addConstr(
                self.variables.prod_new_conv_riva[w, h, n] ==0,
                name='limits new rival unit to node 23')
            for w in range(self.data.Rival_scenarios.shape[1])  # Iterating over scenarios
            for h in range(self.data.hour)        # Iterating over hours
            for n in self.data.nodes             # Iterating over nodes
            if n != 23 
            }

        
        node_to_percentage = Demand_distribution.set_index(1)[2]  # Now node_to_percentage[n] gives the percentage for node n

        self.constraints.Demand_limit = {
            (w, h, n): self.model.addConstr(
                # The demand consumed at a node must be within the range [0, max load for the node]
                self.variables.demand_consumed[w, h, n] <= Demand_scenarios.iloc[:, w-1] * node_to_percentage[n],  
                # Multiply the total system load for scenario w by the percentage assigned to node n
                name='Demand limits  node {n}, scenario {w}, at hour {h}'.format(n=n, w=w, h=h)
                )
            for w in range(self.data.Demand_scenarios.shape[1])  # Iterate over all scenarios (columns in Demand_scenarios)
            for h in range(self.data.hour)  # Iterate over all hours
            for n in self.data.nodes if n in node_to_percentage.index  # Iterate only over nodes present in Demand_distribution
            }
        
        # Constraint for line power flows based on voltage angle differences and line reactance
        self.constraints.line_power_flow = { 
            (w, h, n, m): self.model.addConstr(
                # Power flow is determined by the voltage angle difference and line reactance
                matrix_B[n, m] * (self.variables.voltage_angle[w, h, n] - self.variables.voltage_angle[w, h, m]) <=
                capacity_matrix[n - 1, m - 1],  # Capacity between node n and node m from capacity_matrix
                name='Power flow on line {n}-{m}, scenario {w}, hour {h}'.format(n=n, m=m, w=w, h=h)
                )
            for w in range(self.data.Rival_scenarios.shape[1])  # Iterate over all scenarios
            for h in range(self.data.hour)  # Iterate over all hours
            for n in self.data.nodes       # Iterate over all nodes
            for m in self.data.nodes if m != n  # Iterate over all connected nodes (ensuring m != n)
            }
        
        
        # Constraint to limit voltage angles between -π and +π
        self.constraints.voltage_angle_limits = {
            (w, h, n): (
                self.model.addConstr(
                    self.variables.voltage_angle[w, h, n] >= -math.pi,
                    name='Voltage angle lower limit at node {n}, scenario {w}, hour {h}'.format(n=n, w=w, h=h)
                    ),
                self.model.addConstr(
                    self.variables.voltage_angle[w, h, n] <= math.pi,
                    name='Voltage angle upper limit at node {n}, scenario {w}, hour {h}'.format(n=n, w=w, h=h)
                    )
                )
            for w in range(self.data.Rival_scenarios.shape[1])  # Iterate over all scenarios
            for h in range(self.data.hour)  # Iterate over all hours
            for n in self.data.nodes       # Iterate over all nodes
            }
        
        
        # Constraint to set voltage angle to 0 for node 1
        self.constraints.voltage_angle_fixed_node1 = {
            (w, h): self.model.addConstr(
                self.variables.voltage_angle[w, h, 1] == 0,
                name='Voltage angle fixed to 0 at node 1, scenario {w}, hour {h}'.format(w=w, h=h)
                )
            for w in range(self.data.Rival_scenarios.shape[1])  # Iterate over all scenarios
            for h in range(self.data.hour)  # Iterate over all hours
            }
        
        def _build_objective_function(self):
            # Assuming 'probability_scenario' is meant to be a list of probabilities for each scenario
            probability_scenario = [0.4 for _ in range(16)]  # This should be updated based on the actual scenario probabilities
    
            objective = (
            # Investment costs
                quicksum(Investment_data.iloc[0, 1] * self.variables.cap_invest_conv(n) for n in self.data.nodes) +
                quicksum(Investment_data.iloc[1, 1] * self.variables.PV_invest_bin(n) for n in self.data.nodes) +
                quicksum(Investment_data.iloc[2, 1] * self.variables.wind_invest_bin(n) for n in self.data.nodes) +

                # Revenue terms (from production, based on scenario and hour)
                - quicksum(
                    probability_scenario[w] *  # For each scenario, multiply by the probability
                    quicksum(
                        DA_prices_data(h) * quicksum(
                            # Summing production values across all nodes for each hour
                            self.variables.prod_new_conv_unit(n, w, h) +
                            self.variables.prod_existing_conv(n, w, h) +
                            self.variables.prod_PV(n, w, h) +
                            self.variables.prod_wind(n, w, h)
                            - self.variables.prod_new_conv_unit(n, w, h) * Investment_data.iloc[0, 1]  # Subtracting investment cost for new conventional units
                            - self.variables.prod_existing_conv(n, w, h) * investor_generation_data.iloc[:, 3]  # Subtracting cost for existing conventional units (cost in the 4th column of `investor_generation_data`)
                            for n in self.data.nodes  # Summing over all nodes
                            )
                        for h in range(self.data.hour)  # Iterate over all hours
                        )
                    for w in range(self.data.Rival_scenarios.shape[1])  # Iterate over all scenarios
                ))
            self.model.setObjective(objective, GRB.MINIMIZE)





    def _build_model(self):
        self.model = grb.Model(name='Bilevel offering strategy')
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
    
    DA_prices_data = DA_prices_data.iloc[:, 1].tolist()

    # Sett opp inputdataene
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
        max_investment_capacity=dict(zip(Investment_data["Technology"].tolist(),Investment_data["Max Inv. Capacity (MW)"].tolist())),
        DA_price_data=DA_prices_data
 
    )
    
    # Nå kan vi bruke Optimal_Investment objektet
    model = Optimal_Investment(input_data, complementarity_method='SOS1')
    
    model.run()
    model.display_results()

