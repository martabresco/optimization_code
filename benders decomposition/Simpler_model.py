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
from Data import Omega_n_sets
from Data import DA_price

nodes = list(range(1, 25))
K = 1.6e9  #in dollars, max investment budget
cand_Conv_cost=7.24;

class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class InputData:
    def __init__(
        self,
        investor_generation_data: pd.DataFrame,
        rival_generation_data: pd.DataFrame,
        lines_data: pd.DataFrame,
        wind_PF_data: pd.DataFrame,
        PV_PF_data: pd.DataFrame,
        demand_profile: pd.DataFrame,
        demand_distribution: pd.DataFrame,
        demand_prices: pd.DataFrame,
        investment_data: pd.DataFrame,
        DA_prices: pd.DataFrame,
        demand_scenarios: pd.DataFrame,
        rival_scenarios: pd.DataFrame,
        omega_node_set: dict[int, list],
        capacity_matrix: pd.DataFrame,
        matrix_B: pd.DataFrame,
    ):
        self.investor_generation_data = investor_generation_data
        self.rival_generation_data = rival_generation_data
        self.lines_data = lines_data
        self.wind_PF_data = wind_PF_data
        self.PV_PF_data = PV_PF_data
        self.demand_profile = demand_profile
        self.demand_distribution = demand_distribution
        self.demand_prices = demand_prices
        self.investment_data = investment_data
        self.DA_prices = DA_prices
        self.demand_scenarios = demand_scenarios
        self.rival_scenarios = rival_scenarios
        self.omega_node_set = omega_node_set
        self.capacity_matrix = capacity_matrix
        self.matrix_B = matrix_B


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
        
        
        self.constraints.power_balance = {
            (w, h, n): self.model.addConstr(
                    self.variables.demand_consumed[w, h, n] +
                    quicksum(1/line_X[n, m] * (self.variables.voltage_angle[w, h, n] - self.variables.voltage_angle[w, h, m])
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
        
        
        
        ##### Pretty sure this is wrong   #### Need to figure out how to fix it regarding more than one generator per node and so on.. also see if the rival scenario is right
        
        self.constraints.production_limits_existing_con = {
            (w, h, n): self.model.addConstr(
                0<=self.variables.prod_existing_conv[w,h,n,u]<=investor_generation_data.iloc[:,2], #might be wrong!! 
                name = 'Production limits existing conventional unit {u} in node {n} scenario {w} at hour {h}'.format(u,n,w,h))
            for w in range(self.data.Rival_scenarios.shape[1])  # Assuming you have a list of generators
            for h in range(self.data.hour)        # Iterating over hours
            for n in self.data.nodes             # Iterating over nodes
            for u in self.data.existing_rival_generator_id  # Iterating over units
            }      

        self.constraints.production_limits_existing_rival = {
            (w, h, n): self.model.addConstr(
                0<=self.variables.prod_existing_rival[w,h,n,u]<=rival_generation_data.iloc[:,2], #might be wrong!! 
                name = 'Production limits existing rival unit {u} in node {n} scenario {w} at hour {h}'.format(u,n,w,h))
            for w in range(self.data.Rival_scenarios.shape[1])  # Assuming you have a list of generators
            for h in range(self.data.hour)        # Iterating over hours
            for n in self.data.nodes             # Iterating over nodes
            for u in self.data.existing_rival_generator_id  # Iterating over units
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
        
        ##########################################
        
        
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

        self.constraints.production_limits_con = {
            (w, h, n): self.model.addConstr(
                # The demand consumed at a node must be within the range [0, max load for the node]
                0 <= self.variables.demand_consumed[w, h, n] <= Demand_scenarios.iloc[:, w-1] * node_to_percentage[n],  
                # Multiply the total system load for scenario w by the percentage assigned to node n
                name='Production limits new conventional unit in node {n}, scenario {w}, at hour {h}'.format(n=n, w=w, h=h)
                )
            for w in range(self.data.Rival_scenarios.shape[1])  # Iterate over all scenarios (columns in Demand_scenarios)
            for h in range(self.data.hour)  # Iterate over all hours
            for n in self.data.nodes if n in node_to_percentage.index  # Iterate only over nodes present in Demand_distribution
            }
        
        # Constraint for line power flows based on voltage angle differences and line reactance
        self.constraints.line_power_flow = {
            (w, h, n, m): self.model.addConstr(
                # Power flow is determined by the voltage angle difference and line reactance
                1 / line_X[n, m] * (self.variables.voltage_angle[w, h, n] - self.variables.voltage_angle[w, h, m]) <=
                lines_data[(lines_data.iloc[:, 1] == n) & (lines_data.iloc[:, 2] == m)].iloc[0, 3],  # Line capacity for (n, m)
                name='Power flow on line {n}-{m}, scenario {w}, hour {h}'.format(n=n, m=m, w=w, h=h)
                )
            for w in range(self.data.Rival_scenarios.shape[1])  # Iterate over all scenarios
            for h in range(self.data.hour)  # Iterate over all hours
            for n in self.data.nodes       # Iterate over all nodes
            for m in self.data.nodes if m != n  # Iterate over all connected nodes
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
        
        
        ##### Need to fix the last one for the existing generation costs. 
    def _build_objective_function(self):
       probability_scenario = [0.4 for _ in range(16)] ##### must be canged to the ones for each scenario!!
        objective = (
            quicksum(Investment_data.iloc[0,1]*self.variables.cap_invest_conv(n) for n in self.data.nodes)
            + Investment_data.iloc[1,1]*self.variables.PV_invest_bin(n)
            +Investment_data.iloc[2,1]*self.variables.wind_invest_bin(n)
            -quicksum(probability_scenario(w)* quicksum(DA_prices(h)*quicksum(self.variables.prod_new_conv_unit(n,w,h)
            +self.variables.prod_existing_conv(n,w,h)+self.variables.prod_PV(n,w,h)+self.variables.prod_wind(n,w,h)
            -self.variables.prod_new_conv_unit(n,w,t)*Investment_data.iloc[0,1]-self.variables.prod_existing_conv(n,w,h)*investor_generation_data.iloc[:,3]))
        )
        self.model.setObjective(objective, GRB.MINIMIZE)

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
        max_investment_capacity=dict(zip(Investment_data["Technology"].tolist(),Investment_data["Max Inv. Capacity (MW)"].tolist())),
        #omega_node_set=Omega_n_sets
        
        DA_price=dict(zip(Wind_PF_data["Hour"].tolist(), DA_price["Price"].tolist()))  
        
        
        model = EconomicDispatch(input_data, complementarity_method='SOS1')
        
        model.run()
        model.display_results()
        
        
    )

