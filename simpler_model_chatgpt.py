import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math as m
from data_gpt import (
    investor_generation_data,
    rival_generation_data,
    lines_data,
    wind_PF,
    pv_PF,
    demand_profile,
    demand_distribution,
    demand_prices,
    investment_data,
    DA_prices,
    Df_demand,
    Df_rival,
    Omega_n_sets,
    capacity_matrix,
    susceptance_matrix
)

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

# Instancier l'objet avec les données importées
# input_data = InputData(
#     investor_generation_data=investor_generation_data,
#     rival_generation_data=rival_generation_data,
#     lines_data=lines_data,
#     wind_PF_data=wind_PF,
#     PV_PF_data=pv_PF,
#     demand_profile=demand_profile,
#     demand_distribution=demand_distribution,
#     demand_prices=demand_prices,
#     investment_data=investment_data,
#     DA_prices=DA_prices,
#     demand_scenarios=Df_demand,
#     rival_scenarios=Df_rival,
#     omega_node_set=Omega_n_sets,
#     capacity_matrix=capacity_matrix,
#     matrix_B=susceptance_matrix,
# )

import gurobipy as gp
from gurobipy import GRB
import math as m
#from InputData import InputData  # Ensure InputData is imported correctly

class Optimal_Investment:
    def __init__(self, input_data: InputData, complementarity_method: str = 'SOS1'):
        self.data = input_data  # Reference to the InputData instance
        self.complementarity_method = complementarity_method  # Complementarity method
        self.variables = Expando()  # Container for decision variables
        self.constraints = Expando()  # Container for constraints
        self.results = Expando()  # Container for results
        self.model = gp.Model("OptimalInvestment")  # Gurobi model
        self._build_variables()  # Define variables
        self._build_upper_level_constraint()
        #self._build # Define constraints

    def _build_variables(self):
        # Define production variables for new conventional units
        self.variables.prod_new_conv_unit = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Prod_New_Conv_Unit_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in self.data.rival_scenarios.columns  # Scenarios (columns in Df_rival)
            for h in range(1, 25)  # 24 hours
            for n in range(1, 25)  # Nodes
        }

        # Define production variables for PV units
        self.variables.prod_PV = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=self.data.investment_data["Max_Capacity"].max(),
                name=f"Prod_PV_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Define production variables for wind units
        self.variables.prod_wind = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=self.data.investment_data["Max_Capacity"].max(),
                name=f"Prod_Wind_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Define production variables for existing investor conventional units
        self.variables.prod_existing_conv = {
            (w, h, n, u): self.model.addVar(
                lb=0,
                ub=self.data.investor_generation_data.loc[u, "Pmax"],
                name=f"Prod_Existing_Conv_Unit{u}_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
            for u in self.data.investor_generation_data.index
        }

        # Define production variables for existing rival conventional units
        self.variables.prod_existing_rival = {
            (w, h, n, u): self.model.addVar(
                lb=0,
                ub=self.data.rival_generation_data.loc[u, "Pmax"],
                name=f"Prod_Existing_Rival_Unit{u}_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
            for u in self.data.rival_generation_data.index
        }

        # Define production variables for new rival conventional units
        self.variables.prod_new_conv_rival = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=self.data.rival_scenarios.loc["Capacity", w],
                name=f"Prod_New_Conv_Rival_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Define electricity consumption variables for demand
        self.variables.demand_consumed = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=self.data.demand_scenarios.loc[str(h), w],
                name=f"Demand_Consumed_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in self.data.demand_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Define voltage angle variables
        self.variables.voltage_angle = {
            (w, h, n): self.model.addVar(
                lb=-m.pi,
                ub=m.pi,
                name=f"Voltage_Angle_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in self.data.demand_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Define investment decision variables
        self.variables.cap_invest_conv = {
            n: self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Cap_Invest_Conv_Node{n}"
            )
            for n in range(1, 25)
        }

        self.variables.cap_invest_PV = {
            n: self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Cap_Invest_PV_Node{n}"
            )
            for n in range(1, 25)
        }

        self.variables.cap_invest_wind = {
            n: self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Cap_Invest_Wind_Node{n}"
            )
            for n in range(1, 25)
        }

        # Define binary investment decision variables
        self.variables.conv_invest_bin = {
            n: self.model.addVar(
                vtype=GRB.BINARY,
                name=f"Conv_Invest_Bin_Node{n}"
            )
            for n in range(1, 25)
        }

        self.variables.PV_invest_bin = {
            n: self.model.addVar(
                vtype=GRB.BINARY,
                name=f"PV_Invest_Bin_Node{n}"
            )
            for n in range(1, 25)
        }

        self.variables.wind_invest_bin = {
            n: self.model.addVar(
                vtype=GRB.BINARY,
                name=f"Wind_Invest_Bin_Node{n}"
            )
            for n in range(1, 25)
        }

        self.variables.node_bin = {
            n: self.model.addVar(
                vtype=GRB.BINARY,
                name=f"Node_Bin_Node{n}"
            )
            for n in range(1, 25)
        }

        def _build_upper_level_constraint(self):
            # Maximum capacity investment for conventional units at each node
            self.constraints.upper_level_max_inv_conv = {
                n: self.model.addConstr(
                    self.variables.cap_invest_conv[n] <= self.variables.conv_invest_bin[n] * self.data.investment_data.iloc[0, 2],
                    name=f"Max capacity investment for conventional units in node {n}"
                )
                for n in range(1, 25)  # Nodes 1-24
            }
        
            # Constraint ensuring at most one technology investment per node
            self.constraints.upper_level_max_num_investments_per_node = {
                n: self.model.addConstr(
                    self.variables.conv_invest_bin[n] + self.variables.PV_invest_bin[n] + self.variables.wind_invest_bin[n] <= 3 * self.variables.node_bin[n],
                    name=f"Max one technology investment per node {n}"
                )
                for n in range(1, 25)
            }
        
            # Constraint allowing investment in only one node across the entire system
            self.constraints.upper_level_only_invest_one_node = self.model.addConstr(
                gp.quicksum(self.variables.node_bin[n] for n in range(1, 25)) <= 1,
                name="Only one node can have investments in the system"
            )
        
            # Maximum capacity investment for PV units at each node
            self.constraints.upper_level_max_inv_PV = {
                n: self.model.addConstr(
                    self.variables.cap_invest_PV[n] <= self.variables.PV_invest_bin[n] * self.data.investment_data.iloc[1, 2],
                    name=f"Max capacity investment for PV in node {n}"
                )
                for n in range(1, 25)
            }
        
            # Maximum capacity investment for wind units at each node
            self.constraints.upper_level_max_inv_wind = {
                n: self.model.addConstr(
                    self.variables.cap_invest_wind[n] <= self.variables.wind_invest_bin[n] * self.data.investment_data.iloc[2, 2],
                    name=f"Max capacity investment for wind in node {n}"
                )
                for n in range(1, 25)
            }
        
            # Total investment budget constraint
            self.constraints.upper_level_max_investment_budget = self.model.addConstr(
                gp.quicksum(
                    self.data.investment_data.iloc[0, 1] * self.variables.cap_invest_conv[n] +
                    self.data.investment_data.iloc[1, 1] * self.variables.cap_invest_PV[n] +
                    self.data.investment_data.iloc[2, 1] * self.variables.cap_invest_wind[n]
                    for n in range(1, 25)
                ) <= self.data.investment_budget,
                name="Investment budget limit"
            )
            
            
        def _build_objective_function(self):
            # Assuming 'probability_scenario' is a list of probabilities for each scenario
            probability_scenario = [0.4 for _ in range(16)]  # Adjust based on actual probabilities if available
        
            # Define the investment cost term
            investment_cost = gp.quicksum(
                self.data.investment_cost["Conventional"] * self.variables.cap_invest_conv[n] +
                self.data.investment_cost["PV"] * self.variables.cap_invest_PV[n] +
                self.data.investment_cost["Wind"] * self.variables.cap_invest_wind[n]
                for n in range(1, 25)
            )
        
            # Define the revenue term (production revenue)
            production_revenue = gp.quicksum(
                probability_scenario[w] * gp.quicksum(
                    self.data.DA_prices[h] * (
                        self.variables.prod_new_conv_unit[w, h, n] +
                        self.variables.prod_existing_conv[w, h, n, u] +
                        self.variables.prod_PV[w, h, n] +
                        self.variables.prod_wind[w, h, n]
                    )
                    for n in range(1, 25)  # Nodes
                    for h in range(1, 25)  # Hours
                )
                for w in range(self.data.demand_scenarios.shape[1])  # Scenarios
                for u in self.data.investor_generation_data.index
            )
        
            # Set the objective as the minimization of total cost
            self.model.setObjective(investment_cost - production_revenue, GRB.MINIMIZE)
        
        
        def _build_model(self):
            self.model = gp.Model(name="Bilevel Offering Strategy")
            self._build_variables()
            self._build_upper_level_constraint()
            # self._build_kkt_primal_constraints()  # Define primal constraints for KKT conditions
            # self._build_kkt_first_order_constraints()  # First-order KKT conditions
            # self._build_kkt_complementarity_conditions()  # Complementarity conditions
            self._build_objective_function()  # Define the objective function
            self.model.update()  # Update the model with all changes
        
        
        def _save_results(self):
            # Save the objective value
            self.results.objective_value = self.model.ObjVal
        
            # Save generator production values
            self.results.generator_production = {
                (w, h, n): self.variables.prod_new_conv_unit[w, h, n].x
                for w in range(self.data.demand_scenarios.shape[1])
                for h in range(1, 25)
                for n in range(1, 25)
            }
        
            # Save load consumption
            self.results.load_consumption = {
                (w, h, n): self.variables.demand_consumed[w, h, n].x
                for w in range(self.data.demand_scenarios.shape[1])
                for h in range(1, 25)
                for n in range(1, 25)
            }


def prepare_input_data():
    return InputData(
        investor_generation_data=investor_generation_data,
        rival_generation_data=rival_generation_data,
        lines_data=lines_data,
        wind_PF_data=wind_PF,
        PV_PF_data=pv_PF,
        demand_profile=demand_profile,
        demand_distribution=demand_distribution,
        demand_prices=demand_prices,
        investment_data=investment_data,
        DA_prices=DA_prices,
        demand_scenarios=Df_demand,
        rival_scenarios=Df_rival,
        omega_node_set=Omega_n_sets,
        capacity_matrix=capacity_matrix,
        matrix_B=susceptance_matrix,
    )

if __name__ == "__main__":
    input_data = prepare_input_data()
    model = Optimal_Investment(input_data=input_data, complementarity_method='SOS1')
    model.run()
    model.display_results()




