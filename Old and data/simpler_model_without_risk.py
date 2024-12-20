import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math as m
import numpy as np


from data_simple_model import (
    investor_generation_data,
    wind_PF,
    pv_PF,
    investment_data,
    DA_prices
)

nodes = list(range(1, 25))
K = 1.6e5  #in dollars, max investment budget
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
        wind_PF_data: pd.DataFrame,
        PV_PF_data: pd.DataFrame,
        investment_data: pd.DataFrame,
        DA_prices: np.array
    ):
        self.investor_generation_data = investor_generation_data
        self.wind_PF_data = wind_PF_data
        self.PV_PF_data = PV_PF_data
        self.investment_data = investment_data
        self.DA_prices = DA_prices
        self.nb_scenarios=DA_prices.shape[0]

class Optimal_Investment:
    

    
    def __init__(self, input_data: InputData, complementarity_method: str = 'SOS1'):
        self.data = input_data  # Reference to the InputData instance
        self.variables = Expando()  # Container for decision variables
        self.constraints = Expando()  # Container for constraints
        self.results = Expando()  # Container for results
        self.model = gp.Model("OptimalInvestment")  # Gurobi model
        self._build_variables()  # Define variables
        self._build_upper_level_constraint()
        self._build_objective_function()
        
        self.run()
        self._save_results()

    def _build_variables(self):
        #former lower levels
        # Define production variables for new conventional units  Pc_nwt
        self.variables.prod_new_conv_unit = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Prod_New_Conv_Unit_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in range(0,self.data.nb_scenarios) # Scenarios (columns in Df_rival)
            for h in range(1, 25)  # 24 hours
            for n in range(1, 25)  # Nodes
        }

        # Define production variables for PV units, Pp_nwt
        self.variables.prod_PV = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Prod_PV_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in range(0,self.data.nb_scenarios)
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Define production variables for wind units Pw,nwt
        self.variables.prod_wind = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Prod_Wind_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in range(0,self.data.nb_scenarios)
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Define production variables for existing investor conventional units 
        self.variables.prod_existing_conv = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Prod_Existing_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in range(0,self.data.nb_scenarios)
            for h in range(1, 25)
            for n in range(1, 25)
        }


        # Define investment decision variables   #xnc
        self.variables.cap_invest_conv = {
            n: self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Cap_Invest_Conv_Node{n}"
            )
            for n in range(1, 25)
        }
        #xnP
        self.variables.cap_invest_PV = {
            n: self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Cap_Invest_PV_Node{n}"
            )
            for n in range(1, 25)
        }
        #xnW
        self.variables.cap_invest_wind = {
            n: self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Cap_Invest_Wind_Node{n}"
            )
            for n in range(1, 25)
        }

        # Define binary investment decision variables
        #unC
        self.variables.conv_invest_bin = {
            n: self.model.addVar(
                vtype=GRB.BINARY,
                name=f"Conv_Invest_Bin_Node{n}"
            )
            for n in range(1, 25)
        }
        #unP
        self.variables.PV_invest_bin = {
            n: self.model.addVar(
                vtype=GRB.BINARY,
                name=f"PV_Invest_Bin_Node{n}"
            )
            for n in range(1, 25)
        }
        #unW
        self.variables.wind_invest_bin = {
            n: self.model.addVar(
                vtype=GRB.BINARY,
                name=f"Wind_Invest_Bin_Node{n}"
            )
            for n in range(1, 25)
        }
        #un
        self.variables.node_bin = {
            n: self.model.addVar(
                vtype=GRB.BINARY,
                name=f"Node_Bin_Node{n}"
            )
            for n in range(1, 25)
        }

    def _build_upper_level_constraint(self):
        K = 1.6e9 # budget 
        # Maximum capacity investment for conventional units at each node
        self.constraints.upper_level_max_inv_conv = {
            n: self.model.addConstr(
                self.variables.cap_invest_conv[n] == self.variables.conv_invest_bin[n] * self.data.investment_data.iloc[0, 2],
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
            ) <= K,
            name="Investment budget limit"
        )
        
        
        # Production limits for new conventional units
        self.constraints.production_limits_con = {
            (w, h, n): self.model.addConstr(
                self.variables.prod_new_conv_unit[w, h, n] <= self.variables.cap_invest_conv[n],
                name=f"Prod limit for new conv. unit at node {n}, scenario {w}, hour {h}"
            )
            for w in range(0,self.data.nb_scenarios)
            for h in range(1, 25)
            for n in range(1, 25)
        }
        
        # Production limits for PV units
        self.constraints.production_limits_PV = {
            (w, h, n): self.model.addConstr(
                self.variables.prod_PV[w, h, n] <= 
                self.data.PV_PF_data.iloc[h - 1, 1] * self.variables.cap_invest_PV[n],
                name=f"Prod limit for PV at node {n}, scenario {w}, hour {h}"
            )
            for w in range(0,self.data.nb_scenarios)
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Production limits for wind units
        self.constraints.production_limits_wind = {
            (w, h, n): self.model.addConstr(
                self.variables.prod_wind[w, h, n] <= 
                self.data.wind_PF_data.iloc[h - 1, 1] * self.variables.cap_invest_wind[n],
                name=f"Prod limit for wind at node {n}, scenario {w}, hour {h}"
            )
            for w in range(0,self.data.nb_scenarios)
            for h in range(1, 25)
            for n in range(1, 25)
        }
        # Production limits for existing conventional units
        self.constraints.production_limits_existing_con = {
            (w, h, n): self.model.addConstr(
                self.variables.prod_existing_conv[w, h, n] <= 
                self.data.investor_generation_data.loc[
                    self.data.investor_generation_data["Node"] == n, "Pmax"
                ].values[0],
                name=f"Prod limit for existing conv. at node {n}, scenario {w}, hour {h}"
            )
            for w in range(0,self.data.nb_scenarios)  # Iterate over scenarios
            for h in range(1, 25)  # Iterate over hours
            for n in self.data.investor_generation_data["Node"].unique()  # Only nodes with existing generators
        }



    def _build_objective_function(self):
        # Assuming 'probability_scenario' is a list of probabilities for each scenario
        #probability_scenario=[0.06,0.06,0.06,0.02,0.06,0.06,0.06,0.02,0.09,0.09,0.09,0.03,0.09,0.09,0.09,0.03] # Adjust based on actual probabilities if available
        probability_scenario=1/20
    
        investment_cost = gp.quicksum(
            self.data.investment_data.loc[self.data.investment_data["Technology"] == "Conventional", "Inv_Cost"].values[0] * self.variables.cap_invest_conv[n] +
            self.data.investment_data.loc[self.data.investment_data["Technology"] == "PV", "Inv_Cost"].values[0] * self.variables.cap_invest_PV[n] +
            self.data.investment_data.loc[self.data.investment_data["Technology"] == "Wind", "Inv_Cost"].values[0] * self.variables.cap_invest_wind[n]
            for n in range(1, 25)
        )


        production_revenue = 20*365*gp.quicksum(
            probability_scenario*(
            self.data.DA_prices[w,n-1,h-1] *(  # Adjust h for 0-indexed DA_prices
                self.variables.prod_new_conv_unit[(w, h, n)] +
                (
                    self.variables.prod_existing_conv[(w, h, n)]
                    if n in self.data.investor_generation_data["Node"].values
                    else 0  # Skip if there is no existing generator at node n
                ) +
                self.variables.prod_PV[(w, h, n)] +
                self.variables.prod_wind[(w, h, n)]
            )
            - (
                self.variables.prod_new_conv_unit[(w, h, n)] * cand_Conv_cost  # Subtracting investment cost for new conventional units
                + (
                    self.variables.prod_existing_conv[(w, h, n)] * self.data.investor_generation_data.loc[
                        self.data.investor_generation_data["Node"] == n, "Bid price"
                    ].values[0]
                    if n in self.data.investor_generation_data["Node"].values
                    else 0  # Skip cost if there is no existing generator at node n
                )
            ))
            for (w, h, n) in self.variables.prod_new_conv_unit.keys() ) # Iterate over all keys
        



    
        # Set the objective as the minimization of total cost
        self.model.setObjective(investment_cost - production_revenue, GRB.MINIMIZE)
        
    def run(self):
       self.model.optimize()
       if self.model.status == GRB.OPTIMAL:
           print("Optimization completed successfully.")
           self._save_results()
       else:
           print("Optimization was not successful.")
              
        
    def _save_results(self):
        # Save the objective value
        self.results.objective_value = self.model.ObjVal
    
        # Save generator production values
        self.results.generator_production = {
            (w, h, n): self.variables.prod_new_conv_unit[(w, h, n)].x
            for (w, h, n) in self.variables.prod_new_conv_unit.keys()
        }
    

    def display_results(self):
        print("\n-------------------   RESULTS  -------------------")
        
        # Objective value
        print(f"Optimal Objective Value (Total Cost): {self.results.objective_value:.2f}")
        
        # Investment Decisions
        print("\nInvestment Decisions:")
        for n in range(1, 25):  # Assuming 24 nodes
            print(f"Node {n}:")
            if self.variables.cap_invest_conv[n].x > 0:
                print(f"  - Conventional Capacity: {self.variables.cap_invest_conv[n].x:.2f} MW")
            if self.variables.cap_invest_PV[n].x > 0:
                print(f"  - PV Capacity: {self.variables.cap_invest_PV[n].x:.2f} MW")
            if self.variables.cap_invest_wind[n].x > 0:
                print(f"  - Wind Capacity: {self.variables.cap_invest_wind[n].x:.2f} MW")
            if self.variables.node_bin[n].x > 0:
                print("  - Investment Active in Node")
        
        # # Production Results
        # print("\nProduction Results:")
        # print("New Conventional Production:")
        # for (w, h, n), var in self.variables.prod_new_conv_unit.items():
        #     if var.x > 0:
        #         print(f"Scenario {w}, Hour {h}, Node {n}: {var.x:.2f} MW")
        
        # print("\nPV Production:")
        # for (w, h, n), var in self.variables.prod_PV.items():
        #     if var.x > 0:
        #         print(f"Scenario {w}, Hour {h}, Node {n}: {var.x:.2f} MW")
        
        # print("\nWind Production:")
        # for (w, h, n), var in self.variables.prod_wind.items():
        #     if var.x > 0:
        #         print(f"Scenario {w}, Hour {h}, Node {n}: {var.x:.2f} MW")
        
        # print("\nExisting Conventional Production:")
        # for (w, h, n, u), var in self.variables.prod_existing_conv.items():
        #     if var.x > 0:
        #         print(f"Scenario {w}, Hour {h}, Node {n}, Unit {u}: {var.x:.2f} MW")
        
        # print("\nExisting Rival Production:")
        # for (w, h, n, u), var in self.variables.prod_existing_rival.items():
        #     if var.x > 0:
        #         print(f"Scenario {w}, Hour {h}, Node {n}, Unit {u}: {var.x:.2f} MW")
        
        # print("\nNew Rival Conventional Production:")
        # for (w, h, n), var in self.variables.prod_new_conv_rival.items():
        #     if var.x > 0:
        #         print(f"Scenario {w}, Hour {h}, Node {n}: {var.x:.2f} MW")
        
        # # Demand Served
        # print("\nDemand Served:")
        # for (w, h, n), var in self.variables.demand_consumed.items():
        #     if var.x > 0:
        #         print(f"Scenario {w}, Hour {h}, Node {n}: {var.x:.2f} MW")
        
        # # Voltage Angles
        # print("\nVoltage Angles:")
        # for (w, h, n), var in self.variables.voltage_angle.items():
        #     print(f"Scenario {w}, Hour {h}, Node {n}: Voltage Angle = {var.x:.4f} radians")


    
    



def prepare_input_data():
    return InputData(
        investor_generation_data=investor_generation_data,
        wind_PF_data=wind_PF,
        PV_PF_data=pv_PF,
        investment_data=investment_data,
        DA_prices=DA_prices
    )

if __name__ == "__main__":
    input_data = prepare_input_data()
    model = Optimal_Investment(input_data=input_data, complementarity_method='SOS1')
    model.run()
    model.display_results()




