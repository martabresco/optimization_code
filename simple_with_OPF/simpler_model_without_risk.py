import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math as m
from itertools import product
from scipy.stats import norm
import numpy as np


from data_simple_model import (
    investor_generation_data,
    rival_generation_data,
    lines_data,
    wind_PF,
    pv_PF,
    demand_profile,
    demand_distribution,
    demand_prices,
    investment_data,
    Df_demand,
    Df_rival,
    Omega_n_sets,
    capacity_matrix,
    susceptance_matrix,
    DA_prices
)


nodes = list(range(1, 25))
K = 1.6e10  #in dollars, max investment budget
cand_Conv_cost=55

Demand = np.zeros((24, 24))  # 24x24 numpy array
# Assuming demand_distribution and demand_profile are pandas DataFrames
# Access their elements using .iloc[row, column]
for n in demand_distribution["Node"].unique():
    for h in range(24):
        # Select the value from demand_distribution where "Node" equals n
        demand_value = (demand_distribution.loc[demand_distribution["Node"] == n, demand_distribution.columns[2]].values[0])/100

        # Access the demand_profile value for hour h
        profile_value = demand_profile.iloc[h, 1]

        # Compute and assign to Demand
        Demand[n, h] = demand_value * profile_value
        
Strategy_rival=Df_rival["S3"] #we fix one strategy for the rival, the one of scenario 3



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
        self.nb_scenarios=DA_prices.shape[0]


class Optimal_Investment:
    def __init__(self, input_data: InputData, complementarity_method: str = 'SOS1'): #complementarty not used
        self.data = input_data  # Reference to the InputData instance
        #self.complementarity_method = complementarity_method  # Complementarity method
        self.variables = Expando()  # Container for decision variables
        self.constraints = Expando()  # Container for constraints
        self.results = Expando()  # Container for results
        self.model = gp.Model("OptimalInvestment")  # Gurobi model
        self._build_variables()  # Define variables
        self._build_upper_level_constraint()
        self._build_objective_function()
        
        self.run()
        self._save_results()
        #self._build # Define constraints

    def _build_variables(self):
        # Define production variables for new conventional units  Pc_nwt
        self.variables.prod_new_conv_unit = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Prod_New_Conv_Unit_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in range(0,self.data.nb_scenarios)
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

        # Define production variables for existing rival conventional units PR
        self.variables.prod_existing_rival = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Prod_Existing_Rival_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in range(0,self.data.nb_scenarios)
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Define production variables for new rival conventional units PR_nwt
        self.variables.prod_new_conv_rival = {
            (w, h, n): self.model.addVar(
                lb=0,
                ub=GRB.INFINITY,
                name=f"Prod_New_Conv_Rival_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in range(0,self.data.nb_scenarios)
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # # Define electricity consumption variables for demand - removed bc now it will be fixed and exogenous
        # self.variables.demand_consumed = {
        #     (w, h, n): self.model.addVar(
        #         lb=0,
        #         ub=GRB.INFINITY,
        #         name=f"Demand_Consumed_Scenario{w}_Hour{h}_Node{n}"
        #     )
        #     for w in self.data.demand_scenarios.columns
        #     for h in range(1, 25)
        #     for n in range(1, 25)
        # }

        # Define voltage angle variables
        self.variables.voltage_angle = {
            (w, h, n): self.model.addVar(
                lb=-m.pi,
                ub=m.pi,
                name=f"Voltage_Angle_Scenario{w}_Hour{h}_Node{n}"
            )
            for w in range(0,self.data.nb_scenarios)
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Define investment decision variables#xnc
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
            gp.quicksum(self.variables.node_bin[n] for n in range(1, 25)) <= 14,
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
        


        
        # Get nodes present in the data
        nodes_in_data = set(self.data.investor_generation_data["Node"].unique())
        # Production limits for existing conventional units
        self.constraints.production_limits_existing_con = {
            (w, h, n): self.model.addConstr(
                self.variables.prod_existing_conv[w, h, n] <= (
                    self.data.investor_generation_data.loc[
                        self.data.investor_generation_data["Node"] == n, "Pmax"
                    ].values[0] if n in nodes_in_data else 0  # Default Pmax to 0 for unconstrained nodes
                ),
                name=f"Prod limit for existing conv. at node {n}, scenario {w}, hour {h}"
            )
            for w in range(0, self.data.nb_scenarios)  # Iterate over scenarios
            for h in range(1, 25)  # Iterate over hours
            for n in range(1, 25)  # Iterate over all nodes in variables
        }

        #check here, rival is producing crazy amountss....###############################################################################
        # Production limits for existing rival units
        # Get nodes present in the data
        nodes_in_data = set(self.data.rival_generation_data["Node"].unique())
        self.constraints.production_limits_existing_rival = {
            (w, h, n): self.model.addConstr(
                self.variables.prod_existing_rival[w, h, n] <= (
                    self.data.rival_generation_data.loc[
                        self.data.rival_generation_data["Node"] == n, "Pmax"
                    ].values[0] if n in nodes_in_data else 0  # Default Pmax to 0 for unconstrained nodes
                ),
                name=f"Prod limit for existing rival at node {n}, scenario {w}, hour {h}"
            )
            for w in range(0, self.data.nb_scenarios)  # Iterate over scenarios
            for h in range(1, 25)  # Iterate over hours
            for n in range(1, 25)  # Iterate over all nodes in variables
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
        
        #Power balance constraints
        self.constraints.power_balance = {
            (w, h, n): self.model.addConstr(
                Demand[h-1, n-1] +
                gp.quicksum(
                    self.data.matrix_B.iloc[n - 1, m - 1] * 
                    (self.variables.voltage_angle[w, h, n] - self.variables.voltage_angle[w, h, m])
                    for m in range(1, 25) if m != n
                ) -
                self.variables.prod_new_conv_unit[w, h, n] -
                self.variables.prod_PV[w, h, n] -
                self.variables.prod_wind[w, h, n] -
                self.variables.prod_existing_conv[w, h, n] -
                self.variables.prod_existing_rival[w, h, n] -
                self.variables.prod_new_conv_rival[w, h, n] == 0,
                name=f"Power balance at node {n}, scenario {w}, hour {h}"
            )
            for w in range(0,self.data.nb_scenarios)  # Scenarios
            for h in range(1, 25)  # Hours
            for n in range(1, 25)  # Nodes
        }
        
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

        
                # Production limits for new rival conventional units, restricted to node 23
        self.constraints.production_limits_new_rival = {
            (w, h, n): self.model.addConstr(
                self.variables.prod_new_conv_rival[w, h, n] <= Strategy_rival.iloc[0],
                name=f"Prod limit for new rival unit at node {n}, scenario {w}, hour {h}"
            )
            for w in range(0,self.data.nb_scenarios)  # Iterate over scenario columns
            for h in range(1, 25)  # Iterate over 24 hours
            for n in [23]  # Restrict to node 23 only
        }
        
        # Restrict new rival production to only node 23
        self.constraints.node_limits_new_rival = {
            (w, h, n): self.model.addConstr(
                self.variables.prod_new_conv_rival[w, h, n] == 0,
                name=f"Limit new rival production to node 23, scenario {w}, hour {h}, node {n}"
            )
            for w in range(0,self.data.nb_scenarios)  # Iterate over scenario columns
            for h in range(1, 25)  # Iterate over 24 hours
            for n in range(1, 25) if n != 23  # Apply to all nodes except node 23
        }
        
 

    # Constraint for line power flows based on voltage angle differences and line reactance
        self.constraints.line_power_flow_upper = {
            (w, h, n, m): self.model.addConstr(
                self.data.matrix_B.loc[n, m] * (
                    self.variables.voltage_angle[w, h, n] - self.variables.voltage_angle[w, h, m]
                ) <= self.data.capacity_matrix.loc[n, m],
                name=f"Power flow on line {n}-{m}, scenario {w}, hour {h}"
            )
            for w in range(0,self.data.nb_scenarios)  # Iterate over scenario columns
            for h in range(1, 25)  # Iterate over 24 hours
            for n in self.data.lines_data["From"].unique()  # Iterate over 'From' nodes in lines
            for m in self.data.lines_data.loc[self.data.lines_data["From"] == n, "To"]  # Iterate over 'To' nodes connected to n
        }
        
        self.constraints.line_power_flow_lower = {
            (w, h, n, m): self.model.addConstr(
                self.data.matrix_B.loc[n, m] * (
                    self.variables.voltage_angle[w, h, n] - self.variables.voltage_angle[w, h, m])
                 >= -self.data.capacity_matrix.loc[n, m],
                name=f"Power flow on line {n}-{m}, scenario {w}, hour {h}"
            )
            for w in range(0,self.data.nb_scenarios)  # Iterate over scenario columns
            for h in range(1, 25)  # Iterate over 24 hours
            for n in self.data.lines_data["From"].unique()  # Iterate over 'From' nodes in lines
            for m in self.data.lines_data.loc[self.data.lines_data["From"] == n, "To"]  # Iterate over 'To' nodes connected to n
        }
        



        
                #Constraint to set voltage angle to 0 for the reference node (Node 1)
        self.constraints.voltage_angle_fixed_node1 = {
            (w, h): self.model.addConstr(
                self.variables.voltage_angle[w, h, n] == 0,
                name=f"Voltage angle fixed to 0 at Node 1, scenario {w}, hour {h}"
            )
            for w in range(0,self.data.nb_scenarios)  # Iterate over all scenario columns
            for h in range(1, 25)  # Iterate over 24 hours
            for n in [1]
        }






    def _build_objective_function(self):
        # Assuming 'probability_scenario' is a list of probabilities for each scenario
        probability_scenario=1/20 #probability of each lambda scenario
    
        investment_cost = gp.quicksum(
            self.data.investment_data.loc[self.data.investment_data["Technology"] == "Conventional", "Inv_Cost"].values[0] * self.variables.cap_invest_conv[n] +
            self.data.investment_data.loc[self.data.investment_data["Technology"] == "PV", "Inv_Cost"].values[0] * self.variables.cap_invest_PV[n] +
            self.data.investment_data.loc[self.data.investment_data["Technology"] == "Wind", "Inv_Cost"].values[0] * self.variables.cap_invest_wind[n]
            for n in range(1, 25)
        )

        production_revenue = (
            20 * 365 * gp.quicksum(
                gp.quicksum(
                    gp.quicksum(
                        probability_scenario *
                        DA_prices[w, n - 1, h - 1] * (  # Adjust h for 0-indexed DA_prices
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
                            self.variables.prod_new_conv_unit[(w, h, n)] * cand_Conv_cost  # Investment cost
                            + (
                                self.variables.prod_existing_conv[(w, h, n)] *
                                self.data.investor_generation_data.loc[
                                    self.data.investor_generation_data["Node"] == n, "Bid price"
                                ].values[0]
                                if n in self.data.investor_generation_data["Node"].values
                                else 0  # Skip cost if there is no existing generator at node n
                            )
                        )
                        for w in range(0, self.data.nb_scenarios)  # Iterate over all scenarios
                    )
                    for h in range(1, 25)  # Iterate over 24 hours
                )
                for n in range(1, 25)  # Iterate over all nodes
            )
        )


         # production_revenue = 20*365*gp.quicksum(quicksum(quicksum(
         #    probability_scenario *  
         #    DA_prices[w,n-1,h - 1] * (  # Adjust h for 0-indexed DA_prices
         #        self.variables.prod_new_conv_unit[(w, h, n)] +
         #        (
         #            self.variables.prod_existing_conv[(w, h, n)]
         #            if n in self.data.investor_generation_data["Node"].values
         #            else 0  # Skip if there is no existing generator at node n
         #        ) +
         #        self.variables.prod_PV[(w, h, n)] +
         #        self.variables.prod_wind[(w, h, n)]
         #    )
         #    - (
         #        self.variables.prod_new_conv_unit[(w, h, n)] * cand_Conv_cost  # Subtracting investment cost for new conventional units
         #        + (
         #            self.variables.prod_existing_conv[(w, h, n)] * self.data.investor_generation_data.loc[
         #                self.data.investor_generation_data["Node"] == n, "Bid price"
         #            ].values[0]
         #            if n in self.data.investor_generation_data["Node"].values
         #            else 0  # Skip cost if there is no existing generator at node n
         #        )
         #    )
         #    for w in range(0,self.data.nb_scenarios))  # Iterate over all scenario columns
         #    for h in range(1, 25))  # Iterate over 24 hours
         #    for n in range(1,25)) # Iterate over all keys
    
        # Set the objective as the minimization of total cost
        obj=investment_cost - production_revenue
        self.model.setObjective(obj, GRB.MINIMIZE)
        
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
                print(f"  - Investment Active in Node")
                
                
        #Rival
        # print("\nIRival Decisions:")
        # for n in range(1, 25):  # Assuming 24 nodes
        #     print(f"Node {n}:")
        #     if self.variables.prod_new_conv_rival[1,12,n].x > 0:
        #         print(f"  - Conventional Capacity new rival: {self.variables.prod_new_conv_rival[1,12,n].x:.2f} MW")
        #     if self.variables.prod_existing_rival[1,12,n].x > 0:
        #         print(f"  - Conventional Capacity existing rival: {self.variables.prod_existing_rival[1,12,n].x:.2f} MW")
        
        rival_existing_decision = np.zeros((20,24, 24))
        inv_existing_decision = np.zeros((20,24, 24))
        flow_result = np.zeros((20,24, 24))
        inv_new_prod_PV = np.zeros((20,24, 24))
        inv_new_prod_wind = np.zeros((20,24, 24))
        inv_new_prod_conv = np.zeros((20,24, 24))
        

            
        for w in range(0,self.data.nb_scenarios):
            for h in range (1,24):
                for n in range (1,24):
                    rival_existing_decision[w,h,n]=self.variables.prod_existing_rival[(w,h,n)].x
                    inv_existing_decision[w,h,n]=self.variables.prod_existing_conv[(w,h,n)].x
                    flow_result[w,h,n]=sum(self.data.matrix_B.iloc[n - 1, m - 1] * 
                        (self.variables.voltage_angle[(w,h,n)].x - self.variables.voltage_angle[(w,h,m)].x)
                        for m in range(1, 25) if m != n)
                    inv_new_prod_PV[w,h,n]=self.variables.prod_PV[(w,h,n)].x
                    inv_new_prod_wind[w,h,n]=self.variables.prod_wind[(w,h,n)].x
                    inv_new_prod_conv[w,h,n]=self.variables.prod_new_conv_unit[(w,h,n)].x
                    
         
        print("xxxxxxxxxxx",self.data.investment_data.loc[self.data.investment_data["Technology"] == "Conventional", "Inv_Cost"].values[0])

                    
        return rival_existing_decision, inv_existing_decision,flow_result,  inv_new_prod_conv,  inv_new_prod_wind,  inv_new_prod_PV
        
        
        
        
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
    rival_existing_decision, inv_existing_decision, flow_result,  inv_new_prod_conv,  inv_new_prod_wind,  inv_new_prod_PV=model.display_results()


    


