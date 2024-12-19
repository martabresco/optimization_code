# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:33:54 2024

@author: User
"""

def _build_upper_level_constraint(self):
    K = 1.6e9  # Budget constraint

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
        ) <= K,
        name="Investment budget limit"
    )

    # Power balance constraints
    self.constraints.power_balance = {
        (w, h, n): self.model.addConstr(
            self.variables.demand_consumed[w, h, n] +
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
        for w in self.data.rival_scenarios.columns  # Scenarios
        for h in range(1, 25)  # Hours
        for n in range(1, 25)  # Nodes
    }

    # Voltage angle limits
    self.constraints.voltage_angle_limits = {
        (w, h, n): self.model.addConstr(
            -m.pi <= self.variables.voltage_angle[w, h, n] <= m.pi,
            name=f"Voltage angle bounds at node {n}, scenario {w}, hour {h}"
        )
        for w in self.data.rival_scenarios.columns
        for h in range(1, 25)
        for n in range(1, 25)
    }

    # Voltage angle reference at node 1
    self.constraints.voltage_angle_fixed_node1 = {
        (w, h): self.model.addConstr(
            self.variables.voltage_angle[w, h, 1] == 0,
            name=f"Voltage angle fixed at node 1, scenario {w}, hour {h}"
        )
        for w in self.data.rival_scenarios.columns
        for h in range(1, 25)
    }

    # Production limits for PV units
    self.constraints.production_limits_PV = {
        (w, h, n): self.model.addConstr(
            0 <= self.variables.prod_PV[w, h, n] <= self.data.PV_PF_data.iloc[h - 1, 1] * self.variables.cap_invest_PV[n],
            name=f"Prod limit for PV at node {n}, scenario {w}, hour {h}"
        )
        for w in self.data.rival_scenarios.columns
        for h in range(1, 25)
        for n in range(1, 25)
    }

    def _build_power_balance_constraints(self):
        # Power balance constraints
        self.constraints.power_balance = {
            (w, h, n): self.model.addConstr(
                self.variables.demand_consumed[w, h, n] +
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
            for w in self.data.rival_scenarios.columns  # Scenarios (e.g., S1, S2, ...)
            for h in range(1, 25)  # Hours
            for n in range(1, 25)  # Nodes
        }

    def _build_production_limits(self):
        # Production limits for new conventional units
        self.constraints.production_limits_con = {
            (w, h, n): self.model.addConstr(
                0 <= self.variables.prod_new_conv_unit[w, h, n] <= self.variables.cap_invest_conv[n],
                name=f"Prod limit for new conv. unit at node {n}, scenario {w}, hour {h}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Production limits for PV units
        self.constraints.production_limits_PV = {
            (w, h, n): self.model.addConstr(
                0 <= self.variables.prod_PV[w, h, n] <= 
                self.data.PV_PF_data.iloc[h - 1, 1] * self.variables.cap_invest_PV[n],
                name=f"Prod limit for PV at node {n}, scenario {w}, hour {h}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Production limits for wind units
        self.constraints.production_limits_wind = {
            (w, h, n): self.model.addConstr(
                0 <= self.variables.prod_wind[w, h, n] <= 
                self.data.wind_PF_data.iloc[h - 1, 1] * self.variables.cap_invest_wind[n],
                name=f"Prod limit for wind at node {n}, scenario {w}, hour {h}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Production limits for existing conventional units
        self.constraints.production_limits_existing_con = {
            (w, h, n): self.model.addConstr(
                0 <= self.variables.prod_existing_conv[w, h, n] <= 
                self.data.investor_generation_data.loc[
                    self.data.investor_generation_data["Node"] == n, "Pmax"
                ].max(),
                name=f"Prod limit for existing conv. at node {n}, scenario {w}, hour {h}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

        # Production limits for existing rival conventional units
        self.constraints.production_limits_existing_rival = {
            (w, h, n): self.model.addConstr(
                0 <= self.variables.prod_existing_rival[w, h, n] <= 
                self.data.rival_generation_data.loc[
                    self.data.rival_generation_data["Node"] == n, "Pmax"
                ].max(),
                name=f"Prod limit for existing rival at node {n}, scenario {w}, hour {h}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

    def _build_line_constraints(self):
        # Power flow constraints
        self.constraints.line_power_flow = {
            (w, h, n, m): self.model.addConstr(
                self.data.matrix_B.iloc[n - 1, m - 1] * 
                (self.variables.voltage_angle[w, h, n] - self.variables.voltage_angle[w, h, m]) <= 
                self.data.capacity_matrix.iloc[n - 1, m - 1],
                name=f"Power flow on line {n}-{m}, scenario {w}, hour {h}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
            for m in range(1, 25) if m != n
        }

    def _build_voltage_constraints(self):
        # Voltage angle limits
        self.constraints.voltage_angle_limits = {
            (w, h, n): self.model.addConstr(
                self.variables.voltage_angle[w, h, n] >= -m.pi,
                name=f"Voltage angle lower limit at node {n}, scenario {w}, hour {h}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        }

        self.constraints.voltage_angle_limits.update({
            (w, h, n): self.model.addConstr(
                self.variables.voltage_angle[w, h, n] <= m.pi,
                name=f"Voltage angle upper limit at node {n}, scenario {w}, hour {h}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
            for n in range(1, 25)
        })

        # Voltage angle reference at node 1
        self.constraints.voltage_angle_fixed_node1 = {
            (w, h): self.model.addConstr(
                self.variables.voltage_angle[w, h, 1] == 0,
                name=f"Voltage angle fixed at node 1, scenario {w}, hour {h}"
            )
            for w in self.data.rival_scenarios.columns
            for h in range(1, 25)
        }
