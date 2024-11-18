import gurobipy as gp
from gurobipy import GRB


class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class InputData:
    
    def __init__(
        self, 
        GENERATORS: list, 
        LOADS: list, 
        generator_cost: dict[str, int],     
        generator_capacity: dict[str, int],
        load_utility: dict[str, float],
        load_capacity: dict[str, int]
    ):
        # List of generators 
        self.GENERATORS = GENERATORS
        # List of loads
        self.LOADS = LOADS
        # Generators costs (c^G_i)
        self.generator_cost = generator_cost 
        # Generators capacity (P^G_i)
        self.generator_capacity = generator_capacity
        # Load utility cost (c^D_i)
        self.load_utility = load_utility
        # Loads capacity (P^D_i)
        self.load_capacity = load_capacity 


class EconomicDispatch():

    def __init__(self, input_data: InputData, complementarity_method: str = 'Big M'): # initialize class
        self.data = input_data # define data attributes
        self.complementarity_method = complementarity_method # define method for complementarity conditions
        self.variables = Expando() # define variable attributes
        self.constraints = Expando() # define constraints attributes
        self.results = Expando() # define results attributes
        self._build_model() # build gurobi model
    
    def _build_variables(self):
        # lower-level primal variables
        self.variables.generator_production = {
            g: self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Electricity production of generator {0}'.format(g)
            ) for g in self.data.GENERATORS
        }
        self.variables.load_consumption = {
            d: self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Electricity consumption of load {0}'.format(d)
            ) for d in self.data.LOADS
        }

        ## lower-level dual variables (1 for each constraint of the ED)
        self.variables.balance_dual = self.model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Dual variable of balance equation'
        ) 
        self.variables.min_production_dual = {
            g: self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Dual variable of min. production constraint of generator {0}'.format(g)
            ) for g in self.data.GENERATORS
        }
        self.variables.max_production_dual = {
            g: self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Dual variable of max. production constraint of generator {0}'.format(g)
            ) for g in self.data.GENERATORS
        } 
        self.variables.min_consumption_dual = {
            d: self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Dual variable of min. consumption constraint of load {0}'.format(d)
            ) for d in self.data.LOADS
        }
        self.variables.max_consumption_dual = {
            d: self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Dual variable of max. consumption constraint of load {0}'.format(d)
            ) for d in self.data.LOADS
        }

        ## upper-level variable 
        self.variables.g1_production_DA = self.model.addVar(
            lb=0, ub=GRB.INFINITY, name='Day-ahead electricity production of generator G1'
        )
    
    def _build_upper_level_constraint(self):
        self.constraints.upper_level_max_production_constraint = self.model.addLConstr(
            self.variables.g1_production_DA,
            GRB.LESS_EQUAL,
            self.data.generator_capacity['G1'],
            name='Upper-level max production constraint for generator G1',
        )

    def _build_kkt_primal_constraints(self):
        # build balance constraint
        self.constraints.balance_constraint = (
            self.model.addLConstr(
                gp.quicksum(self.variables.generator_production[g] for g in self.data.GENERATORS),
                GRB.EQUAL,
                gp.quicksum(self.variables.load_consumption[d] for d in self.data.LOADS),
                name='Balance constraint',
            )
        )
        # build max production constraints 
        self.constraints.max_production_constraints = {
            g: self.model.addLConstr(
                self.variables.generator_production[g], 
                GRB.LESS_EQUAL,
                self.data.generator_capacity[g],
            ) for g in self.data.GENERATORS if g != 'G1'
        }
        self.constraints.max_production_constraints['G1'] = self.model.addLConstr(
            self.variables.generator_production['G1'], 
            GRB.LESS_EQUAL,
            self.variables.g1_production_DA,
        )
        # build max consumption constraints
        self.constraints.max_consumption_constraints = {
            d: self.model.addLConstr(
                self.variables.load_consumption[d],
                GRB.LESS_EQUAL,
                self.data.load_capacity[d],
                name='Max consumption constraint of load {0}'.format(d)
            ) for d in self.data.LOADS
        }
    
    def _build_kkt_first_order_constraints(self):
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


if __name__ == '__main__':
    input_data = InputData(
        GENERATORS = ['W1', 'G1', 'G2'],
        LOADS = ['D1', 'D2'],
        generator_cost = {'W1':0,'G1':30,'G2':35},
        generator_capacity = {'W1': 80, 'G1': 80, 'G2': 80},
        load_utility = {'D1':40,'D2':20}, 
        load_capacity = {'D1':100,'D2':50},
    )
    model = EconomicDispatch(input_data, complementarity_method='Big M')
    # model = EconomicDispatch(input_data, complementarity_method='SOS1')
    model.run()
    model.display_results()
