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
        SCENARIOS: list,
        generator_cost: float,
        generator_capacity: float,
        price_DA: float,
        generation_availability: dict[str, float],
        price_B: dict[str, float],
        pi: dict[str, float],
        price_scheme: str,
    ):
        # List of scenarios
        self.SCENARIOS = SCENARIOS
        # Generators costs (c^G_i)
        self.generator_cost = generator_cost 
        # Generators capacity (P^G_i)
        self.generator_capacity = generator_capacity
        # Market clearing price (lambda^DA)
        self.price_DA = price_DA
        # Available wind power in scenarios
        self.generation_availability = generation_availability
        # Scenario probability
        self.pi = pi
        # Balancing price depending on price scheme
        if price_scheme == "one-price":
            self.price_B_up = price_B
            self.price_B_dw = price_B
        elif price_scheme == "two-price":
            self.price_B_up = dict(zip(SCENARIOS, [min(price_DA, price_B[s]) for s in SCENARIOS]))
            self.price_B_dw = dict(zip(SCENARIOS, [max(price_DA, price_B[s]) for s in SCENARIOS]))
        else:
            raise NotImplementedError(f"'{price_scheme}' is not a valid price scheme. Use 'one-price' or 'two-price'.")


class StochasticOfferingStrategy():

    def __init__(self, input_data: InputData, perfect_information: bool = False):
        self.data = input_data 
        self.perfect_information = perfect_information 
        self.variables = Expando()
        self.constraints = Expando() 
        self.results = Expando() 
        self._build_model() 

    def _build_variables(self):
        if self.perfect_information:
            self.variables.generator_production = {
                k: self.model.addVar(
                    lb=0, ub=GRB.INFINITY, name='Electricity production'
                ) for k in self.data.SCENARIOS
            }
        elif not(self.perfect_information):
            self.variables.generator_production = self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Electricity production'
            )
        self.variables.up_regulation = {
            k: self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Up-regulation in BM'
            ) for k in self.data.SCENARIOS
        }
        self.variables.down_regulation = {
            k: self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Down-regulation in BM'
            ) for k in self.data.SCENARIOS
        }
    
    def _build_constraints(self):
        # With perfect information we have a DA production variable for each scenario
        if self.perfect_information:
            self.constraints.min_production_constraints = {
                k: self.model.addLConstr(
                    0,
                    GRB.LESS_EQUAL,
                    self.variables.generator_production[k] + self.variables.up_regulation[k] - self.variables.down_regulation[k],
                    name='Min production constraint',
                ) for k in self.data.SCENARIOS
            }
            self.constraints.max_production_constraints = {
                k: self.model.addLConstr(
                    self.variables.generator_production[k] + self.variables.up_regulation[k] - self.variables.down_regulation[k],
                    GRB.LESS_EQUAL,
                    self.data.generation_availability[k],
                    name='Max production constraint',
                ) for k in self.data.SCENARIOS
            }
            self.constraints.max_DA_production_constraints = {
                k: self.model.addLConstr(
                    self.variables.generator_production[k],
                    GRB.LESS_EQUAL,
                    self.data.generator_capacity,
                    name='Max DA production constraint',
                ) for k in self.data.SCENARIOS
            }
        # Without perfect information we only have one DA production variable
        else:
            self.constraints.min_production_constraints = {
                k: self.model.addLConstr(
                    0,
                    GRB.LESS_EQUAL,
                    self.variables.generator_production + self.variables.up_regulation[k] - self.variables.down_regulation[k],
                    name='Min production constraint',
                ) for k in self.data.SCENARIOS
            }
            self.constraints.max_production_constraints = {
                k: self.model.addLConstr(
                    self.variables.generator_production + self.variables.up_regulation[k] - self.variables.down_regulation[k],
                    GRB.LESS_EQUAL,
                    self.data.generation_availability[k],
                    name='Max production constraint',
                ) for k in self.data.SCENARIOS
            }
            self.constraints.max_DA_production_constraints = self.model.addLConstr(
                self.variables.generator_production,
                GRB.LESS_EQUAL,
                self.data.generator_capacity,
                name='Max DA production constraint',
            )

    def _build_objective_function(self):
        # DA profits
        if self.perfect_information:
            DA_profit = gp.quicksum(
                self.variables.generator_production[k] * self.data.pi[k] * (self.data.price_DA - self.data.generator_cost)
                for k in self.data.SCENARIOS
            )
        else:
            DA_profit = self.variables.generator_production * (self.data.price_DA - self.data.generator_cost)
        # Balancing profits
        B_profit = gp.quicksum(
            self.data.pi[k] * (
                (self.data.price_B_up[k] - self.data.generator_cost) * self.variables.up_regulation[k]
                - (self.data.price_B_dw[k] - self.data.generator_cost) * self.variables.down_regulation[k]
            ) for k in self.data.SCENARIOS
        )
        self.model.setObjective(DA_profit + B_profit, GRB.MAXIMIZE)

    def _build_model(self):
        self.model = gp.Model(name='Two-stage stochastic offering strategy')
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        if self.perfect_information:
            self.results.generator_production_DA = [
                self.variables.generator_production[k].x for k in self.data.SCENARIOS
            ]
        else:
            self.results.generator_production_DA = self.variables.generator_production.x
        self.results.up_regulation = {
            k: self.variables.up_regulation[k].x for k in self.data.SCENARIOS
        }
        self.results.down_regulation = {
            k: self.variables.down_regulation[k].x for k in self.data.SCENARIOS
        }

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"optimization of {model.ModelName} was not successful")
    
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Expected profit:")
        print(self.results.objective_value)
        print("Optimal DA offer:")
        print(self.results.generator_production_DA)
        print("Optimal up-regulation:")
        print(self.results.up_regulation)
        print("Optimal down-regulation:")
        print(self.results.down_regulation)
        print("--------------------------------------------------")


def calculate_evpi(model: StochasticOfferingStrategy, model_PI: StochasticOfferingStrategy):
    print()
    print("Expected value of perfect information (EVPI):")
    print(model_PI.results.objective_value - model.results.objective_value)


if __name__ == '__main__':
    # one-price balancing scheme 
    one_price_input_data = InputData(
        SCENARIOS = ['S1', 'S2', 'S3', 'S4'],
        generator_cost = 15,
        generator_capacity = 150,
        price_DA = 20,
        generation_availability = {'S1': 125, 'S2': 75, 'S3': 125, 'S4': 75},
        price_B = {'S1': 15, 'S2': 15, 'S3': 35, 'S4': 35},
        pi = {'S1': 0.25, 'S2': 0.25, 'S3': 0.25, 'S4': 0.25},
        price_scheme = 'one-price',
    )
    one_price_model = StochasticOfferingStrategy(one_price_input_data, perfect_information=False)
    one_price_model.run()
    one_price_model.display_results()
    one_price_model_PI = StochasticOfferingStrategy(one_price_input_data, perfect_information=True)
    one_price_model_PI.run()
    one_price_model_PI.display_results()
    calculate_evpi(one_price_model, one_price_model_PI)

    # two-price balancing scheme 
    two_price_input_data = InputData(
        SCENARIOS = ['S1', 'S2', 'S3', 'S4'],
        generator_cost = 15,
        generator_capacity = 150,
        price_DA = 20,
        generation_availability = {'S1': 125, 'S2': 75, 'S3': 125, 'S4': 75},
        price_B = {'S1': 15, 'S2': 15, 'S3': 35, 'S4': 35},
        pi = {'S1': 0.25, 'S2': 0.25, 'S3': 0.25, 'S4': 0.25},
        price_scheme = 'two-price',
    )
    two_price_model = StochasticOfferingStrategy(two_price_input_data, perfect_information=False)
    two_price_model.run()
    two_price_model.display_results()
    two_price_model_PI = StochasticOfferingStrategy(two_price_input_data, perfect_information=True)
    two_price_model_PI.run()
    two_price_model_PI.display_results()
    calculate_evpi(two_price_model, two_price_model_PI)


