import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from Data import investor_generation_data
from Data import rival_generation_data
from Data import lines_data
from Data import Wind_PF_data
from Data import PV_PF_data
from Data import Demand_profile
from Data import Demand_distribution
from Data import Demand_prices
from Data import Investment_data



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

print(input_data.existing_investor_generator_id)
print(input_data.existing_investor_generator_node)
print(input_data.existing_investor_generator_cost)
print(input_data.existing_investor_generator_capacity)
print(input_data.line_capacity)
print(input_data.PV_PF)
print(input_data.system_demand)
print(input_data.max_investment_capacity)


