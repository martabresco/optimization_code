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
        
        
        
        
        self.line_from=line_from
        self.line_to=line_to
        self.line_X=line_X
        self.line_capacity=line_capacity
        self.hour=hour
        self.PV_PF=PV_PF
        



if __name__ == "__main__":
    input_data = InputData(
        existing_investor_generator_id=investor_generation_data.iloc[:,0].tolist(),
        existing_investor_generator_node=dict(zip(investor_generation_data.iloc[:,0].tolist(),investor_generation_data.iloc[:,1].tolist())),
        existing_investor_generator_cost=dict(zip(investor_generation_data.iloc[:,0].tolist(),investor_generation_data.iloc[:,3].tolist())),
        existing_investor_generator_capacity=dict(zip(investor_generation_data.iloc[:,0].tolist(),investor_generation_data.iloc[:,2].tolist())),
        
        existing_rival_generator_id=investor_generation_data.iloc[:,0].tolist(),
        existing_rival_generator_node=dict(zip(investor_generation_data.iloc[:,0].tolist(),investor_generation_data.iloc[:,1].tolist())),
        existing_investor_generator_cost=dict(zip(investor_generation_data.iloc[:,0].tolist(),investor_generation_data.iloc[:,3].tolist())),
        existing_investor_generator_capacity=dict(zip(investor_generation_data.iloc[:,0].tolist(),investor_generation_data.iloc[:,2].tolist())),
        
        line_id = lines_data["Line id"].tolist(),
        line_from=dict(zip(lines_data["Line id"].tolist(),lines_data["From"].tolist())),
        line_to=dict(zip(lines_data["Line id"].tolist(),lines_data["To"].tolist())),
        line_X=dict(zip(lines_data["Line id"].tolist(),lines_data["X (pu)"].tolist())),
        line_capacity=dict(zip(lines_data["Line id"].tolist(),lines_data["Capacity (MVA)"].tolist())),
        hour=Wind_PF_data["Hour"].tolist(),
        wind_PF=dict(zip(Wind_PF_data["Hour"].tolist(),Wind_PF_data["Onshore Wind"].tolist())),
        PV_PF=dict(zip(Wind_PF_data["Hour"].tolist(),PV_PF_data["PV"].tolist()))
        
        
        
        
        
    )

print(input_data.existing_generator_id)
print(input_data.existing_generator_node)
print(input_data.existing_generator_cost)
print(input_data.existing_generator_capacity)
print(input_data.line_capacity)
print(input_data.PV_PF)


