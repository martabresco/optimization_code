import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from Data import generation_data
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
        EXISTING_GENERATOR_ID: list, #list with the number of each unit
        existing_generator_node:dict[str,int],
        existing_generator_capacity: dict[str, int],
        existing_generator_cost: dict[str, int],   # Idea: create one attribute   

    ):
        # List of existing generators 
        self.EXISTING_GENERATOR_ID = EXISTING_GENERATOR_ID
        #Dictionary with connection node of each existing generator
        self.existing_generator_node = existing_generator_node
        #Dictionary with capacity of each existing generator
        self.existing_generator_capacity = existing_generator_capacity
        # Dictionary with each cost of existing geenrator
        self.existing_generator_cost = existing_generator_cost 



if __name__ == "__main__":
    input_data = InputData(
        EXISTING_GENERATOR_ID=generation_data.iloc[:,0].tolist(),
        existing_generator_node=dict(zip(generation_data.iloc[:,0].tolist(),generation_data.iloc[:,1].tolist())),
        existing_generator_cost=dict(zip(generation_data.iloc[:,0].tolist(),generation_data.iloc[:,3].tolist())),
        existing_generator_capacity=dict(zip(generation_data.iloc[:,0].tolist(),generation_data.iloc[:,2].tolist()))
    )

print(input_data.EXISTING_GENERATOR_ID)
print(input_data.existing_generator_node)
print(input_data.existing_generator_cost)
print(input_data.existing_generator_capacity)


