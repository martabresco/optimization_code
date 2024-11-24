import pandas as pd
from itertools import product
import random
import matplotlib.pyplot as plt

#Scenario Creation

def scenarios_creation():
    # Import Excel files
    demand_scenario = pd.read_excel('Demand_scenario_prep.xlsx', index_col=0)
    rival_scenario = pd.read_excel('Rival_scenario_prep.xlsx', index_col=0)
    
    
    # Transpose DataFrames
    demand_scenario = demand_scenario.transpose()
    rival_scenario = rival_scenario.transpose()
    
    # Get indexes of each DataFrame
    demand_indexes = demand_scenario.index
    rival_indexes = rival_scenario.index
    
    # Initialize lists to store scenarios
    Demand_scenarios = []
    Rival_scenarios = []

    # Create the Cartesian product of all scenarios
    for demand_index, rival_index in product(demand_indexes, rival_indexes):
        Demand_scenarios.append(demand_scenario.loc[demand_index].values.tolist())
        Rival_scenarios.append(rival_scenario.loc[rival_index].values.tolist())
    
    return Demand_scenarios, Rival_scenarios

#Save this file in your working directory

# Load Generation Data from investor
file_path1 = "Generation Data.xlsx"  # you need to have this file in your working directory
investor_generation_data = pd.read_excel(file_path1, sheet_name = "Generation_investor")
pd.set_option('display.max_rows', None)
print(investor_generation_data)

#load generation data from rival
file_path1 = "Generation Data.xlsx"  # you need to have this file in your working directory
rival_generation_data = pd.read_excel(file_path1, sheet_name = "Generation_rival")
pd.set_option('display.max_rows', None)
print(rival_generation_data)

# Lines Generation Data
file_path2 = "Lines_Data.xlsx"  # you need to have this file in your working directory
lines_data = pd.read_excel(file_path2) 
print(lines_data)

# Power Capacity Factor for Wind Data
file_path3 = "Wind_PowerFactor_AverageDay.xlsx"  # you need to have this file in your working directory
Wind_PF_data = pd.read_excel(file_path3) # normalized wrt the max output during the year
print(Wind_PF_data)

# Power Capacity Factor for PV Data
file_path4 = "PV_PowerFactor_AverageDay.xlsx"  # you need to have this file in your working directory
PV_PF_data = pd.read_excel(file_path4) # normalized wrt the max output during the year
print(PV_PF_data)

#Load profile data (base load for each representative hour)
file_path5 = "Load dataset.xlsx"  # you need to have this file in your working directory
Demand_profile = pd.read_excel(file_path5, sheet_name = "Load profile") #In MW for each hour
print(Demand_profile)

#Load distribution data (base load for each representative hour)
Demand_distribution = pd.read_excel(file_path5, sheet_name = "Load distribution") #in % of system load
print(Demand_distribution)

#Demand prices (Utility of demand) for each representative hour. It is not dependent on the node 
Demand_prices = pd.read_excel(file_path5, sheet_name = "Demand prices") # in $/MWh
print(Demand_prices)

#Investment costs and capacities
file_path6="Investment.xlsx"
Investment_data=pd.read_excel(file_path6)
print(Investment_data)

Demand_scenarios, Rival_scenarios=scenarios_creation()

print("Demand", Demand_scenarios)
print("Rival", Rival_scenarios)




