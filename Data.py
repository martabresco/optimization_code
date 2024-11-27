import pandas as pd
from itertools import product
import random
import matplotlib.pyplot as plt
import numpy as np

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
    
    arr_dem=np.transpose(np.array(Demand_scenarios))
    arr_riv=np.transpose(np.array(Rival_scenarios))

    # print("Demand", arr_dem)
    # print("Rival", arr_riv)
        
    Df_demand=pd.DataFrame(arr_dem,index=['1','2','3','4','5','6',
                                                     '7','8','9','10','11','12',
                                                     '13','14','15','16','17','18',
                                                     '19','20','21','22','23','24'],
                               columns=['S1','S2','S3','S4','S5','S6',
                                                     'S7','S8','S9','S10','S11','S12',
                                                     'S13','S14','S15','S16'])
    Df_rival=pd.DataFrame(arr_riv,index=['Capacity','Cost'], columns=['S1','S2','S3','S4','S5','S6',
                                                     'S7','S8','S9','S10','S11','S12',
                                                     'S13','S14','S15','S16'])
    

    return Df_demand, Df_rival

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

Omega_n_sets = {
    1: [2,3,5],
    2: [1,4,6],
    3: [1,9,24],
    4: [2,9],
    5: [1,10],
    6: [2,10],
    7: [8],
    8: [7,9,10],
    9: [3,4,8,11,12],
    10: [5,6,8,11,12],
    11: [9,13,10,14],
    12: [9,10,13,23],
    13: [11,12,23],
    14: [11,16],
    15: [16,21,24],
    16: [14,15,17,19],
    17: [16,18,22],
    18: [17,21],
    19: [16,20],
    20: [19,23],
    21: [22,18,15],
    22: [17,21],
    23: [12,13,20],
    24: [3,15],
}





