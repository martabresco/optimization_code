import pandas as pd
from itertools import product
import numpy as np
import math as m

# Scenario Creation
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
    
    arr_dem = np.transpose(np.array(Demand_scenarios))
    arr_riv = np.transpose(np.array(Rival_scenarios))
        
    Df_demand = pd.DataFrame(arr_dem, index=['0','1','2','3','4','5','6',
                                             '7','8','9','10','11','12',
                                             '13','14','15','16','17','18',
                                             '19','20','21','22','23'],
                             columns=['S1','S2','S3','S4','S5','S6',
                                      'S7','S8','S9','S10','S11','S12',
                                      'S13','S14','S15','S16'])
    Df_rival = pd.DataFrame(arr_riv, index=['Capacity','Cost'], 
                            columns=['S1','S2','S3','S4','S5','S6',
                                     'S7','S8','S9','S10','S11','S12',
                                     'S13','S14','S15','S16'])
    return Df_demand, Df_rival

# Load Generation Data from investor
file_path1 = "Generation Data.xlsx"
investor_generation_data = pd.read_excel(file_path1, sheet_name="Generation_investor")

# Load generation data from rival
rival_generation_data = pd.read_excel(file_path1, sheet_name="Generation_rival")

# Lines Generation Data
file_path2 = "Lines_Data.xlsx"
lines_data = pd.read_excel(file_path2)

# Power Capacity Factor for Wind Data
file_path3 = "Wind_PowerFactor_AverageDay.xlsx"
Wind_PF_data = pd.read_excel(file_path3)

# Power Capacity Factor for PV Data
file_path4 = "PV_PowerFactor_AverageDay.xlsx"
PV_PF_data = pd.read_excel(file_path4)

# Load profile data
file_path5 = "Load dataset.xlsx"
Demand_profile = pd.read_excel(file_path5, sheet_name="Load profile")

# Load distribution data
Demand_distribution = pd.read_excel(file_path5, sheet_name="Load distribution")

# Demand prices
Demand_prices = pd.read_excel(file_path5, sheet_name="Demand prices")

# Investment costs and capacities
file_path6 = "Investment.xlsx"
Investment_data = pd.read_excel(file_path6)

Demand_scenarios, Rival_scenarios = scenarios_creation()

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

data = [
    (0, 'L1', 1, 2, 0.0146, 175),
    (1, 'L2', 1, 3, 0.2253, 175),
    # (remaining data omitted for brevity)
]

num_nodes = 24
matrix = np.zeros((num_nodes, num_nodes))

for _, _, from_node, to_node, x, _ in data:
    from_idx = from_node - 1
    to_idx = to_node - 1
    value = 1 / x if x != 0 else 0
    matrix[from_idx, to_idx] = value
    matrix[to_idx, from_idx] = value

matrix_B = pd.DataFrame(matrix, index=range(0, num_nodes), columns=range(0, num_nodes))

df = lines_data

num_nodes = max(df["From"].max(), df["To"].max())
capacity_matrix = np.zeros((num_nodes, num_nodes))

for _, row in df.iterrows():
    m, n, capacity = int(row["From"]), int(row["To"]), row["Capacity (MVA)"]
    capacity_matrix[m - 1, n - 1] = capacity
    capacity_matrix[n - 1, m - 1] = capacity

capacity_matrix = pd.DataFrame(capacity_matrix, index=range(0, num_nodes), columns=range(0, num_nodes))

probability_scenario = [1/3, 1/3, 1/3]

file_path1 = "DA_prices.xlsx"
DA_prices_dict = pd.read_excel(file_path1, sheet_name=None)

DA_prices_3d = {sheet_name: sheet_data.values for sheet_name, sheet_data in DA_prices_dict.items()}
