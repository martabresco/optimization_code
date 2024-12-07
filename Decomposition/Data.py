import pandas as pd
from itertools import product
import random
import matplotlib.pyplot as plt
import numpy as np
import math as m
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
        
    Df_demand=pd.DataFrame(arr_dem,index=['0','1','2','3','4','5','6',
                                                     '7','8','9','10','11','12',
                                                     '13','14','15','16','17','18',
                                                     '19','20','21','22','23'],
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

data = [
    (0, 'L1', 1, 2, 0.0146, 175),
    (1, 'L2', 1, 3, 0.2253, 175),
    (2, 'L3', 1, 5, 0.0907, 350),
    (3, 'L4', 2, 4, 0.1356, 175),
    (4, 'L5', 2, 6, 0.205, 175),
    (5, 'L6', 3, 9, 0.1271, 175),
    (6, 'L7', 3, 24, 0.084, 400),
    (7, 'L8', 4, 9, 0.111, 175),
    (8, 'L9', 5, 10, 0.094, 350),
    (9, 'L10', 6, 10, 0.0642, 175),
    (10, 'L11', 7, 8, 0.0652, 350),
    (11, 'L12', 8, 9, 0.1762, 175),
    (12, 'L13', 8, 10, 0.1762, 175),
    (13, 'L14', 9, 11, 0.084, 400),
    (14, 'L15', 9, 12, 0.084, 400),
    (15, 'L16', 10, 11, 0.084, 400),
    (16, 'L17', 10, 12, 0.084, 400),
    (17, 'L18', 11, 13, 0.0488, 500),
    (18, 'L19', 11, 14, 0.0426, 500),
    (19, 'L20', 12, 13, 0.0488, 500),
    (20, 'L21', 12, 23, 0.0985, 500),
    (21, 'L22', 13, 23, 0.0884, 500),
    (22, 'L23', 14, 16, 0.0594, 500),
    (23, 'L24', 15, 16, 0.0172, 500),
    (24, 'L25', 15, 21, 0.0249, 1000),
    (25, 'L26', 15, 24, 0.0529, 500),
    (26, 'L27', 16, 17, 0.0263, 500),
    (27, 'L28', 16, 19, 0.0234, 500),
    (28, 'L29', 17, 18, 0.0143, 500),
    (29, 'L30', 17, 22, 0.1069, 500),
    (30, 'L31', 18, 21, 0.0132, 1000),
    (31, 'L32', 19, 20, 0.0203, 1000),
    (32, 'L33', 20, 23, 0.0112, 1000),
    (33, 'L34', 21, 22, 0.0692, 500)
]

num_nodes=24
matrix = np.zeros((num_nodes, num_nodes))

for _, _, from_node, to_node, x, _ in data:
    from_idx = from_node - 1  # Adjusting for 0-based indexing
    to_idx = to_node - 1
    value = 1 / x if x != 0 else 0  # Avoid division by zero
    matrix[from_idx, to_idx] = value
    matrix[to_idx, from_idx] = value  # Ensure symmetry


# Convert to a DataFrame for better readability
matrix_df = pd.DataFrame(matrix, index=range(0, num_nodes), columns=range(0, num_nodes ))
matrix_df



# Populate the matrix with 1/X values for each line connection

for _, _, from_node, to_node, x, _ in data:
    from_idx = from_node - 1  # Adjusting for 0-based indexing
    to_idx = to_node - 1
    value = 1 / x if x != 0 else 0  # Avoid division by zero
    matrix[from_idx, to_idx] = value
    matrix[to_idx, from_idx] = value  # Ensure symmetry


# Convert to a DataFrame for better readability
matrix_B = pd.DataFrame(matrix, index=range(0, num_nodes), columns=range(0, num_nodes))
matrix_B


df = lines_data

# Determine the number of nodes
num_nodes = max(df["From"].max(), df["To"].max())

# Initialize an n x n matrix with zeros
capacity_matrix = np.zeros((num_nodes, num_nodes))

# Populate the matrix with capacities
for _, row in df.iterrows():
    m, n, capacity = int(row["From"]), int(row["To"]), row["Capacity (MVA)"]
    capacity_matrix[m - 1, n - 1] = capacity  # Adjusting for 0-based indexing
    capacity_matrix[n - 1, m - 1] = capacity  # Assuming undirected graph

# Convert to DataFrame for better visualization (optional)
capacity_matrix = pd.DataFrame(capacity_matrix, index=range(0, num_nodes ), columns=range(0, num_nodes ))

# Print or return the matrix
print(capacity_matrix)


probability_scenario=[1/3,1/3,1/3]


# File path
file_path1 = "DA_prices.xlsx"  # Ensure the file is in your working directory.

# Read all sheets into a dictionary
DA_prices_dict = pd.read_excel(file_path1, sheet_name=None)  # None loads all sheets

# Display all sheets in a dictionary format
for sheet_name, sheet_data in DA_prices_dict.items():
    print(f"Scenario: {sheet_name}")
    print(sheet_data)

# Save all sheets into a 3D-like structure if they have similar rows and columns
DA_prices_3d = {sheet_name: sheet_data.values for sheet_name, sheet_data in DA_prices_dict.items()}

# Accessing individual sheets as numpy array-like structures
print("Accessing one sheet as a numpy array-like structure:")
print(DA_prices_3d['0'])  # Replace with the actual sheet name if different

# Create a 3D numpy array (num_sheets x rows x columns)
# DA_prices_array = np.array([DA_prices_3d[i] for i in range(0, 2)])  # Adjust range if needed
# print("3D Numpy array structure of all scenarios:")
# print(DA_prices_array)



