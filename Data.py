import pandas as pd
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