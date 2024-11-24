import pandas as pd
from itertools import product
import random
import matplotlib.pyplot as plt

def scenarios_creation():
    # Import Excel files
    wind_generation_power = pd.read_excel('wind power generation data.xlsx', index_col=0)
    system_needs_scenario = pd.read_excel('power system need scenarios.xlsx', index_col=0)
    DA_prices = pd.read_excel('DK1_DayAhead_20Days.xlsx', index_col=0)
    
    
    # Transpose DataFrames
    wind_generation_power = wind_generation_power.transpose()
    system_needs_scenario = system_needs_scenario.transpose()
    DA_prices=DA_prices.transpose()
    
    # Get indexes of each DataFrame
    wind_indexes = wind_generation_power.index
    system_indexes = system_needs_scenario.index
    DA_indexes = DA_prices.index
    
    # Initialize lists to store scenarios
    WF_generation_scenarios = []
    DA_prices_scenarios = []
    system_imbalance_scenarios = []

    # Create the Cartesian product of all scenarios
    for wind_index, system_index, DA_index in product(wind_indexes, system_indexes, DA_indexes):
        WF_generation_scenarios.append(wind_generation_power.loc[wind_index].values.tolist())
        DA_prices_scenarios.append(DA_prices.loc[DA_index].values.tolist())
        system_imbalance_scenarios.append(system_needs_scenario.loc[system_index].values.tolist())
    
    return WF_generation_scenarios, DA_prices_scenarios, system_imbalance_scenarios
    
def extract_scenarios(num_scenarios, WF_generation_scenarios, DA_prices_scenarios, system_imbalance_scenarios):

    # Generate num_scenarios unique random indices
    random_indices = random.sample(range(len(WF_generation_scenarios)), num_scenarios)
    
    #Creating a list of all unseen indices, that can be used for the out-of-sample simulation 
   
    # First, create a list of all indices
    all_indices = list(range(len(WF_generation_scenarios)))

    # Create a list of indices that are not part of random_indices
    unseen_indices = [idx for idx in all_indices if idx not in random_indices]
    
    
    # Extract scenarios using the same random indices for all lists
    WF_generation_sublist = [WF_generation_scenarios[i] for i in random_indices]
    DA_prices_sublist = [DA_prices_scenarios[i] for i in random_indices]
    system_imbalance_sublist = [system_imbalance_scenarios[i] for i in random_indices]
    
    # Extract Scenarios using the indices for the unseen scenarios
    WF_generation_sublist_unseen = [WF_generation_scenarios[i] for i in unseen_indices]
    DA_prices_sublist_unseen = [DA_prices_scenarios[i] for i in unseen_indices]
    system_imbalance_sublist_unseen = [system_imbalance_scenarios[i] for i in unseen_indices]
    

    
    # Check the length of sublists
    #print("Number of scenarios in each sublist:", len(WF_generation_sublist))
    
    # Display an example scenario from each sublist
    # print("Example scenario (wind generation):", WF_generation_sublist[0])
    # print("Example scenario (DA prices):", DA_prices_sublist[0])
    # print("Example scenario (system imbalance):", system_imbalance_sublist[0])
    #print("type",type(DA_prices_sublist),'lignes',type(DA_prices_sublist[0]),len(DA_prices_sublist[0]))
    return WF_generation_sublist, DA_prices_sublist, system_imbalance_sublist, WF_generation_sublist_unseen, DA_prices_sublist_unseen, system_imbalance_sublist_unseen

# # Generate scenarios
# WF_generation_scenarios, DA_prices_scenarios, system_imbalance_scenarios = scenarios_creation()

# # Extract 250 scenarios
# WF_generation_sublist, DA_prices_sublist, system_imbalance_sublist, WF_generation_sublist_unseen, DA_prices_sublist_unseen, system_imbalance_sublist_unseen = extract_scenarios(250, WF_generation_scenarios, DA_prices_scenarios, system_imbalance_scenarios)

def data_plot(data, title, x_axis, y_axis, graph_type=1):
     hours = list(range(1, 25))
    
     # Définir une liste de couleurs pour chaque ligne
     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
     # Tracer les données en nuages de points
     if graph_type==1:
         for i, row in enumerate(data):
            plt.scatter(hours, row, color=colors[i % len(colors)], label=f'Hour {i+1}')
     if graph_type==2:
         for i, row in enumerate(data):
             plt.plot(hours, row, color=colors[i % len(colors)], label=f'Hour {i+1}')
     else:
         print("graph_type problem")
     #Ajouter des légendes, un titre et des étiquettes d'axe
     # Ajouter des légendes, un titre et des étiquettes d'axe
     #plt.legend()
     #plt.title(title)
     plt.xlabel(x_axis)
     plt.ylabel(y_axis)
     plt.grid(True)
    
     # Afficher le graphique
     plt.show()


     return None