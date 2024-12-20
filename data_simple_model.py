# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:54:12 2024

@author: User
"""

import pandas as pd
import numpy as np
from itertools import product


# Chargement des données des générateurs
def load_generation_data(filepath, sheet_name):
    return pd.read_excel(filepath, sheet_name=sheet_name)


# Chargement des lignes de transmission
def load_lines_data(filepath):
    return pd.read_excel(filepath)


# Chargement des facteurs de capacité pour l'éolien et le PV
def load_power_factors(filepath_wind, filepath_pv):
    wind_PF = pd.read_excel(filepath_wind)  # Facteurs éoliens
    pv_PF = pd.read_excel(filepath_pv)      # Facteurs PV
    return wind_PF, pv_PF


# Chargement des profils de charge et des prix
def load_demand_data(filepath):
    demand_profile = pd.read_excel(filepath, sheet_name="Load profile")
    demand_distribution = pd.read_excel(filepath, sheet_name="Load distribution")
    demand_prices = pd.read_excel(filepath, sheet_name="Demand prices")
    return demand_profile, demand_distribution, demand_prices


# Chargement des données d'investissement
def load_investment_data(filepath):
    return pd.read_excel(filepath)


from scipy.stats import norm

def DA_prices_creation():
    #now return the numpy array but you could also retrieve the dataframe if you want

    # Charger les données de prix des scénarios
    file_path = 'DK1_DayAhead_20Days.xlsx'
    xls = pd.ExcelFile(file_path)
    data = pd.read_excel(xls, sheet_name='Actual Data')
    
    # Extraire les données des scénarios (S1 à S20)
    scenarios = data.iloc[:, 1:]
    
    # Charger les facteurs de décalage
    shift_file_path = 'shift_parameters.xlsx'
    shift_data = pd.read_excel(shift_file_path)
    
    # Créer un dictionnaire pour stocker les paramètres ajustés pour chaque scénario
    normal_params = {}
    
    # Pour chaque scénario, ajuster les paramètres de la distribution normale en utilisant fit
    for scenario in scenarios.columns:
        scenario_data = scenarios[scenario]
        
        # Ajuster les paramètres de la distribution normale (moyenne et écart-type)
        mean, std = norm.fit(scenario_data)  # Cela ajuste la distribution normale par MLE
        
        # Stocker les résultats dans le dictionnaire
        normal_params[scenario] = {'Mean': mean, 'Std Dev': std}
    
    # Définir les quantiles (26 quantiles égaux entre 0 et 1, excluant 0 et 1)
    quantiles = np.linspace(0.0385, 0.9615, 24)  # Exclut les 0th et 100th percentiles (2nd to 25th quantiles)
    
    # Créer une matrice 3D (20 scénarios, 24 nœuds, 24 heures) pour les prix horaires
    hourly_prices = np.zeros((20, 24, 24))  # 20 scenarios, 24 nodes, 24 hours
    
    # Appliquer les facteurs de décalage aux moyennes pour chaque nœud et calculer les quantiles
    for node_index, row in shift_data.iterrows():
        node = int(row['node'])  # Assurez-vous que node est un entier
        shifting_factor = row['shifting factor']
        
        # Pour chaque scénario, calculer les quantiles
        for scenario_index, scenario in enumerate(normal_params):
            mean = normal_params[scenario]['Mean']
            std = normal_params[scenario]['Std Dev']
            
            # Calculer les quantiles pour chaque scénario et chaque nœud en utilisant la fonction ppf
            scenario_quantiles = norm.ppf(quantiles, loc=mean, scale=std)
            
            # Multiplier les quantiles par le facteur de décalage après calcul
            scenario_quantiles *= shifting_factor
            
            # Remplir la matrice 3D pour les prix horaires
            hourly_prices[scenario_index, node-1, :] = scenario_quantiles  # Utiliser node-1 pour l'indexation correcte
    
    # # Créer les niveaux d'index pour le DataFrame (Scénarios, Nœuds, Heures)
    # scenarios_index = [f'Scenario {i+1}' for i in range(20)]
    # nodes_index = [f'Node {i+1}' for i in range(24)]
    # hours_index = [f'Hour {i+1}' for i in range(24)]
    
    # # Convertir la matrice en DataFrame pour une structure MultiIndex
    # hourly_prices_df = pd.DataFrame(hourly_prices.reshape(20 * 24, 24), columns=hours_index)
    # hourly_prices_df.index = pd.MultiIndex.from_product([scenarios_index, nodes_index], names=["Scenario", "Node"])
    
    return hourly_prices #now return the numpy array but you could also retrieve the dataframe if you want



# Données des générateurs
investor_generation_data = load_generation_data("Generation Data.xlsx", "Generation_investor")
# rival_generation_data = load_generation_data("Generation Data.xlsx", "Generation_rival")

# Données des lignes
# lines_data = load_lines_data("Lines_Data.xlsx")

# Facteurs de capacité
wind_PF, pv_PF = load_power_factors("Wind_PowerFactor_AverageDay.xlsx", "PV_PowerFactor_AverageDay.xlsx")

# Profils et prix de la demande
# demand_profile, demand_distribution, demand_prices = load_demand_data("Load dataset.xlsx")

# Données d'investissement
investment_data = load_investment_data("Investment.xlsx")

# Prix Day-Ahead
DA_prices = DA_prices_creation()

 