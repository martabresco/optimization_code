# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:54:12 2024

@author: User
"""

import pandas as pd
import numpy as np
from itertools import product

# Fonction de création des scénarios
def scenarios_creation():
    # Import des fichiers Excel
    demand_scenario = pd.read_excel('Demand_scenario_prep.xlsx', index_col=0).transpose()
    rival_scenario = pd.read_excel('Rival_scenario_prep.xlsx', index_col=0).transpose()
    
    # Création des produits cartésiens des indices (toutes combinaisons possibles)
    demand_indexes = demand_scenario.index
    rival_indexes = rival_scenario.index
    Demand_scenarios, Rival_scenarios = [], []

    for demand_index, rival_index in product(demand_indexes, rival_indexes):
        Demand_scenarios.append(demand_scenario.loc[demand_index].values.tolist())
        Rival_scenarios.append(rival_scenario.loc[rival_index].values.tolist())
    
    # Transposition pour obtenir des tableaux horaires
    Df_demand = pd.DataFrame(
        np.transpose(Demand_scenarios),
        index=[str(i) for i in range(1, 25)],  # Heures de 1 à 24
        columns=[f'S{i}' for i in range(1, 17)]  # S1 à S16
    )
    Df_rival = pd.DataFrame(
        np.transpose(Rival_scenarios),
        index=['Capacity', 'Cost'],
        columns=[f'S{i}' for i in range(1, 17)]
    )
    
    return Df_demand, Df_rival


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


# Chargement des prix du marché
def load_DA_prices(filepath):
    return pd.read_excel(filepath, usecols=[1])  # Utiliser uniquement la 2e colonne


# Création de la matrice de susceptance
def create_susceptance_matrix(data, num_nodes):
    matrix = np.zeros((num_nodes, num_nodes))
    for _, _, from_node, to_node, x, _ in data:
        value = 1 / x if x != 0 else 0
        matrix[from_node - 1, to_node - 1] = value
        matrix[to_node - 1, from_node - 1] = value
    return pd.DataFrame(matrix, index=range(1, num_nodes + 1), columns=range(1, num_nodes + 1))


# Création de la matrice de capacité
def create_capacity_matrix(lines_data):
    num_nodes = max(lines_data["From"].max(), lines_data["To"].max())
    matrix = np.zeros((num_nodes, num_nodes))
    for _, row in lines_data.iterrows():
        from_node, to_node, capacity = int(row["From"]), int(row["To"]), row["Capacity"]
        matrix[from_node - 1, to_node - 1] = capacity
        matrix[to_node - 1, from_node - 1] = capacity
    return pd.DataFrame(matrix, index=range(1, num_nodes + 1), columns=range(1, num_nodes + 1))



Df_demand, Df_rival = scenarios_creation()
# print("Demand Scenarios:")
# print(Df_demand)
# print("\nRival Scenarios:")
# print(Df_rival)

# Données des générateurs
investor_generation_data = load_generation_data("Generation Data.xlsx", "Generation_investor")
rival_generation_data = load_generation_data("Generation Data.xlsx", "Generation_rival")

# Données des lignes
lines_data = load_lines_data("Lines_Data.xlsx")

# Facteurs de capacité
wind_PF, pv_PF = load_power_factors("Wind_PowerFactor_AverageDay.xlsx", "PV_PowerFactor_AverageDay.xlsx")

# Profils et prix de la demande
demand_profile, demand_distribution, demand_prices = load_demand_data("Load dataset.xlsx")

# Données d'investissement
investment_data = load_investment_data("Investment.xlsx")

# Prix Day-Ahead
DA_prices = load_DA_prices("DA_prices.xlsx")

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

# Matrices
susceptance_matrix = create_susceptance_matrix(data=[
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
], num_nodes=24)
# print("\nSusceptance Matrix:")
# print(susceptance_matrix)

capacity_matrix = create_capacity_matrix(lines_data)
# print("\nCapacity Matrix:")
# print(capacity_matrix)
