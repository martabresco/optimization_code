import pandas as pd
import numpy as np
from scipy.stats import norm

# Functions to download 

def load_generation_data(filepath, sheet_name):
    return pd.read_excel(filepath, sheet_name=sheet_name)

def load_power_factors(filepath_wind, filepath_pv):
    wind_PF = pd.read_excel(filepath_wind)  # Faktorer for vind
    pv_PF = pd.read_excel(filepath_pv)      # Faktorer for sol
    return wind_PF, pv_PF

def load_investment_data(filepath):
    return pd.read_excel(filepath)

def DA_prices_creation():
    file_path = 'DK1_DayAhead_20Days.xlsx'
    xls = pd.ExcelFile(file_path)
    data = pd.read_excel(xls, sheet_name='Actual Data')

    scenarios = data.iloc[:, 1:]

    shift_file_path = 'shift_parameters.xlsx'
    shift_data = pd.read_excel(shift_file_path)

    normal_params = {}

    for scenario in scenarios.columns:
        scenario_data = scenarios[scenario]
        mean, std = norm.fit(scenario_data)
        normal_params[scenario] = {'Mean': mean, 'Std Dev': std}

    quantiles = np.linspace(0.0385, 0.9615, 24)

    hourly_prices = np.zeros((20, 24, 24))

    for node_index, row in shift_data.iterrows():
        node = int(row['node'])
        shifting_factor = row['shifting factor']

        for scenario_index, scenario in enumerate(normal_params):
            mean = normal_params[scenario]['Mean']
            std = normal_params[scenario]['Std Dev']

            scenario_quantiles = norm.ppf(quantiles, loc=mean, scale=std)
            scenario_quantiles *= shifting_factor

            hourly_prices[scenario_index, node-1, :] = scenario_quantiles

    return hourly_prices

# Laste nødvendige data
investor_generation_data = load_generation_data("Generation Data.xlsx", "Generation_investor")
investor_generation_data_d = load_generation_data("Generation Data.xlsx", "Generation_investor_decom")
wind_PF, pv_PF = load_power_factors("Wind_PowerFactor_AverageDay.xlsx", "PV_PowerFactor_AverageDay.xlsx")
investment_data = load_investment_data("Investment.xlsx")
DA_prices = DA_prices_creation()
