# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:15:49 2024

@author: anjal
"""

# Defining the given values
investment_cost = 175000  # in EUR (CAPEx in year 0)
revenue_per_year = 100000  # annual revenue in EUR
operational_cost_per_year = 25000  # annual OPEX in EUR
project_lifetime = 3  # in years
cost_of_capital = 0.085  # 8.5%

# Calculating annual net cash flow (Net CF)
annual_net_cash_flow = revenue_per_year - operational_cost_per_year

# Calculating discounted cash flows for each year and summing them up
discounted_cash_flows = [
    annual_net_cash_flow / ((1 + cost_of_capital) ** year) for year in range(1, project_lifetime + 1)
]

# Summing discounted cash flows
total_discounted_cash_flows = sum(discounted_cash_flows)

# Calculating NPV
npv = total_discounted_cash_flows - investment_cost

# Converting to thousands (tEUR) for output
npv_teur = npv / 1000


print(npv_teur)
