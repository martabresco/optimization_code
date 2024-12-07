import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
import timeit
from Data import investor_generation_data
from Data import DA_prices_3d, probability_scenario, Wind_PF_data, Investment_data

# Sett opp data
N = 24  # Antall noder
T = 24  # Antall tidsperioder
Omega = 3  # Antall scenarioer

# Parametere
phi = {w: 1 / Omega for w in range(Omega)}  # Scenario sannsynligheter

lambda_nwt = {
    (n, w, t): DA_prices_3d[str(w)][t, n]
    for w in range(Omega)
    for t in range(T)
    for n in range(N)
}

CE = Wind_PF_data.iloc[:, 1].to_dict()  # Eksisterende produksjonskostnad
QW_values = Wind_PF_data.iloc[:, 1].to_dict()

# Konstruer QW_max som en dictionary med noder og tidsperioder
QW_max = {(t): QW_values[t] for t in range(T)}
PE_max = investor_generation_data.iloc[:, 2].to_dict()  # Maksimal eksisterende produksjon
XW_max = 150  # Maksimal vindkapasitet
Kmax = 9e7   # Maksimal samlet investering
KW = Investment_data.iloc[2, 1]  # Kostnad per enhet vindkapasitet

# Modell
model = gp.Model("Optimization_Model")

# Variabler
xW = model.addVars(N, name="xW", lb=0, ub=XW_max)
pW = model.addVars(N, Omega, T, name="pW", lb=0)
pE = model.addVars(N, Omega, T, name="pE", lb=0,
                   ub={(n, w, t): PE_max[n] for n in range(N) for w in range(Omega) for t in range(T)})

# Legg til restriksjoner for å tvinge xW[n] = 0 for n ≠ 14
model.addConstrs((xW[n] == 0 for n in range(N) if n != 14), name="Fix_xW_others")

# Tving pW[n,w,t] = 0 for n ≠ 14
model.addConstrs((pW[n, w, t] == 0 for n in range(N) if n != 14 for w in range(Omega) for t in range(T)), name="Fix_pW_others")

# Restriksjoner
model.addConstrs((xW[n] <= XW_max for n in range(N)), name="Wind_Capacity")

model.addConstr(gp.quicksum(KW * xW[n] for n in range(N)) <= Kmax, name="Total_Capacity")

model.addConstrs(
    (pW[n, w, t] <= QW_max[t] * xW[n] for n in range(N) for w in range(Omega) for t in range(T)),
    name="Wind_Production"
)

model.addConstrs(
    (pE[n, w, t] <= PE_max[n] for n in range(N) for w in range(Omega) for t in range(T)),
    name="Existing_Production"
)

# Objektivfunksjon
objective = gp.quicksum(
    phi[w] * gp.quicksum(
        gp.quicksum(
            lambda_nwt[n, w, t] * (pW[n, w, t] + pE[n, w, t]) - pE[n, w, t] * CE[n]
            for n in range(N)
        )
        for t in range(T)
    )
    for w in range(Omega)
) - gp.quicksum(KW * xW[n] for n in range(N))

model.setObjective(objective, GRB.MINIMIZE)

# Optimaliser modellen
model.optimize()

# Resultater
if model.status == GRB.OPTIMAL:
    print("Optimal løsning funnet:")
    print(f"Objektivverdi: {model.objVal}")
    for n in range(N):
        print(f"xW[{n}] = {xW[n].x}")
else:
    print("Ingen optimal løsning funnet.")
