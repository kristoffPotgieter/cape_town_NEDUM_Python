# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:52:15 2021

@author: charl
"""

import numpy as np
import pandas as pd
import scipy

#from inequality import *

path_transportation_costs_BAU = "C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/"
path_BAU = 'C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806'
#path_BAU = 'C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'BAU_basile'
scenario = 'scenario2'
if scenario == 'scenario1':
    path_transportation_costs = "C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_20211103"
    path_scenario = 'C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_20211103'
    #path_scenario = 'C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_20211103_basile'
elif scenario == 'scenario2':
    path_transportation_costs = "C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_bus_taxi_20211103"
    path_scenario = 'C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_bus_taxi_20211103'
    #path_scenario = 'C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_bus_taxi_20211103_basile'
    
#### 4- VALIDATION AND BASELINE

#Validation housing prices
#Validation densities
#People per housing type and income classes

#People per housing type and transportation mode
modalsharebyclass = np.empty((4, 5))
for j in np.arange(4):
    modalShares_noct = np.load(path_transportation_costs_BAU + "modalShares_9.npy")[:, :, :, j]
    ODflows_noct = np.load(path_transportation_costs_BAU + "ODflows_9.npy")[:, :, j]
    simulation_households_noct = np.nansum(np.load(path_BAU + '/simulation_households.npy')[9, : , j, :], 0)
    ODflows_absolutenb = (simulation_households_noct[np.newaxis, :] * ODflows_noct)
    modalShares_absolutenb = ODflows_absolutenb[:, :, np.newaxis] * modalShares_noct
    modalsharebyclass[j] = 100 * np.nansum(np.nansum(modalShares_absolutenb, 0), 0) / np.nansum(modalShares_absolutenb)
df = pd.DataFrame(modalsharebyclass).transpose()

distrib_hh = np.nansum(np.load(path_BAU + '/simulation_households.npy')[9, : , :, :], 0)
pd.DataFrame(distrib_hh).transpose().to_excel('C:/Users/charl/OneDrive/Bureau/inequalityCapeTown/baseline_spatial_distribution_per_income_class.xlsx')

#Map of the spatial distribution of people per housing type

#### 5 - SPATIAL INEQUALITIES

#STEP 1: Change in transportation costs of car users, keeping employment centers and transportation modes constant

#Diff transportation costs

tcost_diff = np.empty((4, 24014))

for j in np.arange(4):
    
    ODflows_noct = np.load(path_transportation_costs_BAU + "ODflows_9.npy")
    modalShares_noct = np.load(path_transportation_costs_BAU + "modalShares_9.npy")
    
    if scenario == 'scenario1':
        modalShares_noct = modalShares_noct[:, :, 2, j] #modal share car
        ODflows_noct = ODflows_noct[:, :, j]
        vec_share_car = ODflows_noct * modalShares_noct #part des gens résidant en un lieu qui vont dans un employment center en voiture
        
    elif scenario == 'scenario2':
        modalShares_noct = modalShares_noct[:, :, 2:5, j]
        ODflows_noct = ODflows_noct[:, :, j]
        vec_share_car = ODflows_noct[:, :, np.newaxis] * modalShares_noct #Si autres modes de transports
    
    transportCostModes_noct = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, j, 0, scenario)
    transportCostModes_ct = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, j, 1, scenario)
    if scenario == 'scenario1':
        transportCostModes_noct = transportCostModes_noct[:, :, 2]
        transportCostModes_ct = transportCostModes_ct[:, :, 2]
    elif scenario == 'scenario2':
        transportCostModes_noct = transportCostModes_noct[:, :, 2:5]
        transportCostModes_ct = transportCostModes_ct[:, :, 2:5]
        
    finalTransportCost_noct = np.empty(24014)
    for i in np.arange(0, 24014):
        if scenario == 'scenario1':
            finalTransportCost_noct[i] = np.average(transportCostModes_noct[:, i][~np.isinf(transportCostModes_noct[:, i])], weights = vec_share_car[:, i][~np.isinf(transportCostModes_noct[:, i])])
        elif scenario == 'scenario2':
            finalTransportCost_noct[i] = np.average(transportCostModes_noct[:, i, :][~np.isinf(transportCostModes_noct[:, i, :])], weights = vec_share_car[:, i, :][~np.isinf(transportCostModes_noct[:, i, :])])
    
    finalTransportCost_ct = np.empty(24014)
    for i in np.arange(0, 24014):
        if scenario == 'scenario1':
            finalTransportCost_ct[i] = np.average(transportCostModes_ct[:, i][~np.isinf(transportCostModes_ct[:, i])], weights = vec_share_car[:, i][~np.isinf(transportCostModes_noct[:, i])])
        elif scenario == 'scenario2':
            finalTransportCost_ct[i] = np.average(transportCostModes_ct[:, i, :][~np.isinf(transportCostModes_ct[:, i, :])], weights = vec_share_car[:, i, :][~np.isinf(transportCostModes_noct[:, i, :])])

    simulation_households = np.nansum(np.load(path_BAU + '/simulation_households.npy')[9, : , j, :], 0)
    tcost_diff[j] = 100 * (finalTransportCost_ct - finalTransportCost_noct) / finalTransportCost_noct
    tcost_diff[j][simulation_households == 0] = np.nan

tcost_diff = pd.DataFrame(tcost_diff)
#tcost_diff.transpose().to_excel("C:/Users/charl/OneDrive/Bureau/tcost_diff.xlsx")

#Diff income

finalTransportCost_noct = np.empty((4, 24014))
finalTransportCost_ct = np.empty((4, 24014))

for j in np.arange(4):
    
    ODflows_noct = np.load(path_transportation_costs_BAU + "ODflows_9.npy")
    modalShares_noct = np.load(path_transportation_costs_BAU + "modalShares_9.npy")
    
    if scenario == 'scenario1':
        #modalShares_noct = modalShares_noct[:, :, 2, j] #modal share car 
        ODflows_noct = ODflows_noct[:, :, j]
        vec_share_car = ODflows_noct * modalShares_noct #part des gens résidant en un lieu qui vont dans un employment center en voiture  
    
    elif scenario == 'scenario2':
        #modalShares_noct = modalShares_noct[:, :, 2:5, j]
        modalShares_noct = modalShares_noct[:, :, :, j]
        ODflows_noct = ODflows_noct[:, :, j]
        vec_share_car = ODflows_noct[:, :, np.newaxis] * modalShares_noct #Si autres modes de transports

    transportCostModes_noct = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, j, 0, scenario)
    transportCostModes_ct = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, j, 1, scenario)
    #if scenario == 'scenario1':
        #transportCostModes_noct = transportCostModes_noct[:, :, 2]
        #transportCostModes_ct = transportCostModes_ct[:, :, 2]
    #elif scenario == 'scenario2':
        #transportCostModes_noct = transportCostModes_noct[:, :, 2:5]
        #transportCostModes_ct = transportCostModes_ct[:, :, 2:5]

    for i in np.arange(0, 24014):
        if scenario == 'scenario1':
            finalTransportCost_noct[j, i] = np.average(transportCostModes_noct[:, i][~np.isinf(transportCostModes_noct[:, i])], weights = vec_share_car[:, i][~np.isinf(transportCostModes_noct[:, i])])
        elif scenario == 'scenario2':
            finalTransportCost_noct[j, i] = np.average(transportCostModes_noct[:, i, :][~np.isinf(transportCostModes_noct[:, i, :])], weights = vec_share_car[:, i, :][~np.isinf(transportCostModes_noct[:, i, :])])
    
    for i in np.arange(0, 24014):
        if scenario == 'scenario1':
            finalTransportCost_ct[j, i] = np.average(transportCostModes_ct[:, i][~np.isinf(transportCostModes_ct[:, i])], weights = vec_share_car[:, i][~np.isinf(transportCostModes_noct[:, i])])
        elif scenario == 'scenario2':
            finalTransportCost_ct[j, i] = np.average(transportCostModes_ct[:, i, :][~np.isinf(transportCostModes_ct[:, i, :])], weights = vec_share_car[:, i, :][~np.isinf(transportCostModes_noct[:, i, :])])

averageIncome_noct = np.load(path_transportation_costs_BAU + "averageIncome_9.npy")
finalTransportCost_noct = (8*20*12) * finalTransportCost_noct
finalTransportCost_ct = (8*20*12) * finalTransportCost_ct
residual_income_noct = averageIncome_noct - finalTransportCost_noct
residual_income_ct = averageIncome_noct - finalTransportCost_ct
diff = 100 * (residual_income_ct - residual_income_noct) / residual_income_noct
simulation_households = np.nansum(np.load(path_BAU + '/simulation_households.npy')[9, : , :, :], 0)
diff[simulation_households == 0] = np.nan
df = pd.DataFrame(diff)
df.transpose().to_excel("C:/Users/charl/OneDrive/Bureau/inequalityCapeTown/Writing/Step1/income_net_of_transportation_costs_before_adjustment_all_hh_basile.xlsx")

#### STEP 2: Changes in income net of transportation costs, assuming that people can change transportation mode and employment center

net_income_diff = np.empty((4, 24014))

for j in np.arange(4):
    incomeNetOfCommuting2020_ct = np.load(path_transportation_costs + "/incomeNetOfCommuting_9.npy")
    incomeNetOfCommuting2020_noct = np.load(path_transportation_costs_BAU + "/incomeNetOfCommuting_9.npy")
    incomeNetOfCommuting2020_ct[incomeNetOfCommuting2020_ct <0] = 0
    incomeNetOfCommuting2020_noct[incomeNetOfCommuting2020_noct <0] = 0
    simulation_households = np.nansum(np.load(path_BAU + '/simulation_households.npy')[9, : , j, :], 0)
    diff = 100 * (incomeNetOfCommuting2020_ct[j] - incomeNetOfCommuting2020_noct[j]) / incomeNetOfCommuting2020_noct[j]
    diff[simulation_households == 0] = np.nan
    net_income_diff[j] = diff
    
df = pd.DataFrame(net_income_diff)
df.transpose().to_excel("C:/Users/charl/OneDrive/Bureau/inequalityCapeTown/Writing/Step2/income_net_of_transportation_costs_after_adjustment_basile.xlsx")

### STEP 3: Rents

simulation_rent_noct = np.load(path_BAU + '/simulation_rent.npy')[9, :]
simulation_rent_ct = np.load(path_scenario + '/simulation_rent.npy')[9, :]
simulation_households_noct = np.load(path_BAU + '/simulation_households.npy')[9, : , :, :]
simulation_households_ct = np.load(path_scenario + '/simulation_households.npy')[9, : , :, :]
simulation_rent_noct[(np.nansum(simulation_households_noct[:, :, :], 1) == 0) | (np.nansum(simulation_households_ct[:, :, :], 1) == 0)] = np.nan
simulation_rent_noct[(np.isnan(np.nansum(simulation_households_noct[:, :, :], 1))) | (np.isnan(np.nansum(simulation_households_ct[:, :, :], 1)))] = np.nan
simulation_rent_ct[(np.nansum(simulation_households_noct[:, :, :], 1) == 0) | (np.nansum(simulation_households_ct[:, :, :], 1) == 0)] = np.nan
simulation_rent_ct[(np.isnan(np.nansum(simulation_households_noct[:, :, :], 1))) | (np.isnan(np.nansum(simulation_households_ct[:, :, :], 1)))] = np.nan
diff_rent = 100 * (simulation_rent_ct - simulation_rent_noct) / simulation_rent_noct
df = pd.DataFrame(diff_rent)
df.transpose().to_excel("C:/Users/charl/OneDrive/Bureau/inequalityCapeTown/Writing/Step3/diff_rent_basile.xlsx")

### STEP 4: Changes in the budget dedicated to housing + transportation

#income net of transportation cost
incomeNetOfCommuting2020_ct = np.load(path_transportation_costs + "/incomeNetOfCommuting_9.npy")
incomeNetOfCommuting2020_noct = np.load(path_transportation_costs_BAU + "/incomeNetOfCommuting_9.npy")
incomeNetOfCommuting2020_ct[incomeNetOfCommuting2020_ct <0] = 0
incomeNetOfCommuting2020_noct[incomeNetOfCommuting2020_noct <0] = 0

#Rents: more complicated because we have i) to use dsize ii) match housing type and income class
simulation_rent_noct = np.load(path_BAU + '/simulation_rent.npy')[9, :]
simulation_rent_ct = np.load(path_scenario + '/simulation_rent.npy')[9, :]

simulation_dsize_noct = np.load(path_BAU + '/simulation_dwelling_size.npy')[9, :]
simulation_dsize_ct = np.load(path_scenario + '/simulation_dwelling_size.npy')[9, :]

#### choose if we allow dwelling sizes to change
budget_housing_by_housing_type_ct = simulation_rent_ct * simulation_dsize_ct
budget_housing_by_housing_type_noct = simulation_rent_noct * simulation_dsize_noct


simulation_households_ct = np.load(path_scenario + '/simulation_households.npy')[9, : , :, :]
simulation_households_noct = np.load(path_BAU + '/simulation_households.npy')[9, : , :, :]
budget_housing_by_income_class_ct = np.empty((4, 24014))
budget_housing_by_income_class_noct = np.empty((4, 24014))

#Do we allow households to move ? No, so we use simulation_households_noct
for i in range(0, 24014):
    for j in range(0, 4):
        if (np.nansum(simulation_households_noct[:, j, i]) == 0):
            budget_housing_by_income_class_ct[j, i] = np.nan
            budget_housing_by_income_class_noct[j, i] = np.nan
        else:
            budget_housing_by_income_class_ct[j, i] = np.average(budget_housing_by_housing_type_ct[:, i], weights = simulation_households_noct[:, j, i])
            budget_housing_by_income_class_noct[j, i] = np.average(budget_housing_by_housing_type_noct[:, i], weights = simulation_households_noct[:, j, i])
    
averageIncome_ct = np.load(path_transportation_costs + "/averageIncome_9.npy")
averageIncome_noct = np.load(path_transportation_costs_BAU + "/averageIncome_9.npy")
transportion_budget_noct = averageIncome_noct - incomeNetOfCommuting2020_noct
transportion_budget_ct = averageIncome_ct - incomeNetOfCommuting2020_ct
basic_need_noct = transportion_budget_noct + budget_housing_by_income_class_noct
basic_need_ct = transportion_budget_ct + budget_housing_by_income_class_ct
diff = 100 * (basic_need_ct - basic_need_noct) / basic_need_noct

df = pd.DataFrame(diff)
df.transpose().to_excel("C:/Users/charl/OneDrive/Bureau/inequalityCapeTown/Writing/Step5/basic_need_after_adjustment_basile.xlsx")

#### 6 - AGGREGATED OUTPUTS

#welfare
simulation_utility_noct = np.load(path_BAU + '/simulation_utility.npy')[9, :]
simulation_utility_ct = np.load(path_scenario + '/simulation_utility.npy')[9, :]
100 * (simulation_utility_ct - simulation_utility_noct) / simulation_utility_noct
#array([-0.6419367 , -1.07120226, -0.34847263, -0.07363705])

#welfare decomposition

spline_agricultural_rent, spline_interest_rate, spline_RDP, spline_population_income_distribution, spline_inflation, spline_income_distribution, spline_population, spline_interest_rate, spline_income, spline_minimum_housing_supply, spline_fuel = import_scenarios(income_2011, param, grid, path_scenarios)


A = amenities


B_formal = 1
B_IS = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/fluvial_and_pluvial/param_pockets.npy')
B_IB = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/fluvial_and_pluvial/param_backyards.npy')
B_IS[(spline_land_informal(29) > 0) & (spline_land_informal(0) == 0)] = 0.79
B = np.zeros((24014, 4))
B[:, 0] = 1
B[:, 3] = 1
B[:, 1] = B_IB
B[:, 2] = B_IS

q0 = np.zeros((24014, 4))
q0 = q0 + param['q0']   ### Vérifier que ce n'est que pour le formel

interest_rate = interpolate_interest_rate(spline_interest_rate, 9) #A vérifier

structure_value = np.zeros((24014, 4))
structure_value[:, 1] = (param["informal_structure_value"] * (interest_rate + param["depreciation_rate"])) * (spline_inflation(9) / spline_inflation(0))
structure_value[:, 2] = (param["informal_structure_value"] * (interest_rate + param["depreciation_rate"])) * (spline_inflation(9) / spline_inflation(0))

def compute_array_utility(DWELLING_SIZE, RENTS, NET_INCOME, A, B, q0, structure_value):
    array_utility = np.zeros((4, 4, 24014))
    for income_class in np.arange(4):
        for housing_type in np.arange(4):        
            array_utility[housing_type, income_class, :] = A * B[:, housing_type] * ((DWELLING_SIZE[housing_type, :] - q0[:, housing_type]) ** param["beta"]) * (((NET_INCOME[income_class, :] - (DWELLING_SIZE[housing_type, :] * RENTS[housing_type, :]) - structure_value[:, housing_type])) ** param["alpha"])
            array_utility[housing_type, income_class, :][((NET_INCOME[income_class, :] - (DWELLING_SIZE[housing_type, :] * RENTS[housing_type, :]) - structure_value[:, housing_type]) < 0)] = np.nan
    return array_utility

R_noct = np.load(path_BAU + '/simulation_rent.npy')[9, :]
q_noct = np.load(path_BAU + '/simulation_dwelling_size.npy')[9, :]
income_net_of_tcost_noct = np.load(path_transportation_costs_BAU + "/incomeNetOfCommuting_9.npy")
simulation_households_noct = np.load(path_BAU + '/simulation_households.npy')[9, : , :, :]
income_net_of_tcost_noct[income_net_of_tcost_noct < 0] = 0

R_ct = np.load(path_scenario + '/simulation_rent.npy')[9, :]
q_ct = np.load(path_scenario + '/simulation_dwelling_size.npy')[9, :]
income_net_of_tcost_ct = np.load(path_transportation_costs + "/incomeNetOfCommuting_9.npy")
simulation_households_ct = np.load(path_scenario + '/simulation_households.npy')[9, : , :, :]
income_net_of_tcost_ct[income_net_of_tcost_ct < 0] = 0

housing_supply = (param["alpha"] * (param["RDP_size"] + param["backyard_size"] - param["q0"]) / (param["backyard_size"])) - (param["beta"] * (income_net_of_tcost_noct[0,:]) / (param["backyard_size"] * R_noct[1, :]))
housing_supply[np.isinf(housing_supply)] = np.nan
housing_supply[R_noct[1, :] == 0] = 0
housing_supply = np.minimum(housing_supply, 1)
housing_supply = np.maximum(housing_supply, 0)
housing_supply[np.isnan(housing_supply)] = 0

housing_supply = (param["alpha"] * (param["RDP_size"] + param["backyard_size"] - param["q0"]) / (param["backyard_size"])) - (param["beta"] * (income_net_of_tcost_ct[0,:]) / (param["backyard_size"] * R_ct[1, :]))
housing_supply[np.isinf(housing_supply)] = np.nan
housing_supply[R_ct[1, :] == 0] = 0
housing_supply = np.minimum(housing_supply, 1)
housing_supply = np.maximum(housing_supply, 0)
housing_supply[np.isnan(housing_supply)] = 0


simulation_households_noct = np.load(path_BAU + '/simulation_households.npy')[9, : , :, :]
total_hh = np.nansum(simulation_households_noct)

def find_spatial_quartiles(distance, array_hh, quartile):
    df = pd.DataFrame(np.array([distance, array_hh])).transpose()
    df.columns = ['grid_dist', 'hh']
    df = df.sort_values('grid_dist')
    x = 0
    sum_hh = 0
    while sum_hh < quartile * np.nansum(array_hh):
        x = x + 1
        sum_hh = np.nansum(df.hh.iloc[0:x])
    df2 = pd.DataFrame([np.array(df.index), np.concatenate([np.ones(x), np.zeros(24014-x)])]).transpose()
    df2.columns = ['id', 'quartile']
    df2 = df2.sort_values('id')
    spatial_quartile = np.array(df2.quartile)
    return spatial_quartile


quart1 = find_spatial_quartiles(grid.dist, np.nansum(np.nansum(simulation_households_noct, 0), 0), 0.25)
quart2 = find_spatial_quartiles(grid.dist, np.nansum(np.nansum(simulation_households_noct, 0), 0), 0.5)
quart3 = find_spatial_quartiles(grid.dist, np.nansum(np.nansum(simulation_households_noct, 0), 0), 0.75)

quart1 = find_spatial_quartiles(grid.dist, np.nansum(simulation_households_noct[:, j, :], 0), 0.25)
quart2 = find_spatial_quartiles(grid.dist, np.nansum(simulation_households_noct[:, j, :], 0), 0.5)
quart3 = find_spatial_quartiles(grid.dist, np.nansum(simulation_households_noct[:, j, :], 0), 0.75)


quart2[quart1 == 1] = 0
quart3[(quart1 == 1) | (quart2 == 1)] = 0
quart4 = np.ones(24014)
quart4[(quart1 == 1) | (quart2 == 1)| (quart3 == 1)] = 0

np.nansum(np.nansum(np.nansum(simulation_households_noct, 0), 0)[quart1 == 1])
np.nansum(np.nansum(np.nansum(simulation_households_noct, 0), 0)[quart2 == 1])
np.nansum(np.nansum(np.nansum(simulation_households_noct, 0), 0)[quart3 == 1])
np.nansum(np.nansum(np.nansum(simulation_households_noct, 0), 0)[quart4 == 1])

np.nansum(np.nansum(simulation_households_noct[:, j, :], 0)[quart1 == 1])
np.nansum(np.nansum(simulation_households_noct[:, j, :], 0)[quart2 == 1])
np.nansum(np.nansum(simulation_households_noct[:, j, :], 0)[quart3 == 1])
np.nansum(np.nansum(simulation_households_noct[:, j, :], 0)[quart4 == 1])

np.nanmean(grid.dist[quart1 == 1])
np.nanmean(grid.dist[quart2 == 1])
np.nanmean(grid.dist[quart3 == 1])
np.nanmean(grid.dist[quart4 == 1])


def utility_by_income_class(array_utility, simulation_households):
    
    inc_class_1 = np.average(array_utility[:, 0, :][~np.isnan(array_utility[:, 0, :])], weights = simulation_households[:, 0, :][~np.isnan(array_utility[:, 0, :])])
    inc_class_2 = np.average(array_utility[:, 1, :][~np.isnan(array_utility[:, 1, :])], weights = simulation_households[:, 1, :][~np.isnan(array_utility[:, 1, :])])
    inc_class_3 = np.average(array_utility[:, 2, :][~np.isnan(array_utility[:, 2, :])], weights = simulation_households[:, 2, :][~np.isnan(array_utility[:, 2, :])])
    inc_class_4 = np.average(array_utility[:, 3, :][~np.isnan(array_utility[:, 3, :])], weights = simulation_households[:, 3, :][~np.isnan(array_utility[:, 3, :])])
    return np.array([inc_class_1, inc_class_2, inc_class_3, inc_class_4])

def utility_by_spatial_class(array_utility, simulation_households, quart1, quart2, quart3, quart4):
    
    spatial_class1 = np.average(array_utility[~np.isnan(array_utility) & (quart1 == 1)], weights = simulation_households[~np.isnan(array_utility) & (quart1 == 1)])
    spatial_class2 = np.average(array_utility[~np.isnan(array_utility) & (quart2 == 1)], weights = simulation_households[~np.isnan(array_utility) & (quart2 == 1)])
    spatial_class3 = np.average(array_utility[~np.isnan(array_utility) & (quart3 == 1)], weights = simulation_households[~np.isnan(array_utility) & (quart3 == 1)])
    spatial_class4 = np.average(array_utility[~np.isnan(array_utility) & (quart4 == 1)], weights = simulation_households[~np.isnan(array_utility) & (quart4 == 1)])
    return np.array([spatial_class1, spatial_class2, spatial_class3, spatial_class4])

def utility_by_housing_type(array_utility, simulation_households):
    FP = np.average(array_utility[0, :, :][~np.isnan(array_utility[0, :, :])], weights = simulation_households[0, :, :][~np.isnan(array_utility[0, :, :])])
    IB = np.average(array_utility[1, :, :][~np.isnan(array_utility[1, :, :])], weights = simulation_households[1, :, :][~np.isnan(array_utility[1, :, :])])
    IS = np.average(array_utility[2, :, :][~np.isnan(array_utility[2, :, :])], weights = simulation_households[2, :, :][~np.isnan(array_utility[2, :, :])])
    FS = np.average(array_utility[3, :, :][~np.isnan(array_utility[3, :, :])], weights = simulation_households[3, :, :][~np.isnan(array_utility[3, :, :])])
    return np.array([FP, IB, IS, FS])

#initial state
np.load(path_BAU + '/simulation_utility.npy')[9, :]
array_utility_init = compute_array_utility(q_noct, R_noct, income_net_of_tcost_noct, A, B, q0, structure_value)
array_utility_init[simulation_households_noct == 0] = np.nan

arr = np.abs(array_utility_init[0, 3, :] - 2283.09)

plt.scatter(grid.x, grid.y, c = arr)
plt.colorbar()

arr_0_0 = array_utility_init[0, 0, :][~np.isnan(array_utility_init[0, 0, :])]
arr_1_0 = array_utility_init[1, 0, :][~np.isnan(array_utility_init[1, 0, :])]
arr_2_0 = array_utility_init[2, 0, :][~np.isnan(array_utility_init[2, 0, :])]
arr_3_0 = array_utility_init[3, 0, :][~np.isnan(array_utility_init[3, 0, :])] #?

arr_0_1 = array_utility_init[0, 1, :][~np.isnan(array_utility_init[0, 1, :])] #?
arr_1_1 = array_utility_init[1, 1, :][~np.isnan(array_utility_init[1, 1, :])]
arr_2_1 = array_utility_init[2, 1, :][~np.isnan(array_utility_init[2, 1, :])]

arr_0_2 = array_utility_init[0, 2, :][~np.isnan(array_utility_init[0, 2, :])] #?
arr_0_3 = array_utility_init[0, 3, :][~np.isnan(array_utility_init[0, 3, :])] #?

utility_income_class_init = utility_by_income_class(array_utility_init, simulation_households_noct)
utility_housing_type_init = utility_by_housing_type(array_utility_init, simulation_households_noct)
utility_spatial_class_init = utility_by_spatial_class(array_utility_init[:, j, :], simulation_households_noct[:, j, :], quart1, quart2, quart3, quart4)

#change in transportation costs

ODflows_noct = np.load(path_transportation_costs_BAU + "/ODflows_9.npy")
modalShares_noct = np.load(path_transportation_costs_BAU + "/modalShares_9.npy")
weighting_noct = ODflows_noct[:, :, np.newaxis, :] * modalShares_noct #part des gens résidant en un lieu qui vont dans un employment center en voiture
transportCostModes_ct_0 = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, 0, 1, scenario)
transportCostModes_ct_1 = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, 1, 1, scenario)
transportCostModes_ct_2 = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, 2, 1, scenario)
transportCostModes_ct_3 = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, 3, 1, scenario)
transportCostModes_ct = np.array([transportCostModes_ct_0, transportCostModes_ct_1, transportCostModes_ct_2, transportCostModes_ct_3])

finalTransportCost_ct = np.empty((4, 24014))
for i in np.arange(0, 24014):
    for k in np.arange(0, 4):
        finalTransportCost_ct[k, i] = np.average(transportCostModes_ct[k, :, i, :][~np.isinf(transportCostModes_ct[k, :, i, :])], weights = weighting_noct[:, i, :, k][~np.isinf(transportCostModes_ct[k, :, i, :])])

averageIncome_noct = np.load(path_transportation_costs_BAU + "averageIncome_9.npy")
finalTransportCost_ct = (8*20*12) * finalTransportCost_ct
residual_income_ct = averageIncome_noct - finalTransportCost_ct
residual_income_ct[residual_income_ct < 0] = 0

array_utility_step1 = compute_array_utility(q_noct, R_noct, residual_income_ct, A, B, q0, structure_value)
utility_income_class_change_tcost = utility_by_income_class(array_utility_step1, simulation_households_noct)
utility_housing_type_change_tcost = utility_by_housing_type(array_utility_step1, simulation_households_noct)
utility_spatial_class_change_tcost = utility_by_spatial_class(array_utility_step1[:, j, :], simulation_households_noct[:, j, :], quart1, quart2, quart3, quart4)

#change in transportation costs, modes and employment centers
array_utility_step2 = compute_array_utility(q_noct, R_noct, income_net_of_tcost_ct, A, B, q0, structure_value)
utility_income_class_change_tcostv2 = utility_by_income_class(array_utility_step2, simulation_households_noct)
utility_housing_type_change_tcostv2 = utility_by_housing_type(array_utility_step2, simulation_households_noct)
utility_spatial_class_change_tcostv2 = utility_by_spatial_class(array_utility_step2[:, j, :], simulation_households_noct[:, j, :], quart1, quart2, quart3, quart4)


#changes in transportation costs, modes, employment centers and rents
array_utility_step3 = compute_array_utility(q_noct, R_ct, income_net_of_tcost_ct, A, B, q0, structure_value)
utility_income_class_change_rents = utility_by_income_class(array_utility_step3, simulation_households_noct)
utility_housing_type_change_rents = utility_by_housing_type(array_utility_step3, simulation_households_noct)
utility_spatial_class_change_rents = utility_by_spatial_class(array_utility_step3[:, j, :], simulation_households_noct[:, j, :], quart1, quart2, quart3, quart4)

#changes in transportation costs, modes, employment centers, rents, and dwelling sizes
array_utility_step3b = compute_array_utility(q_ct, R_ct, income_net_of_tcost_ct, A, B, q0, structure_value)
utility_income_class_change_dsize = utility_by_income_class(array_utility_step3b, simulation_households_noct)
utility_housing_type_change_dsize = utility_by_housing_type(array_utility_step3b, simulation_households_noct)
utility_spatial_class_change_dsize = utility_by_spatial_class(array_utility_step3b[:, j, :], simulation_households_noct[:, j, :], quart1, quart2, quart3, quart4)

#final state
np.load(path_scenario + '/simulation_utility.npy')[9, :]
array_utility_final_state = compute_array_utility(q_ct, R_ct, income_net_of_tcost_ct, A, B, q0, structure_value)
utility_income_class_final = utility_by_income_class(array_utility_final_state, simulation_households_ct)
#utility_housing_type_final = utility_by_housing_type(array_utility_final_state, simulation_households_ct)

100 * (utility_income_class_final - utility_income_class_init) / utility_income_class_init
#array([-0.56497585, -0.97911525, -0.34530937, -0.06237502])
step1 = 100 * (utility_income_class_change_tcost - utility_income_class_init) / utility_income_class_init
step2 = 100 * (utility_income_class_change_tcostv2 - utility_income_class_change_tcost) / utility_income_class_change_tcost
step3 = 100 * (utility_income_class_change_rents - utility_income_class_change_tcostv2) / utility_income_class_change_tcostv2
step3b = 100 * (utility_income_class_change_dsize - utility_income_class_change_rents) / utility_income_class_change_rents
step4 = 100 * (utility_income_class_final - utility_income_class_change_dsize) / utility_income_class_change_dsize
#step4 = 100 * (utility_income_class_final - utility_income_class_change_rents) / utility_income_class_change_rents

100 * ((((1 + step1/100) * (1 + step2/100)) - 1) - ((1 + step1/100) - 1))
100 * ((((1 + step1/100) * (1 + step2/100)* (1 + step3/100)) - 1) - ((1 + step1/100)* (1 + step2/100) - 1))
100 * (((1 + step1/100) * (1 + step2/100)* (1 + step3/100)* (1 + step3b/100)* (1 + step4/100) - 1) - ((1 + step1/100)* (1 + step2/100)* (1 + step3/100) - 1))
100 * (((1 + step1/100) * (1 + step2/100) - 1) - ((1 + step1/100) - 1))

df = pd.DataFrame([utility_income_class_init, utility_income_class_change_tcost, utility_income_class_change_tcostv2, utility_income_class_change_rents, utility_income_class_final])

cmap = plt.cm.get_cmap('magma_r')
rgba = cmap(0.5)
print(rgba)

plt.rcParams['figure.dpi'] = 360
plt.figure(figsize=(8, 4))
plt.style.use('seaborn-whitegrid')
plt.plot(np.arange(5), df[0]/(df[0][0]), label = 'Income class 1', color = cmap(0.2))
plt.plot(np.arange(5), df[1]/(df[1][0]), label = 'Income class 2', color = cmap(0.4))
plt.plot(np.arange(5), df[2]/(df[2][0]), label = 'Income class 3', color = cmap(0.6))
plt.plot(np.arange(5), df[3]/(df[3][0]), label = 'Income class 4', color = cmap(0.8))
plt.ylabel("Total welfare (base 2020)", fontsize = 12)
ax = plt.subplot(111) 
ax.spines["top"].set_visible(False)   
ax.spines["right"].set_visible(False) 
#ax.spines["bottom"].set_visible(False) 
#ax.spines["left"].set_visible(False) 
#ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")   
plt.xticks(np.arange(5), ["Before \n implementation", "Direct impact", "Employment \n center and \n mode choice \n adjustment", "Rents adjustment", "Dwelling size \n and households \n locations adustment"], rotation=0, fontsize = 12)
plt.legend(facecolor="white", framealpha = 1, frameon = 1, fontsize = 12)


step1 = 100 * (utility_spatial_class_change_tcost - utility_spatial_class_init) / utility_spatial_class_init
step2 = 100 * (utility_spatial_class_change_tcostv2 - utility_spatial_class_change_tcost) / utility_spatial_class_change_tcost
step3 = 100 * (utility_spatial_class_change_rents - utility_spatial_class_change_tcostv2) / utility_spatial_class_change_tcostv2
#step3b = 100 * (utility_spatial_class_change_dsize - utility_spatial_class_change_rents) / utility_spatial_class_change_rents

100 * ((1 + step1/100) - 1)
100 * ((1 + step1/100) * (1 + step2/100) - 1) - 100 * ((1 + step1/100) - 1)
100 * ((1 + step1/100) * (1 + step2/100) * (1 + step3/100) - 1) - 100 * ((1 + step1/100) * (1 + step2/100) - 1)
#100 * ((1 + step1/100) * (1 + step2/100) * (1 + step3/100)* (1 + step3b/100) - 1) - 100 * ((1 + step1/100) * (1 + step2/100) * (1 + step3/100) - 1)


df = pd.DataFrame([utility_spatial_class_init, utility_spatial_class_change_tcost, utility_spatial_class_change_tcostv2, utility_spatial_class_change_rents])

cmap = plt.cm.get_cmap('magma')
rgba = cmap(0.5)
print(rgba)

plt.rcParams['figure.dpi'] = 360
plt.figure(figsize=(8, 4))
plt.style.use('seaborn-whitegrid')
plt.plot(np.arange(4), df[0]/(df[0][0]), label = 'Spatial quartile 1', color = cmap(0.2))
plt.plot(np.arange(4), df[1]/(df[1][0]), label = 'Spatial quartile 2', color = cmap(0.4))
plt.plot(np.arange(4), df[2]/(df[2][0]), label = 'Spatial quartile 3', color = cmap(0.6))
plt.plot(np.arange(4), df[3]/(df[3][0]), label = 'Spatial quartile 4', color = cmap(0.8))
plt.ylabel("Total welfare (base 2020)", fontsize = 12)
ax = plt.subplot(111) 
ax.spines["top"].set_visible(False)   
ax.spines["right"].set_visible(False) 
#ax.spines["bottom"].set_visible(False) 
#ax.spines["left"].set_visible(False) 
#ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")   
plt.xticks(np.arange(4), ["Before \n implementation", "Direct impact", "Employment \n center and \n mode choice \n adjustment", "Rents adjustment"], rotation=0, fontsize = 12)
plt.legend(facecolor="white", framealpha = 1, frameon = 1, fontsize = 12)


100 * (utility_housing_type_final - utility_housing_type_init) / utility_housing_type_init

step1 = 100 * (utility_housing_type_change_tcost - utility_housing_type_init) / utility_housing_type_init
step2 = 100 * (utility_housing_type_change_tcostv2 - utility_housing_type_change_tcost) / utility_housing_type_change_tcost
step3 = 100 * (utility_housing_type_change_rents - utility_housing_type_change_tcostv2) / utility_housing_type_change_tcostv2
step3b = 100 * (utility_housing_type_change_dsize - utility_housing_type_change_rents) / utility_housing_type_change_rents
#step4 = 100 * (utility_housing_type_final - utility_housing_type_change_dsize) / utility_housing_type_change_dsize

100 * ((1 + step1/100) * (1 + step2/100) * (1 + step3/100)* (1 + step3b/100) * (1 + step4 /100) - 1)

100 * ((1 + step1/100) - 1)
100 * ((1 + step1/100) * (1 + step2/100) - 1) - 100 * ((1 + step1/100) - 1)
100 * ((1 + step1/100) * (1 + step2/100) * (1 + step3/100) - 1) - 100 * ((1 + step1/100) * (1 + step2/100) - 1)
100 * ((1 + step1/100) * (1 + step2/100) * (1 + step3/100)* (1 + step3b/100) - 1) - 100 * ((1 + step1/100) * (1 + step2/100) * (1 + step3/100) - 1)

df = pd.DataFrame([utility_housing_type_init, utility_housing_type_change_tcost, utility_housing_type_change_tcostv2, utility_housing_type_change_rents, utility_housing_type_change_dsize])

cmap = plt.cm.get_cmap('magma')
rgba = cmap(0.5)
print(rgba)

plt.rcParams['figure.dpi'] = 360
plt.figure(figsize=(8, 4))
plt.style.use('seaborn-whitegrid')
plt.plot(np.arange(5), df[0]/(df[0][0]), label = 'Formal private', color = cmap(0.2))
plt.plot(np.arange(5), df[1]/(df[1][0]), label = 'Informal in backyards', color = cmap(0.4))
plt.plot(np.arange(5), df[2]/(df[2][0]), label = 'Informal settlements', color = cmap(0.6))
plt.plot(np.arange(5), df[3]/(df[3][0]), label = 'Formal subsidized', color = cmap(0.8))
plt.ylabel("Total welfare (base 2020)", fontsize = 12)
ax = plt.subplot(111) 
ax.spines["top"].set_visible(False)   
ax.spines["right"].set_visible(False) 
#ax.spines["bottom"].set_visible(False) 
#ax.spines["left"].set_visible(False) 
#ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")   
plt.xticks(np.arange(5), ["Before \n implementation", "Direct impact", "Employment \n center and \n mode choice \n adjustment", "Rents adjustment", "Dwelling size \n adustment"], rotation=0, fontsize = 12)
plt.legend(facecolor="white", framealpha = 1, frameon = 1, fontsize = 12)


#emissions

pkm_noct = np.empty((4, 5))
pkm_ct = np.empty((4, 5))
transport_times = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/Basile data/Transport_times_GRID.mat')
    
for j in np.arange(4):
    modalShares_noct = np.load(path_transportation_costs_BAU + "/modalShares_9.npy")[:, :, :, j]
    ODflows_noct = np.load(path_transportation_costs_BAU + "/ODflows_9.npy")[:, :, j]
    simulation_households_noct = np.nansum(np.load(path_BAU + '/simulation_households.npy')[9, : , j, :], 0)
    ODflows_absolutenb = (simulation_households_noct[np.newaxis, :] * ODflows_noct)
    modalShares_absolutenb = ODflows_absolutenb[:, :, np.newaxis] * modalShares_noct 
    nb_p_km = modalShares_absolutenb * transport_times["distanceCar"][:, :, np.newaxis]
    pkm_noct[j] = np.nansum(np.nansum(nb_p_km, 0), 0)
    
for j in np.arange(4):
    modalShares_noct = np.load(path_transportation_costs + "/modalShares_9.npy")[:, :, :, j]
    ODflows_noct = np.load(path_transportation_costs + "/ODflows_9.npy")[:, :, j]
    simulation_households_noct = np.nansum(np.load(path_scenario + '/simulation_households.npy')[9, : , j, :], 0)
    ODflows_absolutenb = (simulation_households_noct[np.newaxis, :] * ODflows_noct)
    modalShares_absolutenb = ODflows_absolutenb[:, :, np.newaxis] * modalShares_noct
    nb_p_km = modalShares_absolutenb * transport_times["distanceCar"][:, :, np.newaxis]
    pkm_ct[j] = np.nansum(np.nansum(nb_p_km, 0), 0)
    

100 * (np.nansum(pkm_ct, 0) - np.nansum(pkm_noct, 0)) /  np.nansum(pkm_noct, 0)
em_ct = np.nansum(np.nansum(pkm_ct, 0) * np.array([0, 0.05, 0.150, 0.06, 0.04]))
em_noct = np.nansum(np.nansum(pkm_noct, 0) * np.array([0, 0.05, 0.150, 0.06, 0.04]))
100 * (em_ct -em_noct) /  em_noct

#emissions decomposition
modal_shares_noct = np.empty((4, 5))
modal_shares_ct = np.empty((4, 5))

dist_noct = np.empty((4, 5))
dist_ct = np.empty((4, 5))

for j in np.arange(4):
    modalShares_noct = np.load(path_transportation_costs_BAU + "/modalShares_9.npy")[:, :, :, j]
    ODflows_noct = np.load(path_transportation_costs_BAU + "/ODflows_9.npy")[:, :, j]
    simulation_households_noct = np.nansum(np.load(path_BAU + '/simulation_households.npy')[9, : , j, :], 0)
    ODflows_absolutenb = (simulation_households_noct[np.newaxis, :] * ODflows_noct)
    modalShares_absolutenb = ODflows_absolutenb[:, :, np.newaxis] * modalShares_noct 
    modal_shares_noct[j, :] = (np.nansum(np.nansum(modalShares_absolutenb, 0), 0))
    for i in np.arange(5):
        dist_noct[j, i] = (np.nansum(transport_times["distanceCar"] * modalShares_absolutenb[:, :, i]) / np.nansum(modalShares_absolutenb[:, :, i]))
    

for j in np.arange(4):
    modalShares_noct = np.load(path_transportation_costs + "/modalShares_9.npy")[:, :, :, j]
    ODflows_noct = np.load(path_transportation_costs + "/ODflows_9.npy")[:, :, j]
    simulation_households_noct = np.nansum(np.load(path_scenario + '/simulation_households.npy')[9, : , j, :], 0)
    ODflows_absolutenb = (simulation_households_noct[np.newaxis, :] * ODflows_noct)
    modalShares_absolutenb = ODflows_absolutenb[:, :, np.newaxis] * modalShares_noct
    modal_shares_ct[j, :] = (np.nansum(np.nansum(modalShares_absolutenb, 0), 0))
    for i in np.arange(5):
        dist_ct[j, i] = (np.nansum(transport_times["distanceCar"] * modalShares_absolutenb[:, :, i]) / np.nansum(modalShares_absolutenb[:, :, i]))
    
distance_noct = np.empty(5)
for i in np.arange(5):
    distance_noct[i] = np.average(dist_noct[:, i], weights = modal_shares_noct[:, i])
distance_ct = np.empty(5)
for i in np.arange(5):
    distance_ct[i] = np.average(dist_ct[:, i], weights = modal_shares_ct[:, i])

diff_dist = 100 * (distance_ct - distance_noct) / distance_noct
diff_mode = 100 * (np.nansum(modal_shares_ct, 0) - np.nansum(modal_shares_noct, 0)) / np.nansum(modal_shares_noct, 0)

(100 + diff_dist) * (100 + diff_mode) / 10000

#mean distance

#Average distance to employment centers
#by income class
simulation_households = np.nansum(np.load(path_BAU + '/simulation_households.npy')[0, : , :, :], 0)
transport_times = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/Basile data/Transport_times_GRID.mat')
transport_times = transport_times["distanceCar"]
ODflows_noct = np.load(path_transportation_costs_BAU + "ODflows_0.npy")
ODflows_noct = ODflows_noct.transpose() * simulation_households[:, :, np.newaxis]
for j in np.arange(4):
    print(np.average(transport_times.transpose()[~np.isnan(transport_times.transpose())], weights = ODflows_noct[j, :, :][~np.isnan(transport_times.transpose())]))

#par housing type
simulation_households = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[0, : , :, :]
transport_times = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/Basile data/Transport_times_GRID.mat')
transport_times = transport_times["distanceCar"]
ODflows_noct = np.load(path_transportation_costs_BAU + "ODflows_0.npy")
hh = ODflows_noct[:, :, :, np.newaxis] * simulation_households.transpose()[np.newaxis, :, :, :]
hh = np.nansum(hh, 2)
for j in np.arange(4):
    print(np.average(transport_times[~np.isnan(transport_times)], weights = hh[:, :, j][~np.isnan(transport_times)]))


#### GINI Income spatial ineq

def gini(income='x', weights='w', variable_sort = 's', data=None):

    data = data[[income, weights, variable_sort]].sort_values(variable_sort, ascending=False).copy()
    x = data[income]
    f_x = data[weights] / data[weights].sum()
    F_x = f_x.cumsum()
    mu = np.sum(x * f_x)
    cov = np.cov(x, F_x, rowvar=False, aweights=f_x)[0,1]
    g = 2 * cov / mu
    return g


#Par income class

simulation_utility_noct = np.load(path_BAU + '/simulation_utility.npy')[9, :]
simulation_utility_ct = np.load(path_scenario + '/simulation_utility.npy')[9, :]

simulation_households_ct = np.nansum(np.nansum(np.load(path_scenario + '/simulation_households.npy')[9, : , :, :], 2), 0)
simulation_households_noct = np.nansum(np.nansum(np.load(path_BAU + '/simulation_households.npy')[9, : , :, :], 2), 0)
    
initial_state = pd.DataFrame([utility_income_class_init, simulation_households_noct, [1, 2, 3, 4]]).transpose()
initial_state.columns = ['utility', 'nb_hh', 'ordre']

step1 = pd.DataFrame([utility_income_class_change_tcost, simulation_households_noct, [1, 2, 3, 4]]).transpose()
step1.columns = ['utility', 'nb_hh', 'ordre']

step2 = pd.DataFrame([utility_income_class_change_tcostv2, simulation_households_noct, [1, 2, 3, 4]]).transpose()
step2.columns = ['utility', 'nb_hh', 'ordre']

step3 = pd.DataFrame([utility_income_class_change_rents, simulation_households_noct, [1, 2, 3, 4]]).transpose()
step3.columns = ['utility', 'nb_hh', 'ordre']

step3b = pd.DataFrame([utility_income_class_change_dsize, simulation_households_noct, [1, 2, 3, 4]]).transpose()
step3b.columns = ['utility', 'nb_hh', 'ordre']

final_state = pd.DataFrame([utility_income_class_final, simulation_households_ct, [1, 2, 3, 4]]).transpose()
final_state.columns = ['utility', 'nb_hh', 'ordre']


print(gini('utility', 'nb_hh', 'ordre', initial_state))
print(gini('utility', 'nb_hh', 'ordre', step1))
print(gini('utility', 'nb_hh', 'ordre', step2))
print(gini('utility', 'nb_hh', 'ordre', step3))
#print(gini('utility', 'nb_hh', 'ordre', step3b))
print(gini('utility', 'nb_hh', 'ordre', final_state))

gini_array = [gini('utility', 'nb_hh', 'ordre', initial_state),
              gini('utility', 'nb_hh', 'ordre', step1),
              gini('utility', 'nb_hh', 'ordre', step2),
              gini('utility', 'nb_hh', 'ordre', step3),
              #print(gini('utility', 'nb_hh', 'ordre', step3b))
              gini('utility', 'nb_hh', 'ordre', final_state)
              ]

plt.plot([1, 2, 3, 4, 5],gini_array)
plt.show()


#Spatialement #372661
simulation_households_noct = np.load(path_BAU + '/simulation_households.npy')[9, : , :, :]
simulation_households_ct = np.load(path_scenario + '/simulation_households.npy')[9, : , :, :]

j = 3

util = np.repeat(np.load(path_BAU + '/simulation_utility.npy')[9, :][:, np.newaxis], 24014, axis = 1)
util2 = np.repeat(util[np.newaxis, :, :], 4, axis = 0)

util3 = np.repeat(np.load(path_scenario + '/simulation_utility.npy')[9, :][:, np.newaxis], 24014, axis = 1)
util4 = np.repeat(util[np.newaxis, :, :], 4, axis = 0)


simulation_households_noct[np.abs((array_utility_init - util2)/util2)>0.02] =0 #tot: 1044359
simulation_households_ct[np.abs((array_utility_final_state - util4)/util4)>0.02] =0 #tot: 1044847

simulation_households_noct[:, 0:3, :] = 0 #243942, 222910, 388233, 208941
simulation_households_ct[:, 0:3, :] = 0



simulation_households_noct[np.isnan(array_utility_init)] = 0
array_utility_init[np.isnan(array_utility_init)] = 80000000000000000000

simulation_households_noct[np.isnan(array_utility_step1)] = 0
array_utility_step1[np.isnan(array_utility_step1)] = 80000000000000000000

simulation_households_noct[np.isnan(array_utility_step2)] = 0
array_utility_step2[np.isnan(array_utility_step2)] = 80000000000000000000

simulation_households_noct[np.isnan(array_utility_step3)] = 0
array_utility_step3[np.isnan(array_utility_step3)] = 80000000000000000000

simulation_households_noct[np.isnan(array_utility_step3b)] = 0
array_utility_step3b[np.isnan(array_utility_step3b)] = 80000000000000000000

simulation_households_ct[np.isnan(array_utility_final_state)] = 0
array_utility_final_state[np.isnan(array_utility_final_state)] = 80000000000000000000


weight_spatial_noct = np.nansum(np.nansum(simulation_households_noct, 0), 0)
weight_spatial_ct = np.nansum(np.nansum(simulation_households_ct, 0), 0)
dist_center = grid.dist

weighted_array_utility_init = np.nansum(np.nansum(array_utility_init * simulation_households_noct, 0), 0) / weight_spatial_noct
weighted_array_utility_step1 = np.nansum(np.nansum(array_utility_step1 * simulation_households_noct, 0), 0) / weight_spatial_noct
weighted_array_utility_step2 = np.nansum(np.nansum(array_utility_step2 * simulation_households_noct, 0), 0) / weight_spatial_noct
weighted_array_utility_step3 = np.nansum(np.nansum(array_utility_step3 * simulation_households_noct, 0), 0) / weight_spatial_noct
weighted_array_utility_step3b = np.nansum(np.nansum(array_utility_step3b * simulation_households_noct, 0), 0) / weight_spatial_noct
weighted_array_utility_final = np.nansum(np.nansum(array_utility_final_state * simulation_households_ct, 0), 0) / weight_spatial_ct

initial_state_spatial = pd.DataFrame([weighted_array_utility_init, weight_spatial_noct, dist_center]).transpose()
initial_state_spatial.columns = ['utility', 'nb_hh', 'ordre']

step1_spatial = pd.DataFrame([weighted_array_utility_step1, weight_spatial_noct, dist_center]).transpose()
step1_spatial.columns = ['utility', 'nb_hh', 'ordre']

step2_spatial = pd.DataFrame([weighted_array_utility_step2, weight_spatial_noct, dist_center]).transpose()
step2_spatial.columns = ['utility', 'nb_hh', 'ordre']

step3_spatial = pd.DataFrame([weighted_array_utility_step3, weight_spatial_noct, dist_center]).transpose()
step3_spatial.columns = ['utility', 'nb_hh', 'ordre']

step3b_spatial = pd.DataFrame([weighted_array_utility_step3b, weight_spatial_noct, dist_center]).transpose()
step3b_spatial.columns = ['utility', 'nb_hh', 'ordre']

final_state_spatial = pd.DataFrame([weighted_array_utility_final, weight_spatial_ct, dist_center]).transpose()
final_state_spatial.columns = ['utility', 'nb_hh', 'ordre']

print(gini('utility', 'nb_hh', 'ordre', initial_state_spatial[~np.isnan(initial_state_spatial.utility)]))
print(gini('utility', 'nb_hh', 'ordre', step1_spatial[~np.isnan(step1_spatial.utility)]))
print(gini('utility', 'nb_hh', 'ordre', step2_spatial[~np.isnan(step2_spatial.utility)]))
print(gini('utility', 'nb_hh', 'ordre', step3_spatial[~np.isnan(step3_spatial.utility)]))
#print(gini('utility', 'nb_hh', 'ordre', step3b_spatial[~np.isnan(step3b_spatial.utility)]))
print(gini('utility', 'nb_hh', 'ordre', final_state_spatial[~np.isnan(final_state_spatial.utility)]))

gini_array = [gini('utility', 'nb_hh', 'ordre', initial_state_spatial[~np.isnan(initial_state_spatial.utility)]),
              gini('utility', 'nb_hh', 'ordre', step1_spatial[~np.isnan(step1_spatial.utility)]),
              gini('utility', 'nb_hh', 'ordre', step2_spatial[~np.isnan(step2_spatial.utility)]),
              gini('utility', 'nb_hh', 'ordre', step3_spatial[~np.isnan(step3_spatial.utility)]),
              #print(gini('utility', 'nb_hh', 'ordre', step3b_spatial[~np.isnan(step3b_spatial.utility)]))
              gini('utility', 'nb_hh', 'ordre', final_state_spatial[~np.isnan(final_state_spatial.utility)])
              ]

plt.plot([1, 2, 3, 4, 5],gini_array)
plt.show()



gini_array1 = gini_array
gini_array2 = gini_array
gini_array3 = gini_array
gini_array4 = gini_array

plt.plot([1, 2, 3, 4, 5],gini_array1)
plt.plot([1, 2, 3, 4, 5],gini_array2)
plt.plot([1, 2, 3, 4, 5],gini_array3)
plt.plot([1, 2, 3, 4, 5],gini_array4)


def compute_transport_cost_mode(grid, param, yearTraffic, spline_inflation, spline_fuel, income_class, option_carbon_tax, option_scenario):
        """ Compute travel times and costs """

        #### STEP 1: IMPORT TRAVEL TIMES AND COSTS

        # Import travel times and distances
        transport_times = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/Basile data/Transport_times_GRID.mat')
             
        #Price per km
        priceTrainPerKMMonth = 0.164 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceTrainFixedMonth = 4.48 * 40 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceTaxiPerKMMonth = 0.785 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceTaxiFixedMonth = 4.32 * 40 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceBusPerKMMonth = 0.522 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceBusFixedMonth = 6.24 * 40 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        inflation = spline_inflation(yearTraffic)
        infla_2012 = spline_inflation(2012 - param["baseline_year"])
        priceTrainPerKMMonth = priceTrainPerKMMonth * inflation / infla_2012
        priceTrainFixedMonth = priceTrainFixedMonth * inflation / infla_2012
        priceTaxiPerKMMonth = priceTaxiPerKMMonth * inflation / infla_2012
        priceTaxiFixedMonth = priceTaxiFixedMonth * inflation / infla_2012
        priceBusPerKMMonth = priceBusPerKMMonth * inflation / infla_2012
        priceBusFixedMonth = priceBusFixedMonth * inflation / infla_2012
        priceFuelPerKMMonth = spline_fuel(yearTraffic)
        if (yearTraffic > 8) & (option_carbon_tax  == 1):
            priceFuelPerKMMonth = priceFuelPerKMMonth * 1.2
            if (option_scenario == 'scenario2'):
                priceBusPerKMMonth = priceBusPerKMMonth * 1.2
                priceTaxiPerKMMonth = priceTaxiPerKMMonth * 1.2
            
        #Fixed costs
        priceFixedVehiculeMonth = 400 
        priceFixedVehiculeMonth = priceFixedVehiculeMonth * inflation / infla_2012
        
        #### STEP 2: TRAVEL TIMES AND COSTS AS MATRIX
        
        #parameters
        numberDaysPerYear = 235
        numberHourWorkedPerDay= 8
        annualToHourly = 1 / (8*20*12)
        

        #Time by each mode, aller-retour, en minute
        timeOutput = np.empty((transport_times["durationTrain"].shape[0], transport_times["durationTrain"].shape[1], 5))
        timeOutput[:] = np.nan
        timeOutput[:,:,0] = transport_times["distanceCar"] / param["walking_speed"] * 60 * 1.2 * 2
        timeOutput[:,:,0][np.isnan(transport_times["durationCar"])] = np.nan
        timeOutput[:,:,1] = copy.deepcopy(transport_times["durationTrain"])
        timeOutput[:,:,2] = copy.deepcopy(transport_times["durationCar"])
        timeOutput[:,:,3] = copy.deepcopy(transport_times["durationMinibus"])
        timeOutput[:,:,4] = copy.deepcopy(transport_times["durationBus"])

        #Length (km) using each mode
        multiplierPrice = np.empty((timeOutput.shape))
        multiplierPrice[:] = np.nan
        multiplierPrice[:,:,0] = np.zeros((timeOutput[:,:,0].shape))
        multiplierPrice[:,:,1] = transport_times["distanceCar"]
        multiplierPrice[:,:,2] = transport_times["distanceCar"]
        multiplierPrice[:,:,3] = transport_times["distanceCar"]
        multiplierPrice[:,:,4] = transport_times["distanceCar"]

        #Multiplying by 235 (days per year)
        pricePerKM = np.empty(5)
        pricePerKM[:] = np.nan
        pricePerKM[0] = np.zeros(1)
        pricePerKM[1] = priceTrainPerKMMonth*numberDaysPerYear
        pricePerKM[2] = priceFuelPerKMMonth*numberDaysPerYear          
        pricePerKM[3] = priceTaxiPerKMMonth*numberDaysPerYear
        pricePerKM[4] = priceBusPerKMMonth*numberDaysPerYear
        
        #Distances (not useful to calculate price but useful output)
        distanceOutput = np.empty((timeOutput.shape))
        distanceOutput[:] = np.nan
        distanceOutput[:,:,0] = transport_times["distanceCar"]
        distanceOutput[:,:,1] = transport_times["distanceCar"]
        distanceOutput[:,:,2] = transport_times["distanceCar"]
        distanceOutput[:,:,3] = transport_times["distanceCar"]
        distanceOutput[:,:,4] = transport_times["distanceCar"]

        #Monetary price per year
        monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
        trans_monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
        for index2 in range(0, 5):
            monetaryCost[:,:,index2] = pricePerKM[index2] * multiplierPrice[:,:,index2]
        
        monetaryCost[:,:,1] = monetaryCost[:,:,1] + priceTrainFixedMonth * 12 #train (monthly fare)
        monetaryCost[:,:,2] = monetaryCost[:,:,2] + priceFixedVehiculeMonth * 12 #private car
        monetaryCost[:,:,3] = monetaryCost[:,:,3] + priceTaxiFixedMonth * 12 #minibus-taxi
        monetaryCost[:,:,4] = monetaryCost[:,:,4] + priceBusFixedMonth * 12 #bus
        trans_monetaryCost = copy.deepcopy(monetaryCost)

        #### STEP 3: COMPUTE PROBA TO WORK IN C, EXPECTED INCOME AND EXPECTED NB OF
        #RESIDENTS OF INCOME GROUP I WORKING IN C


        costTime = (timeOutput * param["time_cost"]) / (60 * numberHourWorkedPerDay) #en h de transport par h de travail
        costTime[np.isnan(costTime)] = 10 ** 2
        param_lambda = param["lambda"].squeeze()

        incomeNetOfCommuting = np.zeros((param["nb_of_income_classes"], transport_times["durationCar"].shape[1]))
        averageIncome = np.zeros((param["nb_of_income_classes"], transport_times["durationCar"].shape[1]))
        modalShares = np.zeros((185, transport_times["durationCar"].shape[1], 5, param["nb_of_income_classes"]))
        ODflows = np.zeros((185, transport_times["durationCar"].shape[1], param["nb_of_income_classes"]))
       
        #income
        incomeGroup, households_per_income_class = compute_average_income(spline_population_income_distribution, spline_income_distribution, param, yearTraffic)
        #income in 2011
        households_per_income_class, incomeGroupRef = import_income_classes_data(param, income_2011)
        #income centers
        income_centers_init = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/0. Precalculated inputs/incomeCentersKeep.mat')['incomeCentersKeep']
        incomeCenters = income_centers_init * incomeGroup / incomeGroupRef
    
        #switch to hourly
        monetaryCost = trans_monetaryCost * annualToHourly #en coût par heure
        monetaryCost[np.isnan(monetaryCost)] = 10**3 * annualToHourly
        incomeCenters = incomeCenters * annualToHourly
        
        xInterp = grid.x
        yInterp = grid.y
        
        householdSize = param["household_size"][income_class]
        #whichCenters = incomeCenters[:,income_class] > -100000
        #print(sum(whichCenters))
        incomeCentersGroup = incomeCenters[:, income_class]
        transportCostModes = (householdSize * monetaryCost[:,:,:] + (costTime[:,:,:] * incomeCentersGroup[:, None, None]))
        return transportCostModes
