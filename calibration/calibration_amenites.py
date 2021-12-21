# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:29:30 2020

@author: Charlotte Liotta
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:40:44 2020

@author: Charlotte Liotta
"""

import numpy as np
import scipy.io
import pandas as pd
import numpy.matlib

from inputs.data import *
from inputs.parameters_and_options import *
from equilibrium.compute_equilibrium import *
from outputs.export_outputs import *
from outputs.export_outputs_floods import *
from equilibrium.functions_dynamic import *
from equilibrium.run_simulations import *
from inputs.WBUS2_depth import *

print('**************** NEDUM-Cape-Town - Calibration of the informal housing amenity parameters ****************')

# %% Choose parameters and options

print("\n*** Load parameters and options ***\n")

#IMPORT PARAMETERS AND OPTIONS

options = import_options()
param = import_param(options)
t = np.arange(0, 1)

#OPTIONS FOR THIS SIMULATION

options["WBUS2"] = 1

# %% Load data

print("\n*** Load data ***\n")

#DATA

#Grid
grid, center = import_grid()
amenities = import_amenities()

#Households data
income_class_by_housing_type = import_hypothesis_housing_type()
income_2011 = pd.read_csv('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Basile data/Income_distribution_2011.csv')
mean_income = np.sum(income_2011.Households_nb * income_2011.INC_med) / sum(income_2011.Households_nb)
households_per_income_class, average_income = import_income_classes_data(param, income_2011)
income_mult = average_income / mean_income
income_net_of_commuting_costs = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/year_0.npy")
param["income_year_reference"] = mean_income
data_rdp, housing_types_sp, housing_types_grid, data_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, threshold_income_distribution, income_distribution, cape_town_limits = import_households_data(options)

#Macro data
interest_rate, population = import_macro_data(param)

#Land-use   
options["urban_edge"] = 1
spline_estimate_RDP, spline_land_backyard, spline_land_RDP, spline_RDP, spline_land_constraints, informal, coeff_land_backyard = import_land_use(grid, options, param, data_rdp, housing_types_grid)
number_properties_RDP = spline_estimate_RDP(0)
total_RDP = spline_RDP(0)
coeff_land = import_coeff_land(spline_land_constraints, spline_land_backyard, informal, spline_land_RDP, param, 0)
housing_limit = import_housig_limit(grid, param)
param = import_construction_parameters(param, grid, housing_types_sp, data_sp["dwelling_size"], mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land)
minimum_housing_supply = param["minimum_housing_supply"]
agricultural_rent = param["agricultural_rent_2011"] ** (param["coeff_a"]) * (param["depreciation_rate"] + interest_rate) / (param["coeff_A"] * param["coeff_b"] ** param["coeff_b"])

#FLOOD DATA
param = infer_WBUS2_depth(housing_types_grid, param)
if options["agents_anticipate_floods"] == 1:
    fraction_capital_destroyed, depth_damage_function_structure, depth_damage_function_contents = import_floods_data(options, param)
elif options["agents_anticipate_floods"] == 0:
    fraction_capital_destroyed = pd.DataFrame()
    fraction_capital_destroyed["structure"] = np.zeros(24014)
    fraction_capital_destroyed["contents"] = np.zeros(24014)
    
# %% Run initial state for several values of amenities

#Population
population = 1055925
total_RDP = 194258

#List of parameters
list_amenity_backyard = np.arange(0.65, 0.85, 0.01)
list_amenity_settlement = np.arange(0.65, 0.85, 0.01)
housing_type_total = pd.DataFrame(np.array(np.meshgrid(list_amenity_backyard, list_amenity_settlement)).T.reshape(-1,2))
housing_type_total.columns = ["param_backyard", "param_settlement"]
housing_type_total["formal"] = np.zeros(len(housing_type_total.param_backyard))
housing_type_total["backyard"] = np.zeros(len(housing_type_total.param_backyard))
housing_type_total["informal"] = np.zeros(len(housing_type_total.param_backyard))
housing_type_total["subsidized"] = np.zeros(len(housing_type_total.param_backyard))

sum_housing_types = lambda initial_state_households_housing_types : np.nansum(initial_state_households_housing_types, 1)
for i in range(17, len(list_amenity_backyard)):
    for j in range(0, len(list_amenity_settlement)):
        param["amenity_backyard"] = list_amenity_backyard[i]
        param["amenity_settlement"] = list_amenity_settlement[j]
        param["pockets"] = np.ones(24014) * param["amenity_settlement"]
        param["backyard_pockets"] = np.ones(24014) * param["amenity_backyard"]
        initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])
        housing_type_total.loc[(housing_type_total.param_backyard == param["amenity_backyard"]) & (housing_type_total.param_settlement == param["amenity_settlement"]), 2:6] = sum_housing_types(initial_state_households_housing_types)
        

print('*** End of simulations for chosen parameters ***')

# %% Pick best solution

housing_type_data = np.array([626770, 91132, 143765, 194258])

distance_share = np.abs(housing_type_total.iloc[:, 2:5] - housing_type_data[None, 0:3])
distance_share_score = distance_share.iloc[:,1] + distance_share.iloc[:,2] #+  distance_share.iloc[:,0]
which = np.argmin(distance_share_score)
min_score = np.nanmin(distance_share_score)
calibrated_amenities = housing_type_total.iloc[which, 0:2]

param["amenity_backyard"] = calibrated_amenities[0]
param["amenity_settlement"] = calibrated_amenities[1]

#84 et 81 : 26240

# %% IS Pockets

pockets = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/is_pockets.xlsx')
param["pockets"] = np.zeros(24014) + param["amenity_settlement"]
index = 0
index_max = 12
initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])
metrics = np.zeros(index_max)
save_param = np.zeros((index_max, 24014))
pop_data = ((96300/sum(housing_types_grid.informal_grid_2011)) * (housing_types_grid.informal_grid_2011))

for index in range(0, index_max):
    diff = np.zeros(len(np.unique(pockets.IS_Pocket_[~np.isnan(pockets.IS_Pocket_)])))
    for i in range(0, len(np.unique(pockets.IS_Pocket_[~np.isnan(pockets.IS_Pocket_)]))):
        pocket = int(np.unique(pockets.IS_Pocket_[~np.isnan(pockets.IS_Pocket_)])[i])
        diff[i] = (sum(pop_data[pockets.IS_Pocket_ == pocket]) - sum(initial_state_households_housing_types[2,:][pockets.IS_Pocket_ == pocket]))
        adj = (diff[i] / 10000)
        save_param[index, :] = param["pockets"]
        param["pockets"][pockets.IS_Pocket_ == pocket] = param["pockets"][pockets.IS_Pocket_ == pocket] + adj
    metrics[index] = sum(np.abs(diff))
    param["pockets"][param["pockets"] < 0.05] = 0.05
    param["pockets"][param["pockets"] > 0.99] = 0.99    
    initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])

index_min = np.argmin(metrics)
metrics[index_min]
param["pockets"] = save_param[index_min]
param["pockets"][param["pockets"] < 0.05] = 0.05
param["pockets"][param["pockets"] > 0.99] = 0.99 
initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])

os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name)
validation_density_housing_types(grid,initial_state_households_housing_types, housing_types_grid, name, 1)


# %% IS Pockets

#solver
index = 0
index_max = 30
initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])
metrics = np.zeros(index_max)

#pockets
pockets = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/is_pockets.xlsx')
param["pockets"] = np.zeros(24014) + param["amenity_settlement"]
save_param_informal_settlements = np.zeros((index_max, 24014))
pop_data_is = ((96300/sum(housing_types_grid.informal_grid_2011)) * (housing_types_grid.informal_grid_2011))
metrics_is = np.zeros(index_max)

#backyards
param["backyard_pockets"] = np.zeros(24014) + param["amenity_backyard"]
save_param_backyards = np.zeros((index_max, 24014))
pop_data_ib = ((85600/sum(housing_types_grid.backyard_grid_2011)) * (housing_types_grid.backyard_grid_2011))
metrics_ib = np.zeros(index_max)

for index in range(0, index_max):
    
    print("******************************* NEW ITERATION **************************************")
    print("INDEX  = " + str(index))
    
    #IS
    diff_is = np.zeros(len(np.unique(pockets.IS_Pocket_[~np.isnan(pockets.IS_Pocket_)])))
    for i in range(0, len(np.unique(pockets.IS_Pocket_[~np.isnan(pockets.IS_Pocket_)]))):
        pocket = int(np.unique(pockets.IS_Pocket_[~np.isnan(pockets.IS_Pocket_)])[i])
        diff_is[i] = (sum(pop_data_is[pockets.IS_Pocket_ == pocket]) - sum(initial_state_households_housing_types[2,:][pockets.IS_Pocket_ == pocket])) / sum(pop_data_is[pockets.IS_Pocket_ == pocket])
        adj = (diff_is[i] / (100))
        save_param_informal_settlements[index, :] = param["pockets"]
        param["pockets"][pockets.IS_Pocket_ == pocket] = param["pockets"][pockets.IS_Pocket_ == pocket] + adj
    metrics_is[index] = np.nansum(np.abs(diff_is))
    param["pockets"][param["pockets"] < 0.05] = 0.05
    param["pockets"][param["pockets"] > 0.99] = 0.99
    
    #IS
    #diff_is = np.zeros(166)
    #for i in range(0, 166):
    #    diff_is[i] = (sum(pop_data_is[(grid.dist > i/2) & (grid.dist < i/2 + 0.5)]) - sum(initial_state_households_housing_types[2,:][(grid.dist > i/2) & (grid.dist < i/2 + 0.5)]))
    #    adj = (diff_is[i] / (1000000))
    #    save_param_informal_settlements[index, :] = param["pockets"]
    #    param["pockets"][(grid.dist > i/2) & (grid.dist < i/2 + 0.5)] = param["pockets"][(grid.dist > i/2) & (grid.dist < i/2 + 0.5)] + adj
    #metrics_is[index] = sum(np.abs(diff_is))
    #param["pockets"][param["pockets"] < 0.05] = 0.05
    #param["pockets"][param["pockets"] > 0.99] = 0.99
    
    #IS
    #diff_is = np.zeros(24014)
    #for i in range(0, 24014):
    #    print(i)
    #    diff_is[i] = (pop_data_is[i]) - (initial_state_households_housing_types[2,:][i])
    #    adj = (diff_is[i] / (200000))
    #    save_param_informal_settlements[index, :] = param["pockets"]
    #    param["pockets"][i] = param["pockets"][i] + adj
    #metrics_is[index] = sum(np.abs(diff_is))
    #param["pockets"][param["pockets"] < 0.05] = 0.05
    #param["pockets"][param["pockets"] > 0.99] = 0.99
    
    #IB
    diff_ib = np.zeros(166)
    for i in range(0, 166):
        diff_ib[i] = (sum(pop_data_ib[(grid.dist > i/2) & (grid.dist < i/2 + 0.5)]) - sum(initial_state_households_housing_types[1,:][(grid.dist > i/2) & (grid.dist < i/2 + 0.5)]))
        adj = (diff_ib[i] / 100000)
        save_param_backyards[index, :] = param["backyard_pockets"]
        param["backyard_pockets"][(grid.dist > i/2) & (grid.dist < i/2 + 0.5)] = param["backyard_pockets"][(grid.dist > i/2) & (grid.dist < i/2 + 0.5)] + adj
    metrics_ib[index] = sum(np.abs(diff_ib))
    param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
    param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99
    
    #IB
    #diff_ib = np.zeros(24014)
    #for i in range(0, 24014):
    #    diff_ib[i] = ((pop_data_ib[i]) - (initial_state_households_housing_types[1,:][i]))
    #    adj = (diff_ib[i] / 200000)
    #    save_param_backyards[index, :] = param["backyard_pockets"]
    #    param["backyard_pockets"][i] = param["backyard_pockets"][i] + adj
    #metrics_ib[index] = sum(np.abs(diff_ib))
    #param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
    #param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99
    
    metrics[index] = metrics_is[index] + metrics_ib[index]
    
    initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])

index_min = np.argmin(metrics_is)
metrics[index_min]
param["pockets"] = save_param_informal_settlements[index_min]
param["pockets"][param["pockets"] < 0.05] = 0.05
param["pockets"][param["pockets"] > 0.99] = 0.99 
param["backyard_pockets"] = save_param_backyards[index_min]
param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99
initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])

name = '200pc_dist_pockets'
os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name)
validation_density_housing_types(grid,initial_state_households_housing_types, housing_types_grid, name, 1)

floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"
count_formal = housing_types_grid.formal_grid_2011 - data_rdp["count"]
count_formal[count_formal < 0] = 0
stats_per_housing_type_data = compute_stats_per_housing_type(floods, path_data, count_formal, data_rdp["count"], housing_types_grid.informal_grid_2011, housing_types_grid.backyard_grid_2011, options, param)
stats_per_housing_type_simul = compute_stats_per_housing_type(floods, path_data, initial_state_households_housing_types[0, :], initial_state_households_housing_types[3, :], initial_state_households_housing_types[2, :], initial_state_households_housing_types[1, :], options, param)
validation_flood(name, stats_per_housing_type_data, stats_per_housing_type_simul, 'Data', 'Simul')

plt.plot(grid.dist[~np.isnan(pockets.IS_Pocket_)], param["pockets"][~np.isnan(pockets.IS_Pocket_)], 'o')
m, b = np.polyfit(grid.dist[~np.isnan(pockets.IS_Pocket_)], param["pockets"][~np.isnan(pockets.IS_Pocket_)], 1)
plt.plot(grid.dist, m*grid.dist + b)

plt.plot(grid.dist, param["backyard_pockets"], 'o')
m, b = np.polyfit(grid.dist, param["backyard_pockets"], 1)
plt.plot(grid.dist, m*grid.dist + b)

np.save('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + 'param_pockets.npy', save_param_informal_settlements[index_min])
np.save('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + 'param_backyards.npy', save_param_backyards[index_min])