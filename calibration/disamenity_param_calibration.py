# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:26:14 2022.

@author: vincentviguie
"""

import numpy as np
import pandas as pd
import numpy.matlib
import time
import datetime

# See Aux data/disamenity_workbook?

# %% Calibration

#  General calibration (see Pfeiffer et al., appendix C5)
list_amenity_backyard = np.arange(0.70, 0.90, 0.01)
list_amenity_settlement = np.arange(0.67, 0.87, 0.01)
housing_type_total = pd.DataFrame(np.array(np.meshgrid(
    list_amenity_backyard, list_amenity_settlement)).T.reshape(-1, 2))
housing_type_total.columns = ["param_backyard", "param_settlement"]
housing_type_total["formal"] = np.zeros(len(housing_type_total.param_backyard))
housing_type_total["backyard"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["informal"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["subsidized"] = np.zeros(
    len(housing_type_total.param_backyard))

def sum_housing_types():
    """bla."""

sum_housing_types = lambda initial_state_households_housing_types : np.nansum(initial_state_households_housing_types, 1)

debut_calib_time = time.process_time()
number_total_iterations=len(list_amenity_backyard)*len(list_amenity_settlement)
print(f"** Calibration: {number_total_iterations} iterations **")
for i in range(0, len(list_amenity_backyard)):
    for j in range(0, len(list_amenity_settlement)):
        param["amenity_backyard"] = list_amenity_backyard[i]
        param["amenity_settlement"] = list_amenity_settlement[j]
        param["pockets"] = np.ones(24014) * param["amenity_settlement"]
        param["backyard_pockets"] = np.ones(24014) * param["amenity_backyard"]
        (initial_state_utility, 
         initial_state_error, 
         initial_state_simulated_jobs, 
         initial_state_households_housing_types, 
         initial_state_household_centers, 
         initial_state_households, 
         initial_state_dwelling_size, 
         initial_state_housing_supply, 
         initial_state_rent, 
         initial_state_rent_matrix, 
         initial_state_capital_land, 
         initial_state_average_income, 
         initial_state_limit_city) = compute_equilibrium(
             fraction_capital_destroyed, 
             amenities, 
             param, 
             housing_limit, 
             population, 
             households_per_income_class, 
             total_RDP, 
             coeff_land, 
             income_net_of_commuting_costs, 
             grid, 
             options, 
             agricultural_rent, 
             interest_rate, 
             number_properties_RDP, 
             average_income, 
             mean_income, 
             income_class_by_housing_type, 
             minimum_housing_supply, 
             param["coeff_A"])
        housing_type_total.loc[
            (housing_type_total.param_backyard == param["amenity_backyard"]) 
            & (housing_type_total.param_settlement == param["amenity_settlement"]), 
            2:6] = sum_housing_types(initial_state_households_housing_types)
        
        time_elapsed=time.process_time() - debut_calib_time
        iteration_number= i*len(list_amenity_settlement)+j+1
        print(f"iteration {iteration_number}/{number_total_iterations} finished.",
              str(datetime.timedelta(seconds=round(time_elapsed))),f"elapsed ({round(time_elapsed/iteration_number)}s per iteration). There remains",
              str(datetime.timedelta(seconds=round(time_elapsed/iteration_number*(number_total_iterations-iteration_number))))
            )
    
housing_type_data = np.array([total_formal, total_backyard, total_informal, total_RDP])

distance_share = np.abs(housing_type_total.iloc[:, 2:5] - housing_type_data[None, 0:3])
distance_share_score = distance_share.iloc[:,1] + distance_share.iloc[:,2] +  distance_share.iloc[:,0]
which = np.argmin(distance_share_score)
min_score = np.nanmin(distance_share_score)
calibrated_amenities = housing_type_total.iloc[which, 0:2]

#0.88 et 0.85

param["amenity_backyard"] = calibrated_amenities[0]
param["amenity_settlement"] = calibrated_amenities[1]

param["amenity_backyard"] = 0.89
param["amenity_settlement"] = 0.86

#Location-based calibration
index = 0
index_max = 400
metrics = np.zeros(index_max)
param["pockets"] = np.zeros(24014) + param["amenity_settlement"]
save_param_informal_settlements = np.zeros((index_max, 24014))
metrics_is = np.zeros(index_max)
param["backyard_pockets"] = np.zeros(24014) + param["amenity_backyard"]
save_param_backyards = np.zeros((index_max, 24014))
metrics_ib = np.zeros(index_max)
print("\n* City limits *")
(initial_state_utility, 
 initial_state_error, 
 initial_state_simulated_jobs, 
 initial_state_households_housing_types, 
 initial_state_household_centers, 
 initial_state_households, 
 initial_state_dwelling_size, 
 initial_state_housing_supply, 
 initial_state_rent, 
 initial_state_rent_matrix, 
 initial_state_capital_land, 
 initial_state_average_income, 
 initial_state_limit_city
 ) = compute_equilibrium(
     fraction_capital_destroyed, 
     amenities, 
     param, 
     housing_limit, 
     population, 
     households_per_income_class, 
     total_RDP, 
     coeff_land, 
     income_net_of_commuting_costs, 
     grid, 
     options, 
     agricultural_rent, 
     interest_rate, 
     number_properties_RDP, 
     average_income, 
     mean_income, 
     income_class_by_housing_type, 
     minimum_housing_supply, 
     param["coeff_A"])

# %% Iterations
print("\n** ITERATIONS **")
debut_iterations_time=time.process_time()
number_total_iterations=index_max
for index in range(0, index_max):
    
    #IS
    diff_is = np.zeros(24014)
    for i in range(0, 24014):
        diff_is[i] = (housing_types.informal_grid[i]) - (initial_state_households_housing_types[2,:][i])
        adj = (diff_is[i] / (50000))
        save_param_informal_settlements[index, :] = param["pockets"]
        param["pockets"][i] = param["pockets"][i] + adj
    metrics_is[index] = sum(np.abs(diff_is))
    param["pockets"][param["pockets"] < 0.05] = 0.05
    param["pockets"][param["pockets"] > 0.99] = 0.99
    
    #IB
    diff_ib = np.zeros(24014)
    for i in range(0, 24014):
        diff_ib[i] = ((housing_types.backyard_informal_grid[i] + housing_types.backyard_formal_grid[i]) - (initial_state_households_housing_types[1,:][i]))
        adj = (diff_ib[i] / 50000)
        save_param_backyards[index, :] = param["backyard_pockets"]
        param["backyard_pockets"][i] = param["backyard_pockets"][i] + adj
    metrics_ib[index] = sum(np.abs(diff_ib))
    param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
    param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99
    
    metrics[index] = metrics_is[index] + metrics_ib[index]
    
    initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])
    
    time_elapsed=time.process_time() - debut_iterations_time
    iteration_number= index+1
    print(f"iteration {iteration_number}/{number_total_iterations} finished.",
              str(datetime.timedelta(seconds=round(time_elapsed))),f"elapsed ({round(time_elapsed/iteration_number)}s per iteration). There remains",
              str(datetime.timedelta(seconds=round(time_elapsed/iteration_number*(number_total_iterations-iteration_number))))
            )


index_min = np.argmin(metrics)
metrics[index_min]
param["pockets"] = save_param_informal_settlements[index_min]
param["pockets"][param["pockets"] < 0.05] = 0.05
param["pockets"][param["pockets"] > 0.99] = 0.99 
param["backyard_pockets"] = save_param_backyards[index_min]
param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99

os.mkdir(path_outputs+'' + name)
np.save(path_outputs+'' + name + '/param_pockets.npy', save_param_informal_settlements[index_min])
np.save(path_outputs+'' + name + '/param_backyards.npy', save_param_backyards[index_min])