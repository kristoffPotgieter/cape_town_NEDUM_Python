# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:33:37 2020

@author: Charlotte Liotta
"""

print("\n*** NEDUM-Cape-Town - Floods modelling ***\n")

import numpy as np
import scipy.io
import pandas as pd
import numpy.matlib
import seaborn as sns
import time

from inputs.data import *
from inputs.parameters_and_options import *
from equilibrium.compute_equilibrium import *
from outputs.export_outputs import *
from outputs.export_outputs_floods import *
from outputs.flood_outputs import *
from equilibrium.functions_dynamic import *
from equilibrium.run_simulations import *
from inputs.WBUS2_depth import *


path_code = '..'
path_folder = path_code + '/2. Data/'
precalculated_inputs_path = path_folder + "0. Precalculated inputs/"
path_data = path_folder + "data_Cape_Town/"
precalculated_transport = path_folder + "precalculated_transport/"
path_scenarios = path_folder + 'data_Cape_Town/Scenarios/'
path_outputs = path_code+'/4. Sorties/'

start = time.process_time()
# %% Import parameters and options

print("\n*** Load parameters and options ***\n")

#IMPORT PARAMETERS AND OPTIONS
options = import_options()
param = import_param(options["import_precalculated_parameters"], precalculated_inputs_path)
t = np.arange(0, 1)

#PARAMETERS COMING FROM LOCATION-BASED CALIBRATION
if options["pluvial"] == 0:
    param["pockets"] = np.load(path_outputs+'fluvial_and_pluvial/param_pockets.npy')
    param["backyard_pockets"] = np.load(path_outputs+'fluvial_and_pluvial/param_backyards.npy')

param["pockets"] = np.load(path_outputs+'fluvial_and_pluvial/param_pockets.npy')
param["backyard_pockets"] = np.load(path_outputs+'fluvial_and_pluvial/param_backyards.npy')

 
#NAME OF THE SIMULATION - TO EXPORT THE RESULTS
date = 'no_floods_scenario'
name = date + '_' + str(options["pluvial"]) + '_' + str(options["informal_land_constrained"])

# %% Load data

print("\n*** Load data ***\n")

#DATA

#Grid
grid, center = import_grid(path_data) # analysis grid
amenities = import_amenities(precalculated_inputs_path) # cf. WB working paper for more explanations

#Households and income data
income_class_by_housing_type = import_hypothesis_housing_type()
income_2011 = pd.read_csv(path_data + 'Income_distribution_2011.csv')
mean_income = np.sum(income_2011.Households_nb * income_2011.INC_med) / sum(income_2011.Households_nb)
households_per_income_class, average_income = import_income_classes_data(param, income_2011)
income_mult = average_income / mean_income
income_net_of_commuting_costs = np.load(precalculated_transport + 'incomeNetOfCommuting_0.npy')
param["income_year_reference"] = mean_income
data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, threshold_income_distribution, income_distribution, cape_town_limits = import_households_data(options, precalculated_inputs_path)
housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')
housing_types[np.isnan(housing_types)] = 0

#Macro data
interest_rate, population = import_macro_data(param, path_scenarios)
total_RDP = 194258 #RDP = social housing
total_formal = 626770
total_informal = 143765
total_backyard = 91132

housing_type_data = np.array([total_formal, total_backyard, total_informal, total_RDP]) #taken back from old_code_calibration

#Land-use   
#options["urban_edge"] = 1
spline_estimate_RDP, spline_land_backyard, spline_land_RDP, spline_RDP, spline_land_constraints, spline_land_informal, coeff_land_backyard = import_land_use(grid, options, param, data_rdp, housing_types, total_RDP, path_data, path_folder)
number_properties_RDP = spline_estimate_RDP(0)
coeff_land = import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 0)
param["pockets"][(spline_land_informal(29) > 0) & (spline_land_informal(0) == 0)] = 0.79

housing_limit = import_housig_limit(grid, param)
param = import_construction_parameters(param, grid, housing_types_sp, data_sp["dwelling_size"], mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land)
minimum_housing_supply = param["minimum_housing_supply"]
agricultural_rent = param["agricultural_rent_2011"] ** (param["coeff_a"]) * (param["depreciation_rate"] + interest_rate) / (param["coeff_A"] * param["coeff_b"] ** param["coeff_b"])

#Scenarios

(spline_agricultural_rent, 
 spline_interest_rate, 
 spline_RDP, 
 spline_population_income_distribution, 
 spline_inflation, 
 spline_income_distribution, 
 spline_population, 
 spline_interest_rate, 
 spline_income, 
 spline_minimum_housing_supply, 
 spline_fuel) = import_scenarios(income_2011, param, grid, path_scenarios) #we add required argument
#Most of the import is implicit in run_simulation, but we need to make this explicit for scenario plots


# %% Compute initial state

print("\n*** Solver initial state ***\n")
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



# %% Validation: draw maps and figures

#General validation
export_housing_types(initial_state_households_housing_types, 
                     initial_state_household_centers, 
                     housing_type_data, 
                     households_per_income_class, 
                     'Simulation', 
                     'Data',
                     path_outputs+name)

validation_density(grid, initial_state_households_housing_types, housing_types,path_outputs+name)
validation_density_housing_types(grid,initial_state_households_housing_types, housing_types, 0,path_outputs+name)
validation_housing_price(grid, initial_state_rent, interest_rate, param, center, precalculated_inputs_path,path_outputs+name)
#plot_diagnosis_map_informl(grid, coeff_land, initial_state_households_housing_types, name)

# %% Scenarios

#Compute scenarios
t = np.arange(0, 30)

#Add counterfactual options: here, we may want to consider flood damages and people in flood zones, while keeping housing choices independent of floods (?)
#But is this really working?
if options["agents_anticipate_floods"] == 0:
    fraction_capital_destroyed, *_ = import_floods_data(options, param, path_folder) #need to add parameters
    #fraction_capital_destroyed, content_damages, structural_damages_type4b, structural_damages_type4a, structural_damages_type2, structural_damages_type3a = import_floods_data(options, param, path_folder)


# important: does the simulation
(simulation_households_center, 
 simulation_households_housing_type, 
 simulation_dwelling_size, 
 simulation_rent, 
 simulation_households, 
 simulation_error, 
 simulation_housing_supply, 
 simulation_utility, 
 simulation_deriv_housing,
 simulation_T) = run_simulation(t, 
                                options, 
                                income_2011, 
                                param, 
                                grid, 
                                initial_state_utility, 
                                initial_state_error, 
                                initial_state_households, 
                                initial_state_households_housing_types, 
                                initial_state_housing_supply, 
                                initial_state_household_centers, 
                                initial_state_average_income, 
                                initial_state_rent, 
                                initial_state_dwelling_size, 
                                fraction_capital_destroyed, 
                                amenities, 
                                housing_limit, 
                                spline_estimate_RDP, 
                                spline_land_constraints, 
                                spline_land_backyard, 
                                spline_land_RDP, 
                                spline_land_informal, 
                                income_class_by_housing_type, 
                                path_scenarios, 
                                precalculated_transport)

#Save outputs
name = 'carbon_tax_car_bus_taxi_20211103_basile'

try:
    os.mkdir(path_outputs + name)
except OSError as error:
    print(error) 

np.save(path_outputs + name + '/simulation_households_center.npy', simulation_households_center)
np.save(path_outputs + name + '/simulation_dwelling_size.npy', simulation_dwelling_size)
np.save(path_outputs + name + '/simulation_rent.npy', simulation_rent)
np.save(path_outputs + name + '/simulation_households_housing_type.npy', simulation_households_housing_type)
np.save(path_outputs + name + '/simulation_households.npy', simulation_households)
np.save(path_outputs + name + '/simulation_utility.npy', simulation_utility)



