# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:33:37 2020.

@author: Charlotte Liotta
"""

# TO DO
# Change names and repo structure?
# Set baseline year as a dynamic parameter for loading updated data


# %% Preamble


# IMPORT PACKAGES

import numpy as np
import pandas as pd
import seaborn as sns
import time

import inputs.data as inpdt
import inputs.parameters_and_options as inpprm
import inputs.WBUS2_depth as inpfld

import equilibrium.compute_equilibrium as eqcmp
import equilibrium.functions_dynamic as eqdyn
import equilibrium.run_simulations as eqsim

import outputs.export_outputs as outexp
import outputs.export_outputs_floods as outexpfld
import outputs.flood_outputs as outfld

# import calibration.disamenity_param_calibration as calprm


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/2. Data/'
precalculated_inputs = path_folder + '0. Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
precalculated_transport = path_folder + 'precalculated_transport/'
path_scenarios = path_folder + 'data_Cape_Town/Scenarios/'
path_outputs = path_code + '/4. Sorties/'


# START TIMER FOR CODE OPTIMIZATION

start = time.process_time()


# %% Import parameters and options


# IMPORT DEFAULT PARAMETERS AND OPTIONS

options = inpprm.import_options()
param = inpprm.import_param(precalculated_inputs)
t = np.arange(0, 1)  # when is it used?


# GIVE NAME TO SIMULATION TO EXPORT THE RESULTS
# (change according to custom parameters to be included)

date = 'no_floods_scenario'
name = date + '_' + str(options["pluvial"]) + '_' + str(
    options["informal_land_constrained"])


# %% Load data


# BASIC GEOGRAPHIC DATA

grid, center = inpdt.import_grid(path_data)
amenities = inpdt.import_amenities(precalculated_inputs)


# HOUSEHOLDS AND INCOME DATA

income_class_by_housing_type = inpdt.import_hypothesis_housing_type()

(mean_income, households_per_income_class, average_income, income_mult
 ) = inpdt.import_income_classes_data(param, path_data)

#  Import income net of commuting costs, as calibrated in Pfeiffer et al.
#  (see part 3.1 or appendix C3)
income_net_of_commuting_costs = np.load(
    precalculated_transport + 'incomeNetOfCommuting_0.npy')

#  Is it useful? At least, it is not logical
#  param["income_year_reference"] = mean_income

(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(precalculated_inputs)

#  Import population density per pixel, by housing type
#  There is no RDP, but both formal and informal backyard???
housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')
# Replace missing values by zero
housing_types[np.isnan(housing_types)] = 0


# MACRO DATA

interest_rate, population, housing_type_data = inpdt.import_macro_data(
    param, path_scenarios)


# LAND USE

(spline_RDP, spline_estimate_RDP, spline_land_RDP, coeff_land_backyard,
 spline_land_backyard, spline_land_informal, spline_land_constraints) = (
     inpdt.import_land_use(grid, options, param, data_rdp, housing_types,
                           housing_type_data, path_data, path_folder)
     )

#  
number_properties_RDP = spline_estimate_RDP(0)
coeff_land = import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 0)
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

# IMPORT CUSTOM PARAMETERS AND OPTIONS

#  If want to update the parameters, need to uncomment the following command
#  (may take a full day to run) after initial state
#  calprm...

#  Disamenity parameters for informal settlements and backyard shacks,
#  coming from location-based calibration, as opposed to general calibration
#  used in Pfeiffer et al. (appendix C5)
param["pockets"] = np.load(
    path_outputs+'fluvial_and_pluvial/param_pockets.npy')
param["backyard_pockets"] = np.load(
    path_outputs+'fluvial_and_pluvial/param_backyards.npy')

param["pockets"][(spline_land_informal(29) > 0) & (spline_land_informal(0) == 0)] = 0.79


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



