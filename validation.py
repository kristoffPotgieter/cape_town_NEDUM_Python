# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 12:55:28 2022.

@author: monni
"""

# %% Preamble


# IMPORT PACKAGES

import numpy as np
import pandas as pd
import time
import math

import inputs.data as inpdt
import inputs.parameters_and_options as inpprm

import equilibrium.functions_dynamic as eqdyn


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/2. Data/'
path_precalc_inp = path_folder + '0. Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_folder + 'data_Cape_Town/Scenarios/'
path_outputs = path_code + '/4. Sorties/'
path_floods = path_folder + "FATHOM/"


# START TIMER FOR CODE OPTIMIZATION

start = time.process_time()


# %% Import parameters and options


# IMPORT DEFAULT PARAMETERS AND OPTIONS

options = inpprm.import_options()
param = inpprm.import_param(path_precalc_inp, path_outputs)

# GIVE NAME TO SIMULATION TO EXPORT THE RESULTS
# (change according to custom parameters to be included)

date = 'floods_scenario'
name = date + '_' + str(options["pluvial"]) + '_' + str(
    options["informal_land_constrained"])


# %% Load data


# BASIC GEOGRAPHIC DATA

grid, center = inpdt.import_grid(path_data)
amenities = inpdt.import_amenities(path_precalc_inp)


# MACRO DATA

(interest_rate, population, housing_type_data, total_RDP
 ) = inpdt.import_macro_data(param, path_scenarios)


# HOUSEHOLDS AND INCOME DATA

income_class_by_housing_type = inpdt.import_hypothesis_housing_type()

(mean_income, households_per_income_class, average_income, income_mult,
 income_2011) = inpdt.import_income_classes_data(param, path_data)

#  We create this parameter to maintain money illusion in simulations
#  (see eqsim.run_simulation)
param["income_year_reference"] = mean_income

(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(path_precalc_inp)

#  Import nb of households per pixel, by housing type
#  Note that there is no RDP, but both formal and informal backyard

# Long tu run: uncomment if need to update 'housing_types_grid_sal.xlsx'
# housing_types = inpdt.import_sal_data(grid, path_folder, path_data,
#                                       housing_type_data)
housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')


# LAND USE PROJECTIONS

(spline_RDP, spline_estimate_RDP, spline_land_RDP,
 spline_land_backyard, spline_land_informal, spline_land_constraints,
 number_properties_RDP) = (
     inpdt.import_land_use(grid, options, param, data_rdp, housing_types,
                           housing_type_data, path_data, path_folder)
     )

param["pockets"][
    (spline_land_informal(29) > 0) & (spline_land_informal(0) == 0)
    ] = 0.79

#  We correct areas for each housing type at baseline year for the amount of
#  constructible land in each type
coeff_land = inpdt.import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal,
    spline_land_RDP, param, 0)

#  We update land use parameters at baseline (relies on data)
housing_limit = inpdt.import_housing_limit(grid, param)

(param, minimum_housing_supply, agricultural_rent
 ) = inpprm.import_construction_parameters(
    param, grid, housing_types_sp, data_sp["dwelling_size"],
    mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land,
    interest_rate
    )

# FLOOD DATA (takes some time)
param = inpdt.infer_WBUS2_depth(housing_types, param, path_floods)
if options["agents_anticipate_floods"] == 1:
    (fraction_capital_destroyed, structural_damages_small_houses,
     structural_damages_medium_houses, structural_damages_large_houses,
     content_damages, structural_damages_type1, structural_damages_type2,
     structural_damages_type3a, structural_damages_type3b,
     structural_damages_type4a, structural_damages_type4b
     ) = inpdt.import_full_floods_data(options, param, path_folder,
                                       housing_type_data)
elif options["agents_anticipate_floods"] == 0:
    fraction_capital_destroyed = pd.DataFrame()
    fraction_capital_destroyed["structure_formal_2"] = np.zeros(24014)
    fraction_capital_destroyed["structure_formal_1"] = np.zeros(24014)
    fraction_capital_destroyed["structure_subsidized_2"] = np.zeros(24014)
    fraction_capital_destroyed["structure_subsidized_1"] = np.zeros(24014)
    fraction_capital_destroyed["contents_formal"] = np.zeros(24014)
    fraction_capital_destroyed["contents_informal"] = np.zeros(24014)
    fraction_capital_destroyed["contents_subsidized"] = np.zeros(24014)
    fraction_capital_destroyed["contents_backyard"] = np.zeros(24014)
    fraction_capital_destroyed["structure_backyards"] = np.zeros(24014)
    fraction_capital_destroyed["structure_informal_settlements"
                               ] = np.zeros(24014)

# SCENARIOS

(spline_agricultural_rent, spline_interest_rate,
 spline_population_income_distribution, spline_inflation,
 spline_income_distribution, spline_population,
 spline_income, spline_minimum_housing_supply, spline_fuel
 ) = eqdyn.import_scenarios(income_2011, param, grid, path_scenarios)

# Long to run: uncomment if need to update scenarios for transport data

# for t_temp in np.arange(0, 30):
#     print(t_temp)
#     (incomeNetOfCommuting, modalShares, ODflows, averageIncome
#      ) = inpdt.import_transport_data(
#          grid, param, t_temp, households_per_income_class, average_income,
#          spline_inflation, spline_fuel,
#          spline_population_income_distribution, spline_income_distribution,
#          path_precalc_inp, path_precalc_transp)

#  Import income net of commuting costs, as calibrated in Pfeiffer et al.
#  (see part 3.1 or appendix C3)
income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'incomeNetOfCommuting_0.npy')
# ODflows = np.load(
#     path_precalc_transp + 'ODflows_0.npy')
# averageIncome = np.load(
#     path_precalc_transp + 'averageIncome_0.npy')

#  Import equilibrium outputs

initial_state_utility = np.load(
    path_outputs + name + '/initial_state_utility.npy')
initial_state_error = np.load(
    path_outputs + name + '/initial_state_error.npy')
initial_state_simulated_jobs = np.load(
    path_outputs + name + '/initial_state_simulated_jobs.npy')
initial_state_households_housing_types = np.load(
    path_outputs + name + '/initial_state_households_housing_types.npy')
initial_state_household_centers = np.load(
    path_outputs + name + '/initial_state_household_centers.npy')
initial_state_households = np.load(
    path_outputs + name + '/initial_state_households.npy')
initial_state_dwelling_size = np.load(
    path_outputs + name + '/initial_state_dwelling_size.npy')
initial_state_housing_supply = np.load(
    path_outputs + name + '/initial_state_housing_supply.npy')
initial_state_rent = np.load(
    path_outputs + name + '/initial_state_rent.npy')
initial_state_rent_matrix = np.load(
    path_outputs + name + '/initial_state_rent_matrix.npy')
initial_state_capital_land = np.load(
    path_outputs + name + '/initial_state_capital_land.npy')
initial_state_average_income = np.load(
    path_outputs + name + '/initial_state_average_income.npy')
initial_state_limit_city = np.load(
    path_outputs + name + '/initial_state_limit_city.npy')


#  Import simulation outputs

simulation_households_center = np.load(
    path_outputs + name + '/simulation_households_center.npy')
simulation_households_housing_type = np.load(
    path_outputs + name + '/simulation_households_housing_type.npy')
simulation_dwelling_size = np.load(
    path_outputs + name + '/simulation_dwelling_size.npy')
simulation_rent = np.load(
    path_outputs + name + '/simulation_rent.npy')
simulation_households = np.load(
    path_outputs + name + '/simulation_households.npy')
simulation_error = np.load(
    path_outputs + name + '/simulation_error.npy')
simulation_housing_supply = np.load(
    path_outputs + name + '/simulation_housing_supply.npy')
simulation_utility = np.load(
    path_outputs + name + '/simulation_utility.npy')
simulation_deriv_housing = np.load(
    path_outputs + name + '/simulation_deriv_housing.npy')
simulation_T = np.load(
    path_outputs + name + '/simulation_T.npy')


# %% Figures from working paper (with floods and no actual backyards): no scenario

# %% Validation exercises

# Note that we only have the poorest income group in backyard and informal
# settlements: shouldn't we observe some from the second poorest as well?
# Without actual_backyards, we again find coherent distributions of backyards
# wrt RDP

total_households = np.nansum(initial_state_households, 2)

test = coeff_land[[1, 3], :]
test = test[:, test[0, :] > 0]
test = np.vstack([test, test[0, :] / test[1, :]])

# TODO: need to check calibration fit for average income?
# TODO: should we use ksi or household_size?

ksi = [size / 2 for size in param["household_size"]]
household_size = param["household_size"]

cal_average_income = np.nanmean(averageIncome, 1)
cal_average_wage = household_size * cal_average_income

# Note that equilibrium constraints are satisfied by definition
# TODO: What about condition (v)?
# TODO: do more validation exercises

W_mat = np.zeros((np.ma.size(ODflows, 0), param["nb_of_income_classes"]))

for i in range(np.ma.size(ODflows, 0) - 1):
    W_mat[i, :] = household_size * np.nansum(
        ODflows[i, :, :] * initial_state_household_centers.T, 0)

left = np.nansum(W_mat, 0)
right = np.nansum(household_size * initial_state_household_centers.T, 0)

# TODO: How to explain that there are backyarders where no RDP live?
# TODO: And how can suppliers host more than 5 people each?

backyard_hh = initial_state_households_housing_types[1, :]
rdp_hh = initial_state_households_housing_types[3, :]

rdp_hh = rdp_hh[backyard_hh > 0]
backyard_hh = backyard_hh[backyard_hh > 0]

backyard_mkt = np.stack([rdp_hh, backyard_hh])
divide = np.divide(backyard_mkt[1, :], backyard_mkt[0, :])
backyard_mkt = np.vstack([backyard_mkt, divide])
backyard_mkt[backyard_mkt == math.inf] = math.nan
np.nanmin(backyard_mkt[2, :])
