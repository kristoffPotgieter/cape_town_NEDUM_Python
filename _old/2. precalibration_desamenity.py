# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:26:14 2022.

@author: vincentviguie
"""

# %% Preamble


# IMPORT PACKAGES

import numpy as np
import pandas as pd
import time
import datetime
import os

import inputs.data as inpdt
import inputs.parameters_and_options as inpprm

import equilibrium.compute_equilibrium as eqcmp
import equilibrium.functions_dynamic as eqdyn


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/Data/'
path_precalc_inp = path_folder + 'Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_data + 'Scenarios/'
path_outputs = path_code + '/Output/'
path_floods = path_folder + "FATHOM/"


# START TIMER FOR CODE OPTIMIZATION

start = time.process_time()


# %% Import parameters and options


# IMPORT DEFAULT PARAMETERS AND OPTIONS

options = inpprm.import_options()
param = inpprm.import_param(path_precalc_inp, path_outputs)

# OPTIONS FOR THIS SIMULATION

options["agents_anticipate_floods"] = 0
options["informal_land_constrained"] = 1

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
 income_baseline, households_per_income_and_housing
 ) = inpdt.import_income_classes_data(param, path_data)

#  We create this parameter to maintain money illusion in simulations
#  (see eqsim.run_simulation)
param["income_year_reference"] = mean_income

(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_baseline,
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

# param["pockets"][
#     (spline_land_informal(29) > 0) & (spline_land_informal(0) == 0)
#     ] = 0.79


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
    mitchells_plain_grid_baseline, grid_formal_density_HFA, coeff_land,
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

(spline_agricultural_price, spline_interest_rate,
 spline_population_income_distribution, spline_inflation,
 spline_income_distribution, spline_population,
 spline_income, spline_minimum_housing_supply, spline_fuel
 ) = eqdyn.import_scenarios(income_baseline, param, grid, path_scenarios)

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
incomeNetOfCommuting, *_ = inpdt.import_transport_data(
      grid, param, 0, households_per_income_class, average_income,
      spline_inflation, spline_fuel,
      spline_population_income_distribution, spline_income_distribution,
      path_precalc_inp, path_precalc_transp, 'GRID')
income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')


# %% Calibration of the informal housing parameters
# General calibration (see Pfeiffer et al., appendix C5)

# TODO: does not converge with newly calibrated kappa.
# Then, it works but is slightly slower with newly calibrated amenities
list_amenity_backyard = np.arange(0.72, 0.77, 0.01)
list_amenity_settlement = np.arange(0.68, 0.73, 0.01)
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

debut_calib_time = time.process_time()
number_total_iterations = (
    len(list_amenity_backyard) * len(list_amenity_settlement))
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
         initial_state_limit_city) = eqcmp.compute_equilibrium(
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

        # We fill output matrix with the total number of households per housing
        # type for given values of backyard and informal amenity parameters
        housing_type_total.loc[
            (housing_type_total.param_backyard == param["amenity_backyard"])
            & (housing_type_total.param_settlement
               == param["amenity_settlement"]),
            2:6] = np.nansum(initial_state_households_housing_types, 1)

        time_elapsed = time.process_time() - debut_calib_time
        iteration_number = i * len(list_amenity_settlement) + j + 1

        print(f"iteration {iteration_number}/{number_total_iterations} done.",
              str(datetime.timedelta(seconds=round(time_elapsed))),
              f"elapsed ({round(time_elapsed/iteration_number)}s per iter",
              "There remains:",
              str(datetime.timedelta(seconds=round(
                  time_elapsed
                  / iteration_number
                  * (number_total_iterations-iteration_number)))))

# We choose the set of parameters that minimize the sum of absolute differences
# between simulated and observed total number of households in each housing
# type (without RDP, which is exogenously set equal to data)

# TODO: Correct paper for use of formal sector in score definition
# Note that we consider both formal and informal backyarding as backyarding

distance_share = np.abs(
    housing_type_total.iloc[:, 2:5] - housing_type_data[None, 0:3])
distance_share_score = (
    distance_share.iloc[:, 1] + distance_share.iloc[:, 2])
# distance_share_score = (
#     distance_share.iloc[:, 1] + distance_share.iloc[:, 2]
#     + distance_share.iloc[:, 0])
# distance_share = np.abs(
#     housing_type_total.iloc[:, 3:5] - housing_type_data[None, 1:3])
# distance_share_score = (
#     distance_share.iloc[:, 0] + distance_share.iloc[:, 1])

which = np.argmin(distance_share_score)
min_score = np.nanmin(distance_share_score)
calibrated_amenities = housing_type_total.iloc[which, 0:2]

# Correspond to 0.74 and 0.70 in paper

param["amenity_backyard"] = calibrated_amenities[0]
param["amenity_settlement"] = calibrated_amenities[1]

# Works the same as in paper
np.save(path_precalc_inp + 'param_amenity_backyard.npy',
        param["amenity_backyard"])
np.save(path_precalc_inp + 'param_amenity_settlement.npy',
        param["amenity_settlement"])


# %% Calibration of the informal housing parameters
# Location-based calibration

# Note that this is VERY long! Shorten index_max and run in the background

index = 0
index_max = 20
metrics = np.zeros(index_max)

# We start from where we left (to gain time) and compute the equilibrium again
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
 ) = eqcmp.compute_equilibrium(
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

print("\n** ITERATIONS **")

debut_iterations_time = time.process_time()
number_total_iterations = index_max

# Then we optimize over the number of households per housing type PER PIXEL,
# and not just on the aggregate number (to acccount for differing disamenities
# per location, e.g. eviction probability, infrastructure networks, etc.)

# To do so, we use granular housing_types (from SAL data) instead of aggregate
# housing_types

param["disamenity_cvfactor"] = 100000

for index in range(0, index_max):

    # Note that rescaling the benchmark data is not necessary here, as we are
    # focusing on error terms and not values

    # IS
    diff_is = np.zeros(24014)
    for i in range(0, 24014):
        diff_is[i] = (housing_types.informal_grid[i]
                      - initial_state_households_housing_types[2, :][i])
        # We apply an empirical reweighting that helps convergence
        adj = (diff_is[i] / param["disamenity_cvfactor"])
        # We increase the amenity score when we underestimate the nb of HHs
        param["pockets"][i] = param["pockets"][i] + adj
    # We store iteration outcome and prevent extreme sorting from happening
    # due to the amenity score
    metrics_is[index] = sum(np.abs(diff_is))
    param["pockets"][param["pockets"] < 0.05] = 0.05
    param["pockets"][param["pockets"] > 0.99] = 0.99
    save_param_informal_settlements[index, :] = param["pockets"]

    # IB
    diff_ib = np.zeros(24014)
    for i in range(0, 24014):
        if options["actual_backyards"] == 1:
            diff_ib[i] = (housing_types.backyard_informal_grid[i]
                          + housing_types.backyard_formal_grid[i]
                          - initial_state_households_housing_types[1, :][i])
        elif options["actual_backyards"] == 0:
            diff_ib[i] = (housing_types.backyard_informal_grid[i]
                          - initial_state_households_housing_types[1, :][i])
        adj = (diff_ib[i] / param["disamenity_cvfactor"])
        param["backyard_pockets"][i] = param["backyard_pockets"][i] + adj
    metrics_ib[index] = sum(np.abs(diff_ib))
    param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
    param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99
    save_param_backyards[index, :] = param["backyard_pockets"]

    metrics[index] = metrics_is[index] + metrics_ib[index]

    # We run the equilibrium again with updated values of informal/backyard
    # housing disamenity indices, then go to the next iteration

    (initial_state_utility, initial_state_error, initial_state_simulated_jobs,
     initial_state_households_housing_types, initial_state_household_centers,
     initial_state_households, initial_state_dwelling_size,
     initial_state_housing_supply, initial_state_rent,
     initial_state_rent_matrix, initial_state_capital_land,
     initial_state_average_income, initial_state_limit_city
     ) = eqcmp.compute_equilibrium(
         fraction_capital_destroyed, amenities, param, housing_limit,
         population, households_per_income_class, total_RDP, coeff_land,
         income_net_of_commuting_costs, grid, options, agricultural_rent,
         interest_rate, number_properties_RDP, average_income, mean_income,
         income_class_by_housing_type, minimum_housing_supply,
         param["coeff_A"])

    time_elapsed = time.process_time() - debut_iterations_time
    iteration_number = index + 1

    print(f"iteration {iteration_number}/{number_total_iterations} finished.",
          str(datetime.timedelta(seconds=round(time_elapsed))),
          f"elapsed ({round(time_elapsed/iteration_number)}s per iteration)",
          "There remains:",
          str(datetime.timedelta(seconds=round(
              time_elapsed
              / iteration_number
              * (number_total_iterations-iteration_number))))
          )

# We pick the set of parameters that minimize the sum of absolute differences
# between data and simulation
index_min = np.argmin(metrics)
# metrics[index_min]
param["pockets"] = save_param_informal_settlements[index_min]
param["backyard_pockets"] = save_param_backyards[index_min]

# TODO: check that there is no missing value we need to fill in

try:
    os.mkdir(path_precalc_inp)
except OSError as error:
    print(error)

np.save(path_precalc_inp + 'param_pockets.npy',
        param["pockets"])
np.save(path_precalc_inp + 'param_backyards.npy',
        param["backyard_pockets"])
