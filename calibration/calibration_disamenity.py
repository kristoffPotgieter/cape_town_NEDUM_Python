# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:29:30 2020.

@author: Charlotte Liotta
"""

# %% Preamble


# IMPORT PACKAGES

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

import inputs.data as inpdt
import inputs.parameters_and_options as inpprm
import equilibrium.compute_equilibrium as eqcmp
import equilibrium.functions_dynamic as eqdyn
import outputs.export_outputs as outexp
import outputs.export_outputs_floods as outfld


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

#  Set timeline for simulations
t = np.arange(0, 30)

# GIVE NAME TO SIMULATION TO EXPORT THE RESULTS
# (change according to custom parameters to be included)
# TODO: Change

date = 'no_floods_scenario'
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
#  TODO: Set as a variable, not a parameter
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

#  We correct areas for each housing type at baseline year for the amount of
#  constructible land in each type
coeff_land = inpdt.import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal,
    spline_land_RDP, param, 0)

#  We update land use parameters at baseline (relies on data)
housing_limit = inpdt.import_housing_limit(grid, param)

#  TODO: plug outputs in a new variable (not param) and adapt linked functions
(param, minimum_housing_supply, agricultural_rent
 ) = inpprm.import_construction_parameters(
    param, grid, housing_types_sp, data_sp["dwelling_size"],
    mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land,
    interest_rate
    )

# FLOOD DATA (takes some time)
#  TODO: create a new variable instead of storing in param
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


# %% Run initial state for several values of amenities


# List of parameters

#  We start with arbitrary values we would like to test for disamenity
#  parameters in backyard and informal settlements
list_amenity_backyard = np.arange(0.65, 0.85, 0.01)
list_amenity_settlement = np.arange(0.65, 0.85, 0.01)
#  Returns a grid of 400 (20x20) x-y coordinates determined above
housing_type_total = pd.DataFrame(np.array(np.meshgrid(
    list_amenity_backyard, list_amenity_settlement)).T.reshape(-1, 2))
#  We rename columns and create new ones to store future estimates
housing_type_total.columns = ["param_backyard", "param_settlement"]
housing_type_total["formal"] = np.zeros(len(housing_type_total.param_backyard))
housing_type_total["backyard"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["informal"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["subsidized"] = np.zeros(
    len(housing_type_total.param_backyard))


# We simulate an arbitrary number of equilibria (3x20) for some of the initial
# values of disamenity parameters defined above, and update the row
# corresponding to such values in output matrix with the resulting total number
# of households for each housing type

# Note that this is obviously very long! This is why the number of simulations
# has been reduced to likely values of backyard disamenity parameter
# TODO: how is it determined?

for i in range(17, len(list_amenity_backyard)):
    for j in range(0, len(list_amenity_settlement)):
        param["amenity_backyard"] = list_amenity_backyard[i]
        param["amenity_settlement"] = list_amenity_settlement[j]
        param["pockets"] = np.ones(24014) * param["amenity_settlement"]
        param["backyard_pockets"] = np.ones(24014) * param["amenity_backyard"]
        (initial_state_utility, initial_state_error,
         initial_state_simulated_jobs, initial_state_households_housing_types,
         initial_state_household_centers, initial_state_households,
         initial_state_dwelling_size, initial_state_housing_supply,
         initial_state_rent, initial_state_rent_matrix,
         initial_state_capital_land, initial_state_average_income,
         initial_state_limit_city) = eqcmp.compute_equilibrium(
            fraction_capital_destroyed, amenities, param, housing_limit,
            population, households_per_income_class, total_RDP, coeff_land,
            income_net_of_commuting_costs, grid, options, agricultural_rent,
            interest_rate, number_properties_RDP, average_income, mean_income,
            income_class_by_housing_type, minimum_housing_supply,
            param["coeff_A"])
        housing_type_total.loc[
            (housing_type_total.param_backyard == param["amenity_backyard"]) &
            (housing_type_total.param_settlement == param["amenity_settlement"]
             ), 2:6
            ] = np.nansum(initial_state_households_housing_types, 1)


print('*** End of simulations for chosen parameters ***')


# %% Pick best solution

# We calculate the absolute difference between simulated and actual number of
# households for each housing type (from Claus, not census)
distance_share = np.abs(
    housing_type_total.iloc[:, 2:5] - housing_type_data[None, 0:3])

# We compute a score equal to the sum of those differences for backyard and
# informal settlements
distance_share_score = distance_share.iloc[:, 1] + distance_share.iloc[:, 2]

# We get the index and the value of the minimum score, from which we determine
# the best values for disamenity parameters in output matrix
which = np.argmin(distance_share_score)
min_score = np.nanmin(distance_share_score)
calibrated_amenities = housing_type_total.iloc[which, 0:2]

# We store those values as parameters to be used in main script
# TODO: why not directly?
param["amenity_backyard"] = calibrated_amenities[0]
param["amenity_settlement"] = calibrated_amenities[1]


# %% IS Pockets

pockets = pd.read_excel(
    'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/is_pockets.xlsx')
param["pockets"] = np.zeros(24014) + param["amenity_settlement"]
index = 0
index_max = 12
(initial_state_utility, initial_state_error, initial_state_simulated_jobs,
 initial_state_households_housing_types, initial_state_household_centers,
 initial_state_households, initial_state_dwelling_size,
 initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix,
 initial_state_capital_land, initial_state_average_income,
 initial_state_limit_city) = eqcmp.compute_equilibrium(
    fraction_capital_destroyed, amenities, param, housing_limit, population,
    households_per_income_class, total_RDP, coeff_land,
    income_net_of_commuting_costs, grid, options, agricultural_rent,
    interest_rate, number_properties_RDP, average_income, mean_income,
    income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])
metrics = np.zeros(index_max)
save_param = np.zeros((index_max, 24014))
pop_data = ((96300/sum(housing_types_grid.informal_grid_2011))
            * (housing_types_grid.informal_grid_2011))

for index in range(0, index_max):
    diff = np.zeros(
        len(np.unique(pockets.IS_Pocket_[~np.isnan(pockets.IS_Pocket_)])))
    for i in range(
            0,
            len(np.unique(pockets.IS_Pocket_[~np.isnan(pockets.IS_Pocket_)]))
            ):
        pocket = int(np.unique(pockets.IS_Pocket_[
                     ~np.isnan(pockets.IS_Pocket_)])[i])
        diff[i] = (
            sum(pop_data[pockets.IS_Pocket_ == pocket])
            - sum(initial_state_households_housing_types[2, :][
                pockets.IS_Pocket_ == pocket])
            )
        adj = (diff[i] / 10000)
        save_param[index, :] = param["pockets"]
        (param["pockets"][pockets.IS_Pocket_ == pocket]
         ) = param["pockets"][pockets.IS_Pocket_ == pocket] + adj
    metrics[index] = sum(np.abs(diff))
    param["pockets"][param["pockets"] < 0.05] = 0.05
    param["pockets"][param["pockets"] > 0.99] = 0.99
    (initial_state_utility, initial_state_error, initial_state_simulated_jobs,
     initial_state_households_housing_types, initial_state_household_centers,
     initial_state_households, initial_state_dwelling_size,
     initial_state_housing_supply, initial_state_rent,
     initial_state_rent_matrix, initial_state_capital_land,
     initial_state_average_income, initial_state_limit_city
     ) = eqcmp.compute_equilibrium(
         fraction_capital_destroyed, amenities, param,
         housing_limit, population, households_per_income_class, total_RDP,
         coeff_land, income_net_of_commuting_costs, grid, options,
         agricultural_rent, interest_rate, number_properties_RDP,
         average_income, mean_income, income_class_by_housing_type,
         minimum_housing_supply, param["coeff_A"])

index_min = np.argmin(metrics)
metrics[index_min]
param["pockets"] = save_param[index_min]
param["pockets"][param["pockets"] < 0.05] = 0.05
param["pockets"][param["pockets"] > 0.99] = 0.99
(initial_state_utility, initial_state_error, initial_state_simulated_jobs,
 initial_state_households_housing_types, initial_state_household_centers,
 initial_state_households, initial_state_dwelling_size,
 initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix,
 initial_state_capital_land, initial_state_average_income,
 initial_state_limit_city) = eqcmp.compute_equilibrium(
    fraction_capital_destroyed, amenities, param, housing_limit, population,
    households_per_income_class, total_RDP, coeff_land,
    income_net_of_commuting_costs, grid, options, agricultural_rent,
    interest_rate, number_properties_RDP, average_income, mean_income,
    income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])

# GIVE NAME TO SIMULATION TO EXPORT THE RESULTS
# (change according to custom parameters to be included)

date = 'no_floods_scenario'
name = date + '_' + str(options["pluvial"]) + '_' + str(
    options["informal_land_constrained"])

os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name)
outexp.validation_density_housing_types(
    grid, initial_state_households_housing_types, housing_types_grid, name, 1)


# %% IS Pockets

# solver
index = 0
index_max = 30
(initial_state_utility, initial_state_error, initial_state_simulated_jobs,
 initial_state_households_housing_types, initial_state_household_centers,
 initial_state_households, initial_state_dwelling_size,
 initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix,
 initial_state_capital_land, initial_state_average_income,
 initial_state_limit_city) = eqcmp.compute_equilibrium(
    fraction_capital_destroyed, amenities, param, housing_limit, population,
    households_per_income_class, total_RDP, coeff_land,
    income_net_of_commuting_costs, grid, options, agricultural_rent,
    interest_rate, number_properties_RDP, average_income, mean_income,
    income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])
metrics = np.zeros(index_max)

# pockets
pockets = pd.read_excel(
    'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/is_pockets.xlsx')
param["pockets"] = np.zeros(24014) + param["amenity_settlement"]
save_param_informal_settlements = np.zeros((index_max, 24014))
pop_data_is = ((96300/sum(housing_types_grid.informal_grid_2011))
               * (housing_types_grid.informal_grid_2011))
metrics_is = np.zeros(index_max)

# backyards
param["backyard_pockets"] = np.zeros(24014) + param["amenity_backyard"]
save_param_backyards = np.zeros((index_max, 24014))
pop_data_ib = ((85600/sum(housing_types_grid.backyard_grid_2011))
               * (housing_types_grid.backyard_grid_2011))
metrics_ib = np.zeros(index_max)

for index in range(0, index_max):

    print("********************* NEW ITERATION ************************")
    print("INDEX  = " + str(index))

    # IS
    diff_is = np.zeros(
        len(np.unique(pockets.IS_Pocket_[~np.isnan(pockets.IS_Pocket_)])))
    for i in range(
            0,
            len(np.unique(pockets.IS_Pocket_[~np.isnan(pockets.IS_Pocket_)]))):
        pocket = int(np.unique(pockets.IS_Pocket_[
                     ~np.isnan(pockets.IS_Pocket_)])[i])
        diff_is[i] = (
            sum(pop_data_is[pockets.IS_Pocket_ == pocket])
            - sum(initial_state_households_housing_types[2, :][
                pockets.IS_Pocket_ == pocket])
            ) / sum(pop_data_is[pockets.IS_Pocket_ == pocket])
        adj = (diff_is[i] / (100))
        save_param_informal_settlements[index, :] = param["pockets"]
        (param["pockets"][pockets.IS_Pocket_ == pocket]
         ) = param["pockets"][pockets.IS_Pocket_ == pocket] + adj
    metrics_is[index] = np.nansum(np.abs(diff_is))
    param["pockets"][param["pockets"] < 0.05] = 0.05
    param["pockets"][param["pockets"] > 0.99] = 0.99

    # IB
    diff_ib = np.zeros(166)
    for i in range(0, 166):
        diff_ib[i] = (
            sum(pop_data_ib[(grid.dist > i/2) & (grid.dist < i/2 + 0.5)])
            - sum(initial_state_households_housing_types[1, :][
                (grid.dist > i/2) & (grid.dist < i/2 + 0.5)])
            )
        adj = (diff_ib[i] / 100000)
        save_param_backyards[index, :] = param["backyard_pockets"]
        param["backyard_pockets"][
            (grid.dist > i/2) & (grid.dist < i/2 + 0.5)
            ] = param["backyard_pockets"][
                (grid.dist > i/2) & (grid.dist < i/2 + 0.5)
                ] + adj
    metrics_ib[index] = sum(np.abs(diff_ib))
    param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
    param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99

    metrics[index] = metrics_is[index] + metrics_ib[index]

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
        income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])

index_min = np.argmin(metrics_is)
metrics[index_min]
param["pockets"] = save_param_informal_settlements[index_min]
param["pockets"][param["pockets"] < 0.05] = 0.05
param["pockets"][param["pockets"] > 0.99] = 0.99
param["backyard_pockets"] = save_param_backyards[index_min]
param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99
(initial_state_utility, initial_state_error, initial_state_simulated_jobs,
 initial_state_households_housing_types, initial_state_household_centers,
 initial_state_households, initial_state_dwelling_size,
 initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix,
 initial_state_capital_land, initial_state_average_income,
 initial_state_limit_city) = eqcmp.compute_equilibrium(
    fraction_capital_destroyed, amenities, param, housing_limit, population,
    households_per_income_class, total_RDP, coeff_land,
    income_net_of_commuting_costs, grid, options, agricultural_rent,
    interest_rate, number_properties_RDP, average_income, mean_income,
    income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])

name = '200pc_dist_pockets'
os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name)
outexp.validation_density_housing_types(
    grid, initial_state_households_housing_types, housing_types_grid, name, 1)

floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr',
          'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"
count_formal = housing_types_grid.formal_grid_2011 - data_rdp["count"]
count_formal[count_formal < 0] = 0
stats_per_housing_type_data = outexp.compute_stats_per_housing_type(
    floods, path_data, count_formal, data_rdp["count"],
    housing_types_grid.informal_grid_2011,
    housing_types_grid.backyard_grid_2011, options, param)
stats_per_housing_type_simul = outexp.compute_stats_per_housing_type(
    floods, path_data, initial_state_households_housing_types[0, :],
    initial_state_households_housing_types[3, :],
    initial_state_households_housing_types[2, :],
    initial_state_households_housing_types[1, :], options, param)
outfld.validation_flood(name, stats_per_housing_type_data,
                        stats_per_housing_type_simul, 'Data', 'Simul')

plt.plot(grid.dist[~np.isnan(pockets.IS_Pocket_)],
         param["pockets"][~np.isnan(pockets.IS_Pocket_)], 'o')
m, b = np.polyfit(grid.dist[~np.isnan(pockets.IS_Pocket_)],
                  param["pockets"][~np.isnan(pockets.IS_Pocket_)], 1)
plt.plot(grid.dist, m*grid.dist + b)

plt.plot(grid.dist, param["backyard_pockets"], 'o')
m, b = np.polyfit(grid.dist, param["backyard_pockets"], 1)
plt.plot(grid.dist, m*grid.dist + b)

np.save('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' +
        name + 'param_pockets.npy', save_param_informal_settlements[index_min])
np.save('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' +
        name + 'param_backyards.npy', save_param_backyards[index_min])
