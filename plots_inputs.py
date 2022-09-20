# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:57:30 2022.

@author: monni
"""

# %% Preamble

# TODO: check MAUP

# IMPORT PACKAGES

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import copy

import inputs.parameters_and_options as inpprm
import inputs.data as inpdt
import equilibrium.functions_dynamic as eqdyn
import outputs.export_outputs as outexp
# import outputs.flood_outputs as outfld
# import outputs.export_outputs_floods as outval

print("Import information to be used in the simulation")


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/Data/'
path_precalc_inp = path_folder + 'Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_data + 'Scenarios/'
path_outputs = path_code + '/Output/'
path_floods = path_folder + "FATHOM/"


# IMPORT PARAMETERS AND OPTIONS

options = inpprm.import_options()
param = inpprm.import_param(
    path_precalc_inp, path_outputs, path_folder, options)

# Set custom options for this simulation
#  Dummy for taking floods into account in the utility function
options["agents_anticipate_floods"] = 1
#  Dummy for preventing new informal settlement development
options["informal_land_constrained"] = 0
#  TODO: add option to take into account less likely developments?

# More custom options regarding flood model
#  Dummy for taking pluvial floods into account (on top of fluvial floods)
options["pluvial"] = 1
#  Dummy for reducing pluvial risk for (better protected) formal structures
options["correct_pluvial"] = 1
#  Dummy for taking coastal floods into account (on top of fluvial floods)
options["coastal"] = 1
#  Digital elevation to be used with coastal flood data (MERITDEM or NASADEM)
#  NB: MERITDEM is also the DEM used for fluvial and pluvial flood data
options["dem"] = "MERITDEM"
#  We consider undefended flood maps as our default because they are more
#  reliable
options["defended"] = 0
#  Dummy for taking sea-level rise into account in coastal flood data
#  NB: Projections are up to 2050, based upon IPCC AR5 assessment for the
#  RCP 8.5 scenario
options["slr"] = 1

# More custom options regarding scenarios
options["inc_ineq_scenario"] = 2
options["pop_growth_scenario"] = 4
options["fuel_price_scenario"] = 2

# Processing options for this simulation
options["convert_sp_data"] = 0


# GIVE NAME TO SIMULATION TO EXPORT THE RESULTS
# (change according to custom parameters to be included)

name = ('floods' + str(options["agents_anticipate_floods"])
        + str(options["informal_land_constrained"]) + '_P'
        + str(options["pluvial"]) + str(options["correct_pluvial"])
        + '_C' + str(options["coastal"]) + str(options["slr"])
        + '_scenario' + str(options["inc_ineq_scenario"])
        + str(options["pop_growth_scenario"])
        + str(options["fuel_price_scenario"]))

path_plots = path_outputs + '/input_plots/'
path_tables = path_outputs + '/input_tables/'

year_temp = 0

# %% Load data

print("Load data and results to be plotted as outputs")


# BASIC GEOGRAPHIC DATA

grid, center = inpdt.import_grid(path_data)
geo_grid = gpd.read_file(path_data + "grid_reference_500.shp")
geo_TAZ = gpd.read_file(path_data + "TAZ_ampp_prod_attr_2013_2032.shp")
amenities = inpdt.import_amenities(path_precalc_inp, options)


# MACRO DATA

(interest_rate, population, housing_type_data, total_RDP
 ) = inpdt.import_macro_data(param, path_scenarios, path_folder)


# HOUSEHOLDS AND INCOME DATA

(mean_income, households_per_income_class, average_income, income_mult,
 income_2011, households_per_income_and_housing
 ) = inpdt.import_income_classes_data(param, path_data)

(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(path_precalc_inp)

housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')
housing_types[np.isnan(housing_types)] = 0

# We convert income distribution data (at SP level) to grid dimensions for use
# in income calibration: long to run, uncomment only if needed

if options["convert_sp_data"] == 1:
    print("Convert SP data to grid dimensions - start")
    income_distribution_grid = inpdt.convert_income_distribution(
        income_distribution, grid, path_data, data_sp)
    print("Convert SP data to grid dimensions - end")

income_distribution_grid = np.load(path_data + "income_distrib_grid.npy")


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

#  We update parameter vector with construction parameters
(param, minimum_housing_supply, agricultural_rent
 ) = inpprm.import_construction_parameters(
    param, grid, housing_types_sp, data_sp["dwelling_size"],
    mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land,
    interest_rate, options
    )

# OTHER VALIDATION DATA

# Makes no sense?
data = scipy.io.loadmat(path_precalc_inp + 'data.mat')['data']
data_avg_income = data['gridAverageIncome'][0][0].squeeze()
data_avg_income[np.isnan(data_avg_income)] = 0

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')
cal_average_income = np.load(
    path_precalc_transp + 'GRID_averageIncome_0.npy')
# modal_shares = np.load(
#     path_precalc_transp + 'GRID_modalShares_0.npy')
# od_flows = np.load(
#     path_precalc_transp + 'GRID_ODflows_0.npy')

# LOAD EQUILIBRIUM DATA

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


# LOAD SIMULATION DATA (from main.py)

# simulation_households_center = np.load(
#     path_outputs + name + '/simulation_households_center.npy')
# simulation_households_housing_type = np.load(
#     path_outputs + name + '/simulation_households_housing_type.npy')
# simulation_dwelling_size = np.load(
#     path_outputs + name + '/simulation_dwelling_size.npy')
# simulation_rent = np.load(
#     path_outputs + name + '/simulation_rent.npy')
# simulation_households_housing_type = np.load(
#     path_outputs + name + '/simulation_households_housing_type.npy')
# simulation_households = np.load(
#     path_outputs + name + '/simulation_households.npy')
# simulation_error = np.load(
#     path_outputs + name + '/simulation_error.npy')
# simulation_utility = np.load(
#     path_outputs + name + '/simulation_utility.npy')
# simulation_deriv_housing = np.load(
#     path_outputs + name + '/simulation_deriv_housing.npy')
# simulation_T = np.load(
#     path_outputs + name + '/simulation_T.npy')


# LOAD FLOOD DATA

# We enforce option to show damages even when agents do not anticipate them
options["agents_anticipate_floods"] = 1
(fraction_capital_destroyed, structural_damages_small_houses,
 structural_damages_medium_houses, structural_damages_large_houses,
 content_damages, structural_damages_type1, structural_damages_type2,
 structural_damages_type3a, structural_damages_type3b,
 structural_damages_type4a, structural_damages_type4b
 ) = inpdt.import_full_floods_data(
     options, param, path_folder, housing_type_data)


# SCENARIOS

(spline_agricultural_rent, spline_interest_rate,
 spline_population_income_distribution, spline_inflation,
 spline_income_distribution, spline_population,
 spline_income, spline_minimum_housing_supply, spline_fuel
 ) = eqdyn.import_scenarios(income_2011, param, grid, path_scenarios,
                            options)


# %% INPUT PLOTS

try:
    os.mkdir(path_plots)
except OSError as error:
    print(error)

try:
    os.mkdir(path_tables)
except OSError as error:
    print(error)


# First map land availability

coeef_land_FP_map = outexp.export_map(
    coeff_land[0], grid, geo_grid, path_plots, 'coeff_land_formal',
    "Land availability (in %) for formal private housing",
    path_tables,
    ubnd=1, lbnd=0)

coeef_land_IB_map = outexp.export_map(
    coeff_land[1], grid, geo_grid, path_plots, 'coeff_land_backyard',
    "Land availability (in %) for informal backyard housing",
    path_tables,
    ubnd=1, lbnd=0)

coeef_land_IS_map = outexp.export_map(
    coeff_land[2], grid, geo_grid, path_plots, 'coeff_land_informal',
    "Land availability (in %) for informal settlement housing",
    path_tables,
    ubnd=1, lbnd=0)

coeef_land_FS_map = outexp.export_map(
    coeff_land[3], grid, geo_grid, path_plots, 'coeff_land_subsidized',
    "Land availability (in %) for formal subsidized housing",
    path_tables,
    ubnd=1, lbnd=0)


# For job centers, need to map transport zones

jobsTable, selected_centers = outexp.import_employment_geodata(
    households_per_income_class, param, path_data)

jobs_total = np.nansum([jobsTable["poor_jobs"], jobsTable["midpoor_jobs"],
                        jobsTable["midrich_jobs"], jobsTable["rich_jobs"]],
                       0)
jobs_total = pd.DataFrame(jobs_total)
jobs_total = jobs_total.rename(columns={jobs_total.columns[0]: 'jobs_total'})

jobs_total_2d = pd.merge(geo_TAZ, jobs_total,
                         left_index=True, right_index=True)
jobs_total_2d = pd.merge(jobs_total_2d, selected_centers,
                         left_index=True, right_index=True)
# jobs_total_2d.to_file(path_tables + 'jobs_total' + '.shp')
jobs_total_2d.drop('geometry', axis=1).to_csv(
    path_tables + 'jobs_total' + '.csv')


jobs_total_2d_select = jobs_total_2d[
    (jobs_total_2d.geometry.bounds.maxy < -3740000)
    & (jobs_total_2d.geometry.bounds.maxx < -10000)].copy()
# fig, ax = plt.subplots(figsize=(8, 10))
# ax.set_axis_off()
# plt.title("Selected job centers")
# jobs_total_2d.plot(column='selected_centers', ax=ax)
# plt.savefig(path_plots + 'selected_centers')
# plt.close()
jobs_total_2d_select.loc[
    jobs_total_2d_select["selected_centers"] == 0, 'jobs_total'
    ] = 0
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Number of jobs in selected job centers (data)")
jobs_total_2d_select.plot(column='jobs_total', ax=ax,
                          cmap='Reds', legend=True)
plt.savefig(path_plots + 'jobs_total_in_selected_centers')
plt.close()

# We do the same across income groups
jobs_poor_2d = pd.merge(geo_TAZ, jobsTable["poor_jobs"],
                        left_index=True, right_index=True)
jobs_poor_2d = pd.merge(jobs_poor_2d, selected_centers,
                        left_index=True, right_index=True)
# jobs_poor_2d.to_file(path_tables + 'jobs_poor' + '.shp')
jobs_poor_2d.drop('geometry', axis=1).to_csv(
    path_tables + 'jobs_poor' + '.csv')
jobs_poor_2d_select = jobs_poor_2d[
    (jobs_poor_2d.geometry.bounds.maxy < -3740000)
    & (jobs_poor_2d.geometry.bounds.maxx < -10000)].copy()
jobs_poor_2d_select.loc[
    jobs_poor_2d_select["selected_centers"] == 0, 'jobs_poor'
    ] = 0
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Number of poor jobs in selected job centers (data)")
jobs_poor_2d_select.plot(column='poor_jobs', ax=ax,
                         cmap='Reds', legend=True)
plt.savefig(path_plots + 'jobs_poor_in_selected_centers')
plt.close()

jobs_midpoor_2d = pd.merge(geo_TAZ, jobsTable["midpoor_jobs"],
                           left_index=True, right_index=True)
jobs_midpoor_2d = pd.merge(jobs_midpoor_2d, selected_centers,
                           left_index=True, right_index=True)
# jobs_midpoor_2d.to_file(path_tables + 'jobs_midpoor' + '.shp')
jobs_midpoor_2d.drop('geometry', axis=1).to_csv(
    path_tables + 'jobs_midpoor' + '.csv')
jobs_midpoor_2d_select = jobs_midpoor_2d[
    (jobs_midpoor_2d.geometry.bounds.maxy < -3740000)
    & (jobs_midpoor_2d.geometry.bounds.maxx < -10000)].copy()
jobs_midpoor_2d_select.loc[
    jobs_midpoor_2d_select["selected_centers"] == 0, 'jobs_midpoor'
    ] = 0
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Number of midpoor jobs in selected job centers (data)")
jobs_midpoor_2d_select.plot(column='midpoor_jobs', ax=ax,
                            cmap='Reds', legend=True)
plt.savefig(path_plots + 'jobs_midpoor_in_selected_centers')
plt.close()

jobs_midrich_2d = pd.merge(geo_TAZ, jobsTable["midrich_jobs"],
                           left_index=True, right_index=True)
jobs_midrich_2d = pd.merge(jobs_midrich_2d, selected_centers,
                           left_index=True, right_index=True)
# jobs_midrich_2d.to_file(path_tables + 'jobs_midrich' + '.shp')
jobs_midrich_2d.drop('geometry', axis=1).to_csv(
    path_tables + 'jobs_midrich' + '.csv')
jobs_midrich_2d_select = jobs_midrich_2d[
    (jobs_midrich_2d.geometry.bounds.maxy < -3740000)
    & (jobs_midrich_2d.geometry.bounds.maxx < -10000)].copy()
jobs_midrich_2d_select.loc[
    jobs_midrich_2d_select["selected_centers"] == 0, 'jobs_midrich'
    ] = 0
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Number of midrich jobs in selected job centers (data)")
jobs_midrich_2d_select.plot(column='midrich_jobs', ax=ax,
                            cmap='Reds', legend=True)
plt.savefig(path_plots + 'jobs_midrich_in_selected_centers')
plt.close()

jobs_rich_2d = pd.merge(geo_TAZ, jobsTable["rich_jobs"],
                        left_index=True, right_index=True)
jobs_rich_2d = pd.merge(jobs_rich_2d, selected_centers,
                        left_index=True, right_index=True)
# jobs_rich_2d.to_file(path_tables + 'jobs_rich' + '.shp')
jobs_rich_2d.drop('geometry', axis=1).to_csv(
    path_tables + 'jobs_rich' + '.csv')
jobs_rich_2d_select = jobs_rich_2d[
    (jobs_rich_2d.geometry.bounds.maxy < -3740000)
    & (jobs_rich_2d.geometry.bounds.maxx < -10000)].copy()
jobs_rich_2d_select.loc[
    jobs_rich_2d_select["selected_centers"] == 0, 'jobs_rich'
    ] = 0
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Number of rich jobs in selected job centers (data)")
jobs_rich_2d_select.plot(column='rich_jobs', ax=ax,
                         cmap='Reds', legend=True)
plt.savefig(path_plots + 'jobs_rich_in_selected_centers')
plt.close()

# We also map calibrated incomes per job center (not spatialized through map)

income_centers_init = np.load(
    path_precalc_inp + 'incomeCentersKeep.npy')
income_centers_init[income_centers_init < 0] = 0
income_centers_init_merge = pd.DataFrame(income_centers_init)
income_centers_init_merge = income_centers_init_merge.rename(
    columns={income_centers_init_merge.columns[0]: 'poor_income',
             income_centers_init_merge.columns[1]: 'midpoor_income',
             income_centers_init_merge.columns[2]: 'midrich_income',
             income_centers_init_merge.columns[3]: 'rich_income'})

income_centers_init_merge["count"] = income_centers_init_merge.index + 1

selected_centers_merge = selected_centers.copy()
(selected_centers_merge["count"]
 ) = selected_centers_merge.selected_centers.cumsum()
selected_centers_merge.loc[
    selected_centers_merge.selected_centers == 0, "count"] = 0

income_centers_TAZ = pd.merge(income_centers_init_merge,
                              selected_centers_merge,
                              how='right', on='count')
income_centers_TAZ = income_centers_TAZ.fillna(value=0)

income_centers_2d = pd.merge(geo_TAZ, income_centers_TAZ,
                             left_index=True, right_index=True)
# income_centers_2d.to_file(path_tables + 'income_centers_2d' + '.shp')
income_centers_2d.drop('geometry', axis=1).to_csv(
    path_tables + 'income_centers_2d' + '.csv')

income_centers_2d_select = income_centers_2d[
    (income_centers_2d.geometry.bounds.maxy < -3740000)
    & (income_centers_2d.geometry.bounds.maxx < -10000)].copy()

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Average calibrated incomes per job center for poor households")
income_centers_2d_select.plot(column='poor_income', ax=ax,
                              cmap='Reds', legend=True)
plt.savefig(path_plots + 'poor_income_in_selected_centers')
plt.close()
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Average calibrated incomes per job center for mid-poor households")
income_centers_2d_select.plot(column='midpoor_income', ax=ax,
                              cmap='Reds', legend=True)
plt.savefig(path_plots + 'midpoor_income_in_selected_centers')
plt.close()
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Average calibrated incomes per job center for mid-rich households")
income_centers_2d_select.plot(column='midrich_income', ax=ax,
                              cmap='Reds', legend=True)
plt.savefig(path_plots + 'midrich_income_in_selected_centers')
plt.close()
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_axis_off()
plt.title("Average calibrated incomes per job center for rich households")
income_centers_2d_select.plot(column='rich_income', ax=ax,
                              cmap='Reds', legend=True)
plt.savefig(path_plots + 'rich_income_in_selected_centers')
plt.close()

# TODO: ask for TAZ code dictionnary to identify OD flows for some key
# job centers (CBD, etc.)

amenity_map = outexp.export_map(
    amenities, grid, geo_grid, path_plots,  'amenity_map',
    "Map of average amenity index per location",
    path_tables,
    ubnd=1.3, lbnd=0.8)


# FLOOD MAPS

fluviald_floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr',
                   'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
fluvialu_floods = ['FU_5yr', 'FU_10yr', 'FU_20yr', 'FU_50yr', 'FU_75yr',
                   'FU_100yr', 'FU_200yr', 'FU_250yr', 'FU_500yr', 'FU_1000yr']
pluvial_floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr', 'P_100yr',
                  'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr']
coastal_floods = ['C_MERITDEM_1_0000', 'C_MERITDEM_1_0002',
                  'C_MERITDEM_1_0005', 'C_MERITDEM_1_0010',
                  'C_MERITDEM_1_0025', 'C_MERITDEM_1_0050',
                  'C_MERITDEM_1_0100', 'C_MERITDEM_1_0250']

for flood in fluviald_floods:
    ref_flood = np.squeeze(pd.read_excel(path_floods + flood + ".xlsx"))
    ref_flood_area = ref_flood["prop_flood_prone"]
    ref_flood_depth = ref_flood["flood_depth"]
    ref_flood_map_area = outexp.export_map(
        ref_flood_area, grid, geo_grid,
        path_plots, flood + '_map_area',
        "",
        path_tables,
        ubnd=1)
    ref_flood_map_depth = outexp.export_map(
        ref_flood_depth, grid, geo_grid,
        path_plots, flood + '_map_depth',
        "",
        path_tables,
        ubnd=4)

for flood in fluvialu_floods:
    ref_flood = np.squeeze(pd.read_excel(path_floods + flood + ".xlsx"))
    ref_flood_area = ref_flood["prop_flood_prone"]
    ref_flood_depth = ref_flood["flood_depth"]
    ref_flood_map_area = outexp.export_map(
        ref_flood_area, grid, geo_grid,
        path_plots, flood + '_map_area',
        "",
        path_tables,
        ubnd=1)
    ref_flood_map_depth = outexp.export_map(
        ref_flood_depth, grid, geo_grid,
        path_plots, flood + '_map_depth',
        "",
        path_tables,
        ubnd=4)

for flood in pluvial_floods:
    ref_flood = np.squeeze(pd.read_excel(path_floods + flood + ".xlsx"))
    ref_flood_area = ref_flood["prop_flood_prone"]
    ref_flood_depth = ref_flood["flood_depth"]
    ref_flood_map_area = outexp.export_map(
        ref_flood_area, grid, geo_grid,
        path_plots, flood + '_map_area',
        "",
        path_tables,
        ubnd=1)
    ref_flood_map_depth = outexp.export_map(
        ref_flood_depth, grid, geo_grid,
        path_plots, flood + '_map_depth',
        "",
        path_tables,
        ubnd=4)

for flood in coastal_floods:
    ref_flood = np.squeeze(pd.read_excel(path_floods + flood + ".xlsx"))
    ref_flood_area = ref_flood["prop_flood_prone"]
    ref_flood_depth = ref_flood["flood_depth"]
    ref_flood_map_area = outexp.export_map(
        ref_flood_area, grid, geo_grid,
        path_plots, flood + '_map_area',
        "",
        path_tables,
        ubnd=1)
    ref_flood_map_depth = outexp.export_map(
        ref_flood_depth, grid, geo_grid,
        path_plots, flood + '_map_depth',
        "",
        path_tables,
        ubnd=4)

# We could plot fraction of capital destroyed separately for each
# flood type, but it would be similar due to bath-tub model
# NB: note that content damage function is the same for all housing types
for col in fraction_capital_destroyed.columns:
    value = fraction_capital_destroyed[col]
    outexp.export_map(value, grid, geo_grid,
                      path_plots, col + '_fract_K_destroyed', "",
                      path_tables,
                      ubnd=1)


# Finally, we map exogenous scenarios

scenario_income_distribution_1 = pd.read_csv(
    path_scenarios + 'Scenario_inc_distrib_1.csv', sep=';')
scenario_income_distribution_2 = pd.read_csv(
    path_scenarios + 'Scenario_inc_distrib_2.csv', sep=';')
scenario_income_distribution_3 = pd.read_csv(
    path_scenarios + 'Scenario_inc_distrib_3.csv', sep=';')

scenario_population_4 = pd.read_csv(
    path_scenarios + 'Scenario_pop_20201209.csv', sep=';')
scenario_population_3 = pd.read_csv(
    path_scenarios + 'Scenario_pop_3.csv', sep=';')
scenario_population_2 = pd.read_csv(
    path_scenarios + 'Scenario_pop_2.csv', sep=';')
scenario_population_1 = pd.read_csv(
    path_scenarios + 'Scenario_pop_1.csv', sep=';')

scenario_inflation = pd.read_csv(
    path_scenarios + 'Scenario_inflation_1.csv', sep=';')

scenario_interest_rate = pd.read_csv(
    path_scenarios + 'Scenario_interest_rate_1.csv', sep=';')

scenario_price_fuel_1 = pd.read_csv(
    path_scenarios + 'Scenario_price_fuel_1.csv', sep=',')
scenario_price_fuel_2 = pd.read_csv(
    path_scenarios + 'Scenario_price_fuel_2.csv', sep=';')
scenario_price_fuel_3 = pd.read_csv(
    path_scenarios + 'Scenario_price_fuel_3.csv', sep=',')

# NB: save tables?

year_simul = np.arange(2011, 2011 + 30)
income_groups = np.arange(1, 13)
barWidth = 0.25

fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(income_groups - barWidth,
       scenario_income_distribution_1["Households_nb_2040"],
       width=barWidth,
       color="tab:red", label="Low")
ax.bar(income_groups, scenario_income_distribution_2["Households_nb_2040"],
       width=barWidth,
       color="tab:blue", label="Medium")
ax.bar(income_groups + barWidth,
       scenario_income_distribution_3["Households_nb_2040"],
       width=barWidth,
       color="tab:green", label="High")
ax.set_ylim(0)
ax.yaxis.set_major_formatter(
    mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()

plt.xticks(income_groups)
plt.ylabel("Nb of HHs in each (data) income group in 2040"
           + " (inequality scenarios)", labelpad=15)
plt.savefig(path_plots + 'inc_dist_nb_scenario.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(income_groups - barWidth,
       scenario_income_distribution_1["INC_med_2040"],
       width=barWidth,
       color="tab:red", label="Low")
ax.bar(income_groups, scenario_income_distribution_2["INC_med_2040"],
       width=barWidth,
       color="tab:blue", label="Medium")
ax.bar(income_groups + barWidth,
       scenario_income_distribution_3["INC_med_2040"],
       width=barWidth,
       color="tab:green", label="High")
ax.set_ylim(0)
ax.yaxis.set_major_formatter(
    mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
# plt.tick_params(labelbottom=True)
plt.xticks(income_groups)
plt.ylabel("Med. income of HHs in each (data) income group in 2040"
           + " (inequality scenarios)", labelpad=15)
plt.savefig(path_plots + 'inc_dist_val_scenario.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(scenario_population_1["Year_pop"], scenario_population_1["HH_total"],
        color="tab:red", label="Low")
ax.plot(scenario_population_1["Year_pop"], scenario_population_2["HH_total"],
        color="tab:blue", label="Medium")
ax.plot(scenario_population_1["Year_pop"], scenario_population_3["HH_total"],
        color="tab:green", label="High")
ax.set_ylim(0)
ax.yaxis.set_major_formatter(
    mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
plt.tick_params(labelbottom=True)
plt.ylabel("Total population growth (scenarios)",
           labelpad=15)
plt.savefig(path_plots + 'pop_growth_scenario.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(scenario_inflation["Year_infla"],
        scenario_inflation["inflation_base_2010"],
        color="tab:blue")
ax.set_ylim(0)
ax.yaxis.set_major_formatter(
    mpl.ticker.StrMethodFormatter('{x:,.0f}'))
# plt.legend()
plt.tick_params(labelbottom=True)
plt.ylabel("Inflation growth relative to 2010 (in base 100)",
           labelpad=15)
plt.savefig(path_plots + 'infla_growth_scenario.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(scenario_interest_rate["Year_interest_rate"],
        scenario_interest_rate["real_interest_rate"],
        color="tab:blue")
# ax.set_ylim(0)
ax.yaxis.set_major_formatter(
    mpl.ticker.StrMethodFormatter('{x:,.0f}'))
# plt.legend()
plt.tick_params(labelbottom=True)
plt.ylabel("Interest rate (in %) history and projections",
           labelpad=15)
plt.savefig(path_plots + 'interest_rate_scenario.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(scenario_price_fuel_1["Year_fuel"],
        scenario_price_fuel_1["price_fuel"],
        color="tab:red", label="Low")
ax.plot(scenario_price_fuel_2["Year_fuel"],
        scenario_price_fuel_2["price_fuel"],
        color="tab:blue", label="Medium")
ax.plot(scenario_price_fuel_3["Year_fuel"],
        scenario_price_fuel_3["price_fuel"],
        color="tab:green", label="High")
# ax.set_ylim(0)
ax.yaxis.set_major_formatter(
    mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
plt.tick_params(labelbottom=True)
plt.ylabel("Fuel price history and projections (scenarios)",
           labelpad=15)
plt.savefig(path_plots + 'price_fuel_scenario.png')
plt.close()


# We also do maps for informal settlement scenarios

# TODO: apply polygon_medium_timing correction?

#  First for timing
informal_risks_short = pd.read_csv(
    path_folder + 'Land occupation/informal_settlements_risk_SHORT.csv',
    sep=',')
informal_risks_short = informal_risks_short.rename(
    columns={"area": "area_short"})
informal_risks_medium = pd.read_csv(
    path_folder + 'Land occupation/informal_settlements_risk_MEDIUM.csv',
    sep=',')
informal_risks_medium = informal_risks_medium.rename(
    columns={"area": "area_medium"})
informal_risks_long = pd.read_csv(
    path_folder + 'Land occupation/informal_settlements_risk_LONG.csv',
    sep=',')
informal_risks_long = informal_risks_long.rename(
    columns={"area": "area_long"})

informal_risks_timing = pd.concat(
    [informal_risks_short["area_short"],
     informal_risks_medium["area_medium"],
     informal_risks_long["area_long"]],
    axis=1)
informal_risks_timing["sum"] = (
    informal_risks_timing["area_short"]
    + informal_risks_timing["area_medium"]
    + informal_risks_timing["area_long"])
informal_risks_timing["argmax"] = np.zeros(24014)
informal_risks_timing["argmax"] = np.nan
informal_risks_timing[
    "argmax"][informal_risks_timing["sum"] > 0] = np.nanargmax(
        informal_risks_timing[["area_short", "area_medium", "area_long"]], 1)
informal_risks_timing["color"] = "tab:grey"
informal_risks_timing.loc[
    informal_risks_timing["argmax"] == 0, "color"] = "tab:red"
informal_risks_timing.loc[
    informal_risks_timing["argmax"] == 1, "color"] = "tab:blue"
informal_risks_timing.loc[
    informal_risks_timing["argmax"] == 2, "color"] = "tab:green"

plt.figure(figsize=(10, 7))
Map = plt.scatter(grid.x, grid.y, s=None,
                  c=informal_risks_timing["color"],
                  marker='.')
custom_lines = [Line2D([0], [0], color="tab:red", lw=4),
                Line2D([0], [0], color="tab:blue", lw=4),
                Line2D([0], [0], color="tab:green", lw=4)]
plt.legend(custom_lines, ['Short', 'Medium', 'Long'],
           loc='upper right', bbox_to_anchor=(0.925, 0.9))
plt.axis('off')
plt.title("Timing of informal settlement expansion risk")
plt.savefig(path_plots + "informal_settlement_risk_timing")
plt.close()
informal_risks_timing.to_csv(path_tables
                             + 'informal_settlement_risk_timing.csv')

# Then for probability

informal_risks_LOW = pd.read_csv(
    path_folder + 'Land occupation/informal_settlements_risk_pLOW.csv',
    sep=',')
informal_risks_LOW = informal_risks_LOW.rename(
    columns={"area": "area_LOW"})
informal_risks_MEDIUM = pd.read_csv(
    path_folder + 'Land occupation/informal_settlements_risk_pMEDIUM.csv',
    sep=',')
informal_risks_MEDIUM = informal_risks_MEDIUM.rename(
    columns={"area": "area_MEDIUM"})
informal_risks_HIGH = pd.read_csv(
    path_folder + 'Land occupation/informal_settlements_risk_pHIGH.csv',
    sep=',')
informal_risks_HIGH = informal_risks_HIGH.rename(
    columns={"area": "area_HIGH"})
informal_risks_VERYHIGH = pd.read_csv(
    path_folder + 'Land occupation/informal_settlements_risk_pVERYHIGH.csv',
    sep=',')
informal_risks_VERYHIGH = informal_risks_VERYHIGH.rename(
    columns={"area": "area_VERYHIGH"})

informal_risks_proba = pd.concat(
    [informal_risks_LOW["area_LOW"],
     informal_risks_MEDIUM["area_MEDIUM"],
     informal_risks_HIGH["area_HIGH"],
     informal_risks_VERYHIGH["area_VERYHIGH"]],
    axis=1)
informal_risks_proba["sum"] = (
    informal_risks_proba["area_LOW"]
    + informal_risks_proba["area_MEDIUM"]
    + informal_risks_proba["area_HIGH"]
    + informal_risks_proba["area_VERYHIGH"])
informal_risks_proba["argmax"] = np.zeros(24014)
informal_risks_proba["argmax"] = np.nan
informal_risks_proba[
    "argmax"][informal_risks_proba["sum"] > 0] = np.nanargmax(
        informal_risks_proba[
            ["area_LOW", "area_MEDIUM", "area_HIGH", "area_VERYHIGH"]
            ], 1)
informal_risks_proba["color"] = "tab:grey"
informal_risks_proba.loc[
    informal_risks_proba["argmax"] == 0, "color"] = "tab:green"
informal_risks_proba.loc[
    informal_risks_proba["argmax"] == 1, "color"] = "tab:blue"
informal_risks_proba.loc[
    informal_risks_proba["argmax"] == 2, "color"] = "tab:orange"
informal_risks_proba.loc[
    informal_risks_proba["argmax"] == 3, "color"] = "tab:red"

plt.figure(figsize=(10, 7))
Map = plt.scatter(grid.x, grid.y, s=None,
                  c=informal_risks_proba["color"],
                  marker='.')
custom_lines = [Line2D([0], [0], color="tab:green", lw=4),
                Line2D([0], [0], color="tab:blue", lw=4),
                Line2D([0], [0], color="tab:orange", lw=4),
                Line2D([0], [0], color="tab:red", lw=4)]
plt.legend(custom_lines, ['Low', 'Medium', 'High', "Very high"],
           loc='upper right', bbox_to_anchor=(0.925, 0.9))
plt.axis('off')
plt.title("Probability of informal settlement expansion risk")
plt.savefig(path_plots + "informal_settlement_risk_proba")
plt.close()
informal_risks_proba.to_csv(path_tables
                            + 'informal_settlement_risk_proba.csv')
