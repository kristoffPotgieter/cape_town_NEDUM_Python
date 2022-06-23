# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:57:30 2022.

@author: monni
"""

# %% Preamble

# IMPORT PACKAGES

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy
import matplotlib.pyplot as plt
# import copy

import inputs.parameters_and_options as inpprm
import inputs.data as inpdt
import outputs.export_outputs as outexp

print("Import information to be used in the simulation")


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/2. Data/'
path_precalc_inp = path_folder + '0. Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_folder + 'data_Cape_Town/Scenarios/'
path_outputs = path_code + '/4. Sorties/'


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
#  Dummy for taking sea-level rise into account in coastal flood data
#  NB: Projections are up to 2050, based upon IPCC AR5 assessment for the
#  RCP 8.5 scenario
options["slr"] = 1

# Processing options for this simulation
options["convert_sp_data"] = 0


# GIVE NAME TO SIMULATION TO EXPORT THE RESULTS
# (change according to custom parameters to be included)

name = ('floods' + str(options["agents_anticipate_floods"])
        + str(options["informal_land_constrained"]) + '_P'
        + str(options["pluvial"]) + str(options["correct_pluvial"])
        + '_C' + str(options["coastal"]) + str(options["slr"])
        + '_loc')

path_plots = path_outputs + name + '/plots/'
path_tables = path_outputs + name + '/tables/'


# %% Load data

print("Load data and results to be plotted as outputs")


# BASIC GEOGRAPHIC DATA

grid, center = inpdt.import_grid(path_data)
geo_grid = gpd.read_file(path_data + "grid_reference_500.shp")
geo_TAZ = gpd.read_file(path_data + "TAZ_ampp_prod_attr_2013_2032.shp")


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


# OTHER VALIDATION DATA

data = scipy.io.loadmat(path_precalc_inp + 'data.mat')['data']

# TODO: do loop for simulations
income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')
cal_average_income = np.load(
    path_precalc_transp + 'GRID_averageIncome_0.npy')
modal_shares = np.load(
    path_precalc_transp + 'GRID_modalShares_0.npy')
od_flows = np.load(
    path_precalc_transp + 'GRID_ODflows_0.npy')

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

simulation_households_center = np.load(
    path_outputs + name + '/simulation_households_center.npy')
simulation_households_housing_type = np.load(
    path_outputs + name + '/simulation_households_housing_type.npy')
simulation_dwelling_size = np.load(
    path_outputs + name + '/simulation_dwelling_size.npy')
simulation_rent = np.load(
    path_outputs + name + '/simulation_rent.npy')
simulation_households_housing_type = np.load(
    path_outputs + name + '/simulation_households_housing_type.npy')
simulation_households = np.load(
    path_outputs + name + '/simulation_households.npy')
simulation_error = np.load(
    path_outputs + name + '/simulation_error.npy')
simulation_utility = np.load(
    path_outputs + name + '/simulation_utility.npy')
simulation_deriv_housing = np.load(
    path_outputs + name + '/simulation_deriv_housing.npy')
simulation_T = np.load(
    path_outputs + name + '/simulation_T.npy')


# %% Validation: draw maps and figures

print("Static equilibrium validation")
# TODO: integrate in shapefiles and csv + plotly

# POPULATION OUTPUTS

try:
    os.mkdir(path_plots)
except OSError as error:
    print(error)

try:
    os.mkdir(path_tables)
except OSError as error:
    print(error)

# Note that aggregate fit on income groups hold by construction
# Aggregate (and local) fit on housing types is enforced through
# disamenity parameter calibration but is not perfect,
# hence needs to be checked
agg_housing_type_valid = outexp.export_housing_types(
    initial_state_households_housing_types, housing_type_data,
    'Simulation', 'Data', path_plots, path_tables
    )

# We also validate the fit across housing types and income groups
(agg_FP_income_valid, agg_IB_income_valid, agg_IS_income_valid
 ) = outexp.export_households(
     initial_state_households, households_per_income_and_housing, 'Simulation',
     'Data', path_plots, path_tables)


#  IN ONE DIMENSION

# Now, we validate overall households density across space
dens_valid_1d = outexp.validation_density(
    grid, initial_state_households_housing_types, housing_types,
    path_plots, path_tables)

# We do the same for total number of households across space,
# housing types and income groups
dist_HH_per_housing_1d = outexp.validation_density_housing_types(
    grid, initial_state_households_housing_types, housing_types,
    path_plots, path_tables
    )
# TODO: switch back to SP level for more precision in validation data?
# Else, aggregate distance at a higher level?
dist_HH_per_income_1d = outexp.validation_density_income_groups(
    grid, initial_state_household_centers, income_distribution_grid,
    path_plots, path_tables
    )

# We also plot income groups across space (in 1D) for each housing type,
# even if we cannot validate such output
(dist_HH_per_housing_and_income_1d
 ) = outexp.validation_density_housing_and_income_groups(
     grid, initial_state_households, path_plots, path_tables)


#  IN TWO DIMENSIONS

#  For overall households
sim_nb_households_tot = np.nansum(initial_state_households_housing_types, 0)
total_sim = outexp.export_map(
    sim_nb_households_tot, grid, geo_grid, path_plots + 'total_sim',
    "Total number of households (simulation)",
    path_tables,
    ubnd=5000)
# Note that we lack households in Mitchell's Plain
# TODO: should we correct it?
data_nb_households_tot = np.nansum(housing_types[
    ["informal_grid", "backyard_informal_grid", "formal_grid"]
    ], 1)
total_data = outexp.export_map(
    data_nb_households_tot, grid, geo_grid, path_plots + 'total_data',
    "Total number of households (data)",
    path_tables,
    ubnd=5000)

#  Per housing type
sim_nb_households_formal = initial_state_households_housing_types[0, :]
formal_sim = outexp.export_map(
    sim_nb_households_formal, grid, geo_grid, path_plots + 'formal_sim',
    "Number of households in formal private (simulation)",
    path_tables,
    ubnd=1000)
data_nb_households_formal = (housing_types["formal_grid"]
                             - initial_state_households_housing_types[3, :])
formal_data = outexp.export_map(
    data_nb_households_formal, grid, geo_grid, path_plots + 'formal_data',
    "Number of households in formal private (data)",
    path_tables,
    ubnd=1000)

sim_nb_households_backyard = initial_state_households_housing_types[1, :]
backyard_sim = outexp.export_map(
    sim_nb_households_backyard, grid, geo_grid, path_plots + 'backyard_sim',
    "Number of households in informal backyard (simulation)",
    path_tables,
    ubnd=1000)
data_nb_households_backyard = housing_types["backyard_informal_grid"]
backyard_data = outexp.export_map(
    data_nb_households_backyard, grid, geo_grid, path_plots + 'backyard_data',
    "Number of households in informal backyard (data)",
    path_tables,
    ubnd=1000)

sim_nb_households_informal = initial_state_households_housing_types[2, :]
informal_sim = outexp.export_map(
    sim_nb_households_informal, grid, geo_grid, path_plots + 'informal_sim',
    "Number of households in informal settlements (simulation)",
    path_tables,
    ubnd=3000)
data_nb_households_informal = housing_types["informal_grid"]
informal_data = outexp.export_map(
    data_nb_households_informal, grid, geo_grid, path_plots + 'informal_data',
    "Number of households in informal settlements (data)",
    path_tables,
    ubnd=3000)

data_nb_households_rdp = initial_state_households_housing_types[3, :]
rdp_sim = outexp.export_map(
    data_nb_households_rdp, grid, geo_grid, path_plots + 'rdp_sim',
    "Number of households in formal subsidized (data)",
    path_tables,
    ubnd=1800)

#  Per income group
#  NB: validation data is disaggregated from SP, hence the smooth appearance,
#  not necessarily corresponding to reality (we do not plot it)
sim_nb_households_poor = initial_state_household_centers[0, :]
poor_sim = outexp.export_map(
    sim_nb_households_poor, grid, geo_grid, path_plots + 'poor_sim',
    "Number of poor households (simulation)",
    path_tables,
    ubnd=5000)
sim_nb_households_midpoor = initial_state_household_centers[1, :]
midpoor_sim = outexp.export_map(
    sim_nb_households_midpoor, grid, geo_grid, path_plots + 'midpoor_sim',
    "Number of mid-poor households (simulation)",
    path_tables,
    ubnd=2000)
sim_nb_households_midrich = initial_state_household_centers[2, :]
midrich_sim = outexp.export_map(
    sim_nb_households_midrich, grid, geo_grid, path_plots + 'midrich_sim',
    "Number of mid-rich households (simulation)",
    path_tables,
    ubnd=1000)
sim_nb_households_rich = initial_state_household_centers[3, :]
rich_sim = outexp.export_map(
    sim_nb_households_rich, grid, geo_grid, path_plots + 'rich_sim',
    "Number of rich households (simulation)",
    path_tables,
    ubnd=500)


# HOUSING SUPPLY OUTPUTS

# By plotting the housing supply per unit of available land, we may check
# whether the bell-shaped curve of urban development holds
avg_hsupply_1d = outexp.plot_housing_supply(
    grid, initial_state_housing_supply, path_plots, path_tables)

# We now consider overall land to recover building density
#  TODO: pb with Mitchell's Plain?
housing_supply = initial_state_housing_supply * coeff_land * 0.25
hsupply_noland_1d = outexp.plot_housing_supply_noland(
    grid, housing_supply, path_plots, path_tables)

hsupply_tot = np.nansum(housing_supply, 0)
hsupply_2d_sim = outexp.export_map(
    hsupply_tot, grid, geo_grid, path_plots + 'hsupply_2d_sim',
    "Total housing supply (in m²)",
    path_tables,
    ubnd=50000)
FAR = np.nansum(housing_supply, 0) / (0.25 * 1000000)
FAR_2d_sim = outexp.export_map(
    FAR, grid, geo_grid, path_plots + 'FAR_2d_sim',
    "Overall floor-area ratio",
    path_tables,
    ubnd=0.3)

hsupply_formal = housing_supply[0, :]
hsupply_formal_2d_sim = outexp.export_map(
    hsupply_formal, grid, geo_grid, path_plots + 'hsupply_formal_2d_sim',
    "Total housing supply in private formal (in m²)",
    path_tables,
    ubnd=35000)
FAR_formal = housing_supply[0, :] / (0.25 * 1000000)
FAR_formal_2d_sim = outexp.export_map(
    FAR_formal, grid, geo_grid, path_plots + 'FAR_formal_2d_sim',
    "Floor-area ratio in formal private",
    path_tables,
    ubnd=0.15)

# Pb of validation in hyper-centre is also reflected in price
sim_HFA_dens_formal = initial_state_housing_supply[0, :] / 1000000
HFA_dens_formal_2d_sim = outexp.export_map(
    sim_HFA_dens_formal, grid, geo_grid, path_plots + 'HFA_dens_formal_2d_sim',
    "Households density in formal private HFA (simulation)",
    path_tables,
    ubnd=1)
grid_formal_density_HFA[np.isnan(grid_formal_density_HFA)] = 0
data_HFA_dens_formal = grid_formal_density_HFA
HFA_dens_formal_2d_data = outexp.export_map(
    data_HFA_dens_formal, grid, geo_grid,
    path_plots + 'HFA_dens_formal_2d_data',
    "Households density in formal private HFA (data)",
    path_tables,
    ubnd=1)

hsupply_backyard = housing_supply[1, :]
hsupply_backyard_2d_sim = outexp.export_map(
    hsupply_backyard, grid, geo_grid, path_plots + 'hsupply_backyard_2d_sim',
    "Total housing supply in informal backyards (in m²)",
    path_tables,
    ubnd=30000)
FAR_backyard = housing_supply[1, :] / (0.25 * 1000000)
FAR_backyard_2d_sim = outexp.export_map(
    FAR_backyard, grid, geo_grid, path_plots + 'FAR_backyard_2d_sim',
    "Floor-area ratio in informal backyards",
    path_tables,
    ubnd=0.10)

hsupply_informal = housing_supply[2, :]
hsupply_informal_2d_sim = outexp.export_map(
    hsupply_informal, grid, geo_grid, path_plots + 'hsupply_informal_2d_sim',
    "Total housing supply in informal settlements (in m²)",
    path_tables,
    ubnd=70000)
FAR_informal = housing_supply[2, :] / (0.25 * 1000000)
FAR_informal_2d_sim = outexp.export_map(
    FAR_informal, grid, geo_grid, path_plots + 'FAR_informal_2d_sim',
    "Floor-area ratio in informal settlements",
    path_tables,
    ubnd=0.30)

hsupply_rdp = housing_supply[3, :]
hsupply_rdp_2d_sim = outexp.export_map(
    hsupply_rdp, grid, geo_grid, path_plots + 'hsupply_rdp_2d_sim',
    "Total housing supply in formal subsidized (in m²)",
    path_tables,
    ubnd=25000)
FAR_rdp = housing_supply[3, :] / (0.25 * 1000000)
FAR_rdp_2d_sim = outexp.export_map(
    FAR_rdp, grid, geo_grid, path_plots + 'FAR_rdp_2d_sim',
    "Floor-area ratio in formal subsidized",
    path_tables,
    ubnd=0.10)

# As we do not know surface of built land (just of available land),
# we need to rely on dwelling size to compute build heigth in
# formal private


# DWELLING SIZE OUTPUTS

# outexp.validation_housing_price(
#     grid, initial_state_rent, interest_rate, param, center, path_precalc_inp,
#     path_outputs + plot_repo
#     )

# # TODO: Also do breakdown for poorest across housing types and backyards
# # vs. RDP

# # test = (initial_state_households_housing_types[1, :]
# #         / initial_state_households_housing_types[3, :])
# # test = test[test > 0]

# # TODO: does it matter if price distribution is shifted to the right?

# # TODO: how can dwelling size be so big?
# outexp.plot_housing_demand(grid, center, initial_state_dwelling_size,
#                            path_precalc_inp, path_outputs + plot_repo)

# TRANSPORT DATA

#  Income net of commuting costs
netincome_poor = income_net_of_commuting_costs[0, :]
netincome_poor_2d_sim = outexp.export_map(
    netincome_poor, grid, geo_grid, path_plots + 'netincome_poor_2d_sim',
    "Estimated income net of commuting costs (poor)",
    path_tables,
    ubnd=25000, lbnd=-15000, cmap='bwr')
netincome_midpoor = income_net_of_commuting_costs[1, :]
netincome_midpoor_2d_sim = outexp.export_map(
    netincome_midpoor, grid, geo_grid, path_plots + 'netincome_midpoor_2d_sim',
    "Estimated income net of commuting costs (mid-poor)",
    path_tables,
    ubnd=70000, lbnd=-20000, cmap='bwr')
netincome_midrich = income_net_of_commuting_costs[2, :]
netincome_midrich_2d_sim = outexp.export_map(
    netincome_midrich, grid, geo_grid, path_plots + 'netincome_midrich_2d_sim',
    "Estimated income net of commuting costs (mid-rich)",
    path_tables,
    ubnd=200000, lbnd=25000)
netincome_rich = income_net_of_commuting_costs[3, :]
netincome_rich_2d_sim = outexp.export_map(
    netincome_rich, grid, geo_grid, path_plots + 'netincome_rich_2d_sim',
    "Estimated income net of commuting costs (rich)",
    path_tables,
    ubnd=850000, lbnd=250000)

(avg_income_net_of_commuting_1d
 ) = outexp.plot_income_net_of_commuting_costs(
     grid, income_net_of_commuting_costs, path_plots, path_tables)


#  Average income

avgincome_poor = cal_average_income[0, :]
avgincome_poor_2d_sim = outexp.export_map(
    avgincome_poor, grid, geo_grid, path_plots + 'avgincome_poor_2d_sim',
    "Estimated average income (poor)",
    path_tables,
    ubnd=25000, lbnd=10000)
avgincome_midpoor = cal_average_income[1, :]
avgincome_midpoor_2d_sim = outexp.export_map(
    avgincome_midpoor, grid, geo_grid, path_plots + 'avgincome_midpoor_2d_sim',
    "Estimated average income (mid-poor)",
    path_tables,
    ubnd=70000, lbnd=25000)
avgincome_midrich = cal_average_income[2, :]
avgincome_midrich_2d_sim = outexp.export_map(
    avgincome_midrich, grid, geo_grid, path_plots + 'avgincome_midrich_2d_sim',
    "Estimated average income (mid-rich)",
    path_tables,
    ubnd=200000, lbnd=100000)
avgincome_rich = cal_average_income[3, :]
avgincome_rich_2d_sim = outexp.export_map(
    avgincome_rich, grid, geo_grid, path_plots + 'avgincome_rich_2d_sim',
    "Estimated average income (rich)",
    path_tables,
    ubnd=850000, lbnd=550000)

(avg_income_1d
 ) = outexp.plot_average_income(
     grid, cal_average_income, path_plots, path_tables)

# We also conduct validation with overall average income
# Also do fit in 1D
np.seterr(divide='ignore', invalid='ignore')
overall_avg_income = (cal_average_income
                      * initial_state_household_centers
                      / np.nansum(initial_state_household_centers, 0))
overall_avg_income[np.isnan(overall_avg_income)] = 0
overall_avg_income = np.nansum(overall_avg_income, 0)
data_avg_income = data['gridAverageIncome'][0][0].squeeze()
data_avg_income[np.isnan(data_avg_income)] = 0

# The validation is pretty bad, but we should check if this is indeed
# a problem (calibration </> final output) and if validation data refers
# to same incomes (we can make the case that we only model employed households
# with two members, etc.)
avgincome_all_2d_sim = outexp.export_map(
    overall_avg_income, grid, geo_grid, path_plots + 'avgincome_all_2d_sim',
    "Estimated average income (all income groups)",
    path_tables,
    ubnd=850000)
avgincome_all_2d_data = outexp.export_map(
    data_avg_income, grid, geo_grid, path_plots + 'avgincome_all_2d_data',
    "Average income from data (all income groups)",
    path_tables,
    ubnd=850000)
overall_avg_income_valid_1d = outexp.validate_average_income(
    grid, overall_avg_income, data_avg_income,
    path_plots, path_tables)


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
jobs_total_2d.to_file(path_tables + 'jobs_total' + '.shp')

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
jobs_poor_2d.to_file(path_tables + 'jobs_poor' + '.shp')
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
jobs_midpoor_2d.to_file(path_tables + 'jobs_midpoor' + '.shp')
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
jobs_midrich_2d.to_file(path_tables + 'jobs_midrich' + '.shp')
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
jobs_rich_2d.to_file(path_tables + 'jobs_rich' + '.shp')
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

incomes_rich_2d = pd.merge(geo_TAZ, income_centers_init[0],
                        left_index=True, right_index=True)
jobs_rich_2d = pd.merge(jobs_rich_2d, selected_centers,
                        left_index=True, right_index=True)
jobs_rich_2d.to_file(path_tables + 'jobs_rich' + '.shp')
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

# TODO: ask for TAZ code dictionnary to identify OD flows for some key
# job centers (CBD, etc.)

