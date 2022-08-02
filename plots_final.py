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
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
# import copy

import inputs.parameters_and_options as inpprm
import inputs.data as inpdt
import equilibrium.functions_dynamic as eqdyn
import outputs.export_outputs as outexp
import outputs.flood_outputs as outfld
import outputs.export_outputs_floods as outval

print("Import information to be used in the simulation")


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/2. Data/'
path_precalc_inp = path_folder + '0. Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_folder + 'data_Cape_Town/Scenarios/'
path_outputs = path_code + '/4. Sorties/'
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

# TODO: makes no sense?
data = scipy.io.loadmat(path_precalc_inp + 'data.mat')['data']
data_avg_income = data['gridAverageIncome'][0][0].squeeze()
data_avg_income[np.isnan(data_avg_income)] = 0


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
 ) = eqdyn.import_scenarios(income_2011, param, grid, path_scenarios)


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
    sim_nb_households_tot, grid, geo_grid, path_plots,  'total_sim',
    "Total number of households (simulation)",
    path_tables,
    ubnd=5000)

# Note that we lack households in Mitchell's Plain
# TODO: should we correct it?
data_nb_households_tot = np.nansum(housing_types[
    ["informal_grid", "backyard_informal_grid", "formal_grid"]
    ], 1)
total_data = outexp.export_map(
    data_nb_households_tot, grid, geo_grid, path_plots,  'total_data',
    "Total number of households (data)",
    path_tables,
    ubnd=5000)

#  Per housing type
sim_nb_households_formal = initial_state_households_housing_types[0, :]
formal_sim = outexp.export_map(
    sim_nb_households_formal, grid, geo_grid, path_plots,  'formal_sim',
    "Number of households in formal private (simulation)",
    path_tables,
    ubnd=1000)
data_nb_households_formal = (housing_types["formal_grid"]
                             - initial_state_households_housing_types[3, :])
formal_data = outexp.export_map(
    data_nb_households_formal, grid, geo_grid, path_plots,  'formal_data',
    "Number of households in formal private (data)",
    path_tables,
    ubnd=1000)

sim_nb_households_backyard = initial_state_households_housing_types[1, :]
backyard_sim = outexp.export_map(
    sim_nb_households_backyard, grid, geo_grid, path_plots,  'backyard_sim',
    "Number of households in informal backyard (simulation)",
    path_tables,
    ubnd=1000)
data_nb_households_backyard = housing_types["backyard_informal_grid"]
backyard_data = outexp.export_map(
    data_nb_households_backyard, grid, geo_grid, path_plots,  'backyard_data',
    "Number of households in informal backyard (data)",
    path_tables,
    ubnd=1000)

sim_nb_households_informal = initial_state_households_housing_types[2, :]
informal_sim = outexp.export_map(
    sim_nb_households_informal, grid, geo_grid, path_plots,  'informal_sim',
    "Number of households in informal settlements (simulation)",
    path_tables,
    ubnd=3000)
data_nb_households_informal = housing_types["informal_grid"]
informal_data = outexp.export_map(
    data_nb_households_informal, grid, geo_grid, path_plots,  'informal_data',
    "Number of households in informal settlements (data)",
    path_tables,
    ubnd=3000)

data_nb_households_rdp = initial_state_households_housing_types[3, :]
rdp_sim = outexp.export_map(
    data_nb_households_rdp, grid, geo_grid, path_plots,  'rdp_sim',
    "Number of households in formal subsidized (data)",
    path_tables,
    ubnd=1800)

#  Per income group
#  NB: validation data is disaggregated from SP, hence the smooth appearance,
#  not necessarily corresponding to reality (we do not plot it)
sim_nb_households_poor = initial_state_household_centers[0, :]
poor_sim = outexp.export_map(
    sim_nb_households_poor, grid, geo_grid, path_plots,  'poor_sim',
    "Number of poor households (simulation)",
    path_tables,
    ubnd=5000)
sim_nb_households_midpoor = initial_state_household_centers[1, :]
midpoor_sim = outexp.export_map(
    sim_nb_households_midpoor, grid, geo_grid, path_plots,  'midpoor_sim',
    "Number of mid-poor households (simulation)",
    path_tables,
    ubnd=2000)
sim_nb_households_midrich = initial_state_household_centers[2, :]
midrich_sim = outexp.export_map(
    sim_nb_households_midrich, grid, geo_grid, path_plots,  'midrich_sim',
    "Number of mid-rich households (simulation)",
    path_tables,
    ubnd=1000)
sim_nb_households_rich = initial_state_household_centers[3, :]
rich_sim = outexp.export_map(
    sim_nb_households_rich, grid, geo_grid, path_plots,  'rich_sim',
    "Number of rich households (simulation)",
    path_tables,
    ubnd=500)


# %% HOUSING SUPPLY OUTPUTS

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
    hsupply_tot, grid, geo_grid, path_plots,  'hsupply_2d_sim',
    "Total housing supply (in m²)",
    path_tables,
    ubnd=50000)
FAR = np.nansum(housing_supply, 0) / (0.25 * 1000000)
FAR_2d_sim = outexp.export_map(
    FAR, grid, geo_grid, path_plots,  'FAR_2d_sim',
    "Overall floor-area ratio",
    path_tables,
    ubnd=0.3)

hsupply_formal = housing_supply[0, :]
hsupply_formal_2d_sim = outexp.export_map(
    hsupply_formal, grid, geo_grid, path_plots,  'hsupply_formal_2d_sim',
    "Total housing supply in private formal (in m²)",
    path_tables,
    ubnd=35000)
FAR_formal = housing_supply[0, :] / (0.25 * 1000000)
FAR_formal_2d_sim = outexp.export_map(
    FAR_formal, grid, geo_grid, path_plots,  'FAR_formal_2d_sim',
    "Floor-area ratio in formal private",
    path_tables,
    ubnd=0.15)

# Pb of validation in hyper-centre is also reflected in price
sim_HFA_dens_formal = initial_state_housing_supply[0, :] / 1000000
HFA_dens_formal_2d_sim = outexp.export_map(
    sim_HFA_dens_formal, grid, geo_grid, path_plots, 'HFA_dens_formal_2d_sim',
    "Households density in formal private HFA (simulation)",
    path_tables,
    ubnd=1)
grid_formal_density_HFA[np.isnan(grid_formal_density_HFA)] = 0
data_HFA_dens_formal = grid_formal_density_HFA
HFA_dens_formal_2d_data = outexp.export_map(
    data_HFA_dens_formal, grid, geo_grid,
    path_plots, 'HFA_dens_formal_2d_data',
    "Households density in formal private HFA (data)",
    path_tables,
    ubnd=1)

hsupply_backyard = housing_supply[1, :]
hsupply_backyard_2d_sim = outexp.export_map(
    hsupply_backyard, grid, geo_grid, path_plots, 'hsupply_backyard_2d_sim',
    "Total housing supply in informal backyards (in m²)",
    path_tables,
    ubnd=30000)
FAR_backyard = housing_supply[1, :] / (0.25 * 1000000)
FAR_backyard_2d_sim = outexp.export_map(
    FAR_backyard, grid, geo_grid, path_plots, 'FAR_backyard_2d_sim',
    "Floor-area ratio in informal backyards",
    path_tables,
    ubnd=0.10)

hsupply_informal = housing_supply[2, :]
hsupply_informal_2d_sim = outexp.export_map(
    hsupply_informal, grid, geo_grid, path_plots, 'hsupply_informal_2d_sim',
    "Total housing supply in informal settlements (in m²)",
    path_tables,
    ubnd=70000)
FAR_informal = housing_supply[2, :] / (0.25 * 1000000)
FAR_informal_2d_sim = outexp.export_map(
    FAR_informal, grid, geo_grid, path_plots, 'FAR_informal_2d_sim',
    "Floor-area ratio in informal settlements",
    path_tables,
    ubnd=0.30)

hsupply_rdp = housing_supply[3, :]
hsupply_rdp_2d_sim = outexp.export_map(
    hsupply_rdp, grid, geo_grid, path_plots, 'hsupply_rdp_2d_sim',
    "Total housing supply in formal subsidized (in m²)",
    path_tables,
    ubnd=25000)
FAR_rdp = housing_supply[3, :] / (0.25 * 1000000)
FAR_rdp_2d_sim = outexp.export_map(
    FAR_rdp, grid, geo_grid, path_plots, 'FAR_rdp_2d_sim',
    "Floor-area ratio in formal subsidized",
    path_tables,
    ubnd=0.10)

# As we do not know surface of built land (just of available land),
# we need to rely on dwelling size to compute build heigth in
# formal private


# %% HOUSING PRICE OUTPUTS

# First in one dimension
# housing_price_1d, data_land_price = outexp.validation_housing_price(
#     grid, initial_state_rent, interest_rate, param, center,
#     housing_types_sp, data_sp, path_plots, path_tables,
#     land_price=1)
# housing_price_1d, data_housing_price = outexp.validation_housing_price(
#     grid, initial_state_rent, interest_rate, param, center,
#     housing_types_sp, data_sp, path_plots, path_tables,
#     land_price=0)

land_price_1d, data_land_price = outexp.validation_housing_price_test(
    grid, initial_state_rent, initial_state_households_housing_types,
    interest_rate, param, center,
    housing_types_sp, data_sp, path_plots, path_tables,
    land_price=1)
housing_price_1d, data_housing_price = outexp.validation_housing_price_test(
    grid, initial_state_rent, initial_state_households_housing_types,
    interest_rate, param, center,
    housing_types_sp, data_sp, path_plots, path_tables,
    land_price=0)

# data_land_price_copy = data_land_price.copy()
# data_housing_price_copy = data_housing_price.copy()
# data_land_price_copy[np.isnan(data_land_price)] = 0
# data_housing_price_copy[np.isnan(data_housing_price)] = 0

# Then in two dimensions

# TODO: why don't we manage to reproduce very high prices in center?
# Has to do with amenity map?
rent_formal_simul = initial_state_rent[0, :].copy()
housing_price_formal_2d_sim = outexp.export_map(
    rent_formal_simul, grid, geo_grid, path_plots,  'rent_formal_2d_sim',
    "Simulated average housing rents per location (private formal)",
    path_tables,
    ubnd=4000)
rent_backyard_simul = initial_state_rent[1, :].copy()
housing_price_backyard_2d_sim = outexp.export_map(
    rent_backyard_simul, grid, geo_grid, path_plots,  'rent_backyard_2d_sim',
    "Simulated average housing rents per location (informal backyards)",
    path_tables,
    ubnd=2500)
rent_informal_simul = initial_state_rent[2, :].copy()
housing_price_informal_2d_sim = outexp.export_map(
    rent_informal_simul, grid, geo_grid, path_plots,  'rent_informal_2d_sim',
    "Simulated average housing rents per location (informal settlements)",
    path_tables,
    ubnd=2500)

land_rent = (
    (initial_state_rent[0:3, :] * param["coeff_A"])
    ** (1 / param["coeff_a"])
    * param["coeff_a"]
    * (param["coeff_b"] / (interest_rate + param["depreciation_rate"]))
    ** (param["coeff_b"] / param["coeff_a"])
    / interest_rate
    )
landrent_formal_simul = land_rent[0, :].copy()
land_price_formal_2d_sim = outexp.export_map(
    landrent_formal_simul, grid, geo_grid,
    path_plots, 'landrent_formal_2d_sim',
    "Simulated average land rents per location (private formal)",
    path_tables,
    ubnd=15000)
landrent_backyard_simul = land_rent[1, :].copy()
land_price_backyard_2d_sim = outexp.export_map(
    landrent_backyard_simul, grid, geo_grid,
    path_plots, 'landrent_backyard_2d_sim',
    "Simulated average land rents per location (informal backyards)",
    path_tables,
    ubnd=10000)
landrent_informal_simul = land_rent[2, :].copy()
land_price_informal_2d_sim = outexp.export_map(
    landrent_informal_simul, grid, geo_grid,
    path_plots, 'landrent_informal_2d_sim',
    "Simulated average land rents per location (informal settlements)",
    path_tables,
    ubnd=10000)

# grid_intersect = pd.read_csv(
#     path_data + 'grid_SP_intersect.csv', sep=';')
# rent_formal_data = inpdt.gen_small_areas_to_grid(
#     grid, grid_intersect, data_housing_price_copy,
#     data_sp["sp_code"], 'SP')

# housing_price_2d_data = outexp.export_map(
#     rent_formal_data, grid, geo_grid, path_plots,  'rent_formal_2d_data',
#     "Data average housing rents per location (private formal)",
#     path_tables,
#     ubnd=2500)

# %% DWELLING SIZE OUTPUTS

# Note that we start getting a lot of nan values around 30km
# Could this explain the low number of households in Mitchell's Plain
# in spite of the housing supply
# TODO: how should we interpret such high values
dwelling_size_1d = outexp.plot_housing_demand(
    grid, center, initial_state_dwelling_size,
    initial_state_households_housing_types,
    housing_types_sp, data_sp,
    path_plots, path_tables)

formal_dwelling_size = initial_state_dwelling_size[0, :]
dwelling_size_2d = outexp.export_map(
    formal_dwelling_size, grid, geo_grid,
    path_plots, 'formal_dwellingsize_2d_sim',
    "Simulated average dwelling size per location (formal private)",
    path_tables,
    ubnd=300)


# %% TRANSPORT OUTPUTS

#  Income net of commuting costs
netincome_poor = income_net_of_commuting_costs[0, :]
netincome_poor_2d_sim = outexp.export_map(
    netincome_poor, grid, geo_grid, path_plots,  'netincome_poor_2d_sim',
    "Estimated income net of commuting costs (poor)",
    path_tables,
    ubnd=25000, lbnd=-15000, cmap='bwr')
netincome_midpoor = income_net_of_commuting_costs[1, :]
netincome_midpoor_2d_sim = outexp.export_map(
    netincome_midpoor, grid, geo_grid, path_plots,  'netincome_midpoor_2d_sim',
    "Estimated income net of commuting costs (mid-poor)",
    path_tables,
    ubnd=70000, lbnd=-20000, cmap='bwr')
netincome_midrich = income_net_of_commuting_costs[2, :]
netincome_midrich_2d_sim = outexp.export_map(
    netincome_midrich, grid, geo_grid, path_plots,  'netincome_midrich_2d_sim',
    "Estimated income net of commuting costs (mid-rich)",
    path_tables,
    ubnd=200000, lbnd=25000)
netincome_rich = income_net_of_commuting_costs[3, :]
netincome_rich_2d_sim = outexp.export_map(
    netincome_rich, grid, geo_grid, path_plots,  'netincome_rich_2d_sim',
    "Estimated income net of commuting costs (rich)",
    path_tables,
    ubnd=850000, lbnd=250000)

(avg_income_net_of_commuting_1d
 ) = outexp.plot_income_net_of_commuting_costs(
     grid, income_net_of_commuting_costs, path_plots, path_tables)


#  Average income

avgincome_poor = cal_average_income[0, :]
avgincome_poor_2d_sim = outexp.export_map(
    avgincome_poor, grid, geo_grid, path_plots,  'avgincome_poor_2d_sim',
    "Estimated average income (poor)",
    path_tables,
    ubnd=25000, lbnd=10000)
avgincome_midpoor = cal_average_income[1, :]
avgincome_midpoor_2d_sim = outexp.export_map(
    avgincome_midpoor, grid, geo_grid, path_plots,  'avgincome_midpoor_2d_sim',
    "Estimated average income (mid-poor)",
    path_tables,
    ubnd=70000, lbnd=25000)
avgincome_midrich = cal_average_income[2, :]
avgincome_midrich_2d_sim = outexp.export_map(
    avgincome_midrich, grid, geo_grid, path_plots,  'avgincome_midrich_2d_sim',
    "Estimated average income (mid-rich)",
    path_tables,
    ubnd=200000, lbnd=100000)
avgincome_rich = cal_average_income[3, :]
avgincome_rich_2d_sim = outexp.export_map(
    avgincome_rich, grid, geo_grid, path_plots,  'avgincome_rich_2d_sim',
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

# The validation is pretty bad, but we should check if this is indeed
# a problem (calibration </> final output) and if validation data refers
# to same incomes (we can make the case that we only model employed households
# with two members, etc.)
avgincome_all_2d_sim = outexp.export_map(
    overall_avg_income, grid, geo_grid, path_plots,  'avgincome_all_2d_sim',
    "Estimated average income (all income groups)",
    path_tables,
    ubnd=850000)
avgincome_all_2d_data = outexp.export_map(
    data_avg_income, grid, geo_grid, path_plots,  'avgincome_all_2d_data',
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
income_centers_2d.to_file(path_tables + 'income_centers_2d' + '.shp')

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


# %% FLOOD OUPUTS

# TODO: first do input flood maps in 2D, to be superimposed with
# some previous maps
# Then also compute damages, welfare impacts, aggregate effects, etc.
# Finally, need to do dynamics and comparisons across scenarios, LVC, etc.

# TODO: also need to add options for the code to run seamlessly across
# scenarios

# TODO: recompute floods with maximum bath-tub perspective (also coastal
# matching)

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

# Also for income groups and across the two?
# NB: evolution is not necessarily monotonous on the short run because of
# some decreasing flood depths (never proportion of flood-prone area)
# TODO: is it normal?

stats_fluvialu_per_housing_data = outfld.compute_stats_per_housing_type(
    fluvialu_floods, path_floods, data_nb_households_formal,
    data_nb_households_rdp, data_nb_households_informal,
    data_nb_households_backyard, path_tables, 'fluvialu_data')
stats_fluvialu_per_housing_sim = outfld.compute_stats_per_housing_type(
    fluvialu_floods, path_floods, sim_nb_households_formal,
    data_nb_households_rdp,
    sim_nb_households_informal,
    sim_nb_households_backyard,
    path_tables, 'fluvialu_sim')
outval.validation_flood(
    stats_fluvialu_per_housing_data, stats_fluvialu_per_housing_sim,
    'Data', 'Simul', 'fluvialu', path_plots)

stats_fluviald_per_housing_data = outfld.compute_stats_per_housing_type(
    fluviald_floods, path_floods, data_nb_households_formal,
    data_nb_households_rdp, data_nb_households_informal,
    data_nb_households_backyard, path_tables, 'fluviald_data')
stats_fluviald_per_housing_sim = outfld.compute_stats_per_housing_type(
    fluviald_floods, path_floods, sim_nb_households_formal,
    data_nb_households_rdp,
    sim_nb_households_informal,
    sim_nb_households_backyard,
    path_tables, 'fluviald_sim')
outval.validation_flood(
    stats_fluviald_per_housing_data, stats_fluviald_per_housing_sim,
    'Data', 'Simul', 'fluviald', path_plots)

stats_pluvial_per_housing_data = outfld.compute_stats_per_housing_type(
    pluvial_floods, path_floods, data_nb_households_formal,
    data_nb_households_rdp, data_nb_households_informal,
    data_nb_households_backyard, path_tables, 'pluvial_data')
stats_pluvial_per_housing_sim = outfld.compute_stats_per_housing_type(
    pluvial_floods, path_floods, sim_nb_households_formal,
    data_nb_households_rdp,
    sim_nb_households_informal,
    sim_nb_households_backyard,
    path_tables, 'pluvial_sim')
outval.validation_flood(
    stats_pluvial_per_housing_data, stats_pluvial_per_housing_sim,
    'Data', 'Simul', 'pluvial', path_plots)

stats_coastal_per_housing_data = outfld.compute_stats_per_housing_type(
    coastal_floods, path_floods, data_nb_households_formal,
    data_nb_households_rdp, data_nb_households_informal,
    data_nb_households_backyard, path_tables, 'coastal_data')
stats_coastal_per_housing_sim = outfld.compute_stats_per_housing_type(
    coastal_floods, path_floods, sim_nb_households_formal,
    data_nb_households_rdp,
    sim_nb_households_informal,
    sim_nb_households_backyard,
    path_tables, 'coastal_sim')
outval.validation_flood_coastal(
    stats_coastal_per_housing_data, stats_coastal_per_housing_sim,
    'Data', 'Simul', 'coastal', path_plots)

# TODO: could add validation data if needed

fluviald_floods_dict = outfld.create_flood_dict(
    fluviald_floods, path_floods, path_tables,
    sim_nb_households_poor, sim_nb_households_midpoor,
    sim_nb_households_midrich, sim_nb_households_rich)
fluvialu_floods_dict = outfld.create_flood_dict(
    fluvialu_floods, path_floods, path_tables,
    sim_nb_households_poor, sim_nb_households_midpoor,
    sim_nb_households_midrich, sim_nb_households_rich)
pluvial_floods_dict = outfld.create_flood_dict(
    pluvial_floods, path_floods, path_tables,
    sim_nb_households_poor, sim_nb_households_midpoor,
    sim_nb_households_midrich, sim_nb_households_rich)
coastal_floods_dict = outfld.create_flood_dict(
    coastal_floods, path_floods, path_tables,
    sim_nb_households_poor, sim_nb_households_midpoor,
    sim_nb_households_midrich, sim_nb_households_rich)

barWidth = 0.1
transparency = [1, 0.5, 0.25]

outval.plot_flood_severity_distrib(barWidth, transparency,
                                   fluviald_floods_dict, 'FD',
                                   path_plots, ylim=15000)
outval.plot_flood_severity_distrib(barWidth, transparency,
                                   fluvialu_floods_dict, 'FU',
                                   path_plots, ylim=15000)
outval.plot_flood_severity_distrib(barWidth, transparency,
                                   pluvial_floods_dict, 'P',
                                   path_plots, ylim=90000)
outval.plot_flood_severity_distrib(barWidth, transparency,
                                   coastal_floods_dict, 'C_MERITDEM_1',
                                   path_plots, ylim=1000)


# %% FLOOD DAMAGES

# TODO: check MAUP
# We get damages per housing type for one representative household!

content_cost = outfld.compute_content_cost(
    initial_state_household_centers, initial_state_housing_supply,
    income_net_of_commuting_costs, param,
    fraction_capital_destroyed, initial_state_rent,
    initial_state_dwelling_size, interest_rate)

# TODO: construction_coeff will need to be updated along other parameters
# in simulations
# NOTE THAT CAPITAL IS IN MONETARY VALUES
formal_structure_cost = outfld.compute_formal_structure_cost_method2(
        initial_state_rent, param, interest_rate, coeff_land,
        initial_state_households_housing_types, param["coeff_A"])

# Then we run the aggregate tables

fluviald_damages_data = outfld.compute_damages(
    fluviald_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'fluviald_data')
fluviald_damages_sim = outfld.compute_damages(
    fluviald_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'fluviald_sim')

fluvialu_damages_data = outfld.compute_damages(
    fluvialu_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'fluvialu_data')
fluvialu_damages_sim = outfld.compute_damages(
    fluvialu_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'fluvialu_sim')

pluvial_damages_data = outfld.compute_damages(
    pluvial_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'pluvial_data')
pluvial_damages_sim = outfld.compute_damages(
    pluvial_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'pluvial_sim')

coastal_damages_data = outfld.compute_damages(
    coastal_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'coastal_data')
coastal_damages_sim = outfld.compute_damages(
    coastal_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'coastal_sim')

# We get aggregate graphs

outval.plot_damages(
    fluviald_damages_sim, fluviald_damages_data,
    path_plots, 'fluviald', options)
outval.plot_damages(
    fluvialu_damages_sim, fluvialu_damages_data,
    path_plots, 'fluvialu', options)
outval.plot_damages(
    pluvial_damages_sim, pluvial_damages_data,
    path_plots, 'pluvial', options)
outval.plot_damages(
    coastal_damages_sim, coastal_damages_data,
    path_plots, 'coastal', options)


# Now in two dimensions

fluviald_damages_2d_data = outfld.compute_damages_2d(
    fluviald_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'fluviald_data')
fluviald_damages_2d_sim = outfld.compute_damages_2d(
    fluviald_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'fluviald_sim')

fluvialu_damages_2d_data = outfld.compute_damages_2d(
    fluvialu_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'fluvialu_data')
fluvialu_damages_2d_sim = outfld.compute_damages_2d(
    fluvialu_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'fluvialu_sim')

pluvial_damages_2d_data = outfld.compute_damages_2d(
    pluvial_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'pluvial_data')
pluvial_damages_2d_sim = outfld.compute_damages_2d(
    pluvial_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'pluvial_sim')

coastal_damages_2d_data = outfld.compute_damages_2d(
    coastal_floods, path_floods, param, content_cost,
    data_nb_households_formal, data_nb_households_rdp,
    data_nb_households_informal, data_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'coastal_data')
coastal_damages_2d_sim = outfld.compute_damages_2d(
    coastal_floods, path_floods, param, content_cost,
    sim_nb_households_formal, data_nb_households_rdp,
    sim_nb_households_informal, sim_nb_households_backyard,
    initial_state_dwelling_size, formal_structure_cost, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a, options,
    spline_inflation, year_temp, path_tables, 'coastal_sim')

# Hence the maps and shapefiles
# TODO: should we annualize? Problem is that we need to loop over dfs
# Make ndarray!
fluviald_damages_2d_data_stacked = np.stack(
    [df for df in fluviald_damages_2d_data.values()])
fluviald_formal_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluviald_formal_structure_2d_data[j] = outfld.annualize_damages(
        fluviald_damages_2d_data_stacked[:, j, 0],
        'fluviald', 'formal', options)
fluviald_subsidized_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluviald_subsidized_structure_2d_data[j] = outfld.annualize_damages(
        fluviald_damages_2d_data_stacked[:, j, 1],
        'fluviald', 'subsidized', options)
fluviald_informal_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluviald_informal_structure_2d_data[j] = outfld.annualize_damages(
        fluviald_damages_2d_data_stacked[:, j, 2],
        'fluviald', 'informal', options)
fluviald_backyard_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluviald_backyard_structure_2d_data[j] = outfld.annualize_damages(
        fluviald_damages_2d_data_stacked[:, j, 3],
        'fluviald', 'backyard', options)
fluviald_formal_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluviald_formal_content_2d_data[j] = outfld.annualize_damages(
        fluviald_damages_2d_data_stacked[:, j, 4],
        'fluviald', 'formal', options)
fluviald_subsidized_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluviald_subsidized_content_2d_data[j] = outfld.annualize_damages(
        fluviald_damages_2d_data_stacked[:, j, 5],
        'fluviald', 'subsidized', options)
fluviald_informal_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluviald_informal_content_2d_data[j] = outfld.annualize_damages(
        fluviald_damages_2d_data_stacked[:, j, 6],
        'fluviald', 'informal', options)
fluviald_backyard_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluviald_backyard_content_2d_data[j] = outfld.annualize_damages(
        fluviald_damages_2d_data_stacked[:, j, 7],
        'fluviald', 'backyard', options)

fluvialu_damages_2d_data_stacked = np.stack(
    [df for df in fluvialu_damages_2d_data.values()])
fluvialu_formal_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_formal_structure_2d_data[j] = outfld.annualize_damages(
        fluvialu_damages_2d_data_stacked[:, j, 0],
        'fluvialu', 'formal', options)
fluvialu_subsidized_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_subsidized_structure_2d_data[j] = outfld.annualize_damages(
        fluvialu_damages_2d_data_stacked[:, j, 1],
        'fluvialu', 'subsidized', options)
fluvialu_informal_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_informal_structure_2d_data[j] = outfld.annualize_damages(
        fluvialu_damages_2d_data_stacked[:, j, 2],
        'fluvialu', 'informal', options)
fluvialu_backyard_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_backyard_structure_2d_data[j] = outfld.annualize_damages(
        fluvialu_damages_2d_data_stacked[:, j, 3],
        'fluvialu', 'backyard', options)
fluvialu_formal_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_formal_content_2d_data[j] = outfld.annualize_damages(
        fluvialu_damages_2d_data_stacked[:, j, 4],
        'fluvialu', 'formal', options)
fluvialu_subsidized_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_subsidized_content_2d_data[j] = outfld.annualize_damages(
        fluvialu_damages_2d_data_stacked[:, j, 5],
        'fluvialu', 'subsidized', options)
fluvialu_informal_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_informal_content_2d_data[j] = outfld.annualize_damages(
        fluvialu_damages_2d_data_stacked[:, j, 6],
        'fluvialu', 'informal', options)
fluvialu_backyard_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_backyard_content_2d_data[j] = outfld.annualize_damages(
        fluvialu_damages_2d_data_stacked[:, j, 7],
        'fluvialu', 'backyard', options)

pluvial_damages_2d_data_stacked = np.stack(
    [df for df in pluvial_damages_2d_data.values()])
pluvial_formal_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    pluvial_formal_structure_2d_data[j] = outfld.annualize_damages(
        pluvial_damages_2d_data_stacked[:, j, 0],
        'pluvial', 'formal', options)
pluvial_subsidized_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    pluvial_subsidized_structure_2d_data[j] = outfld.annualize_damages(
        pluvial_damages_2d_data_stacked[:, j, 1],
        'pluvial', 'subsidized', options)
pluvial_informal_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    pluvial_informal_structure_2d_data[j] = outfld.annualize_damages(
        pluvial_damages_2d_data_stacked[:, j, 2],
        'pluvial', 'informal', options)
pluvial_backyard_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    pluvial_backyard_structure_2d_data[j] = outfld.annualize_damages(
        pluvial_damages_2d_data_stacked[:, j, 3],
        'pluvial', 'backyard', options)
pluvial_formal_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    pluvial_formal_content_2d_data[j] = outfld.annualize_damages(
        pluvial_damages_2d_data_stacked[:, j, 4],
        'pluvial', 'formal', options)
pluvial_subsidized_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    pluvial_subsidized_content_2d_data[j] = outfld.annualize_damages(
        pluvial_damages_2d_data_stacked[:, j, 5],
        'pluvial', 'subsidized', options)
pluvial_informal_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    pluvial_informal_content_2d_data[j] = outfld.annualize_damages(
        pluvial_damages_2d_data_stacked[:, j, 6],
        'pluvial', 'informal', options)
pluvial_backyard_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    pluvial_backyard_content_2d_data[j] = outfld.annualize_damages(
        pluvial_damages_2d_data_stacked[:, j, 7],
        'pluvial', 'backyard', options)

coastal_damages_2d_data_stacked = np.stack(
    [df for df in coastal_damages_2d_data.values()])
coastal_formal_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    coastal_formal_structure_2d_data[j] = outfld.annualize_damages(
        coastal_damages_2d_data_stacked[:, j, 0],
        'coastal', 'formal', options)
coastal_subsidized_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    coastal_subsidized_structure_2d_data[j] = outfld.annualize_damages(
        coastal_damages_2d_data_stacked[:, j, 1],
        'coastal', 'subsidized', options)
coastal_informal_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    coastal_informal_structure_2d_data[j] = outfld.annualize_damages(
        coastal_damages_2d_data_stacked[:, j, 2],
        'coastal', 'informal', options)
coastal_backyard_structure_2d_data = np.zeros(24014)
for j in np.arange(24014):
    coastal_backyard_structure_2d_data[j] = outfld.annualize_damages(
        coastal_damages_2d_data_stacked[:, j, 3],
        'coastal', 'backyard', options)
coastal_formal_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    coastal_formal_content_2d_data[j] = outfld.annualize_damages(
        coastal_damages_2d_data_stacked[:, j, 4],
        'coastal', 'formal', options)
coastal_subsidized_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    coastal_subsidized_content_2d_data[j] = outfld.annualize_damages(
        coastal_damages_2d_data_stacked[:, j, 5],
        'coastal', 'subsidized', options)
coastal_informal_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    coastal_informal_content_2d_data[j] = outfld.annualize_damages(
        coastal_damages_2d_data_stacked[:, j, 6],
        'coastal', 'informal', options)
coastal_backyard_content_2d_data = np.zeros(24014)
for j in np.arange(24014):
    coastal_backyard_content_2d_data[j] = outfld.annualize_damages(
        coastal_damages_2d_data_stacked[:, j, 7],
        'coastal', 'backyard', options)


fluviald_damages_2d_sim_stacked = np.stack(
    [df for df in fluviald_damages_2d_sim.values()])
fluviald_formal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluviald_formal_structure_2d_sim[j] = outfld.annualize_damages(
        fluviald_damages_2d_sim_stacked[:, j, 0],
        'fluviald', 'formal', options)
fluviald_subsidized_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluviald_subsidized_structure_2d_sim[j] = outfld.annualize_damages(
        fluviald_damages_2d_sim_stacked[:, j, 1],
        'fluviald', 'subsidized', options)
fluviald_informal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluviald_informal_structure_2d_sim[j] = outfld.annualize_damages(
        fluviald_damages_2d_sim_stacked[:, j, 2],
        'fluviald', 'informal', options)
fluviald_backyard_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluviald_backyard_structure_2d_sim[j] = outfld.annualize_damages(
        fluviald_damages_2d_sim_stacked[:, j, 3],
        'fluviald', 'backyard', options)
fluviald_formal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluviald_formal_content_2d_sim[j] = outfld.annualize_damages(
        fluviald_damages_2d_sim_stacked[:, j, 4],
        'fluviald', 'formal', options)
fluviald_subsidized_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluviald_subsidized_content_2d_sim[j] = outfld.annualize_damages(
        fluviald_damages_2d_sim_stacked[:, j, 5],
        'fluviald', 'subsidized', options)
fluviald_informal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluviald_informal_content_2d_sim[j] = outfld.annualize_damages(
        fluviald_damages_2d_sim_stacked[:, j, 6],
        'fluviald', 'informal', options)
fluviald_backyard_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluviald_backyard_content_2d_sim[j] = outfld.annualize_damages(
        fluviald_damages_2d_sim_stacked[:, j, 7],
        'fluviald', 'backyard', options)

fluvialu_damages_2d_sim_stacked = np.stack(
    [df for df in fluvialu_damages_2d_sim.values()])
fluvialu_formal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_formal_structure_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 0],
        'fluvialu', 'formal', options)
fluvialu_subsidized_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_subsidized_structure_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 1],
        'fluvialu', 'subsidized', options)
fluvialu_informal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_informal_structure_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 2],
        'fluvialu', 'informal', options)
fluvialu_backyard_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_backyard_structure_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 3],
        'fluvialu', 'backyard', options)
fluvialu_formal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_formal_content_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 4],
        'fluvialu', 'formal', options)
fluvialu_subsidized_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_subsidized_content_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 5],
        'fluvialu', 'subsidized', options)
fluvialu_informal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_informal_content_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 6],
        'fluvialu', 'informal', options)
fluvialu_backyard_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    fluvialu_backyard_content_2d_sim[j] = outfld.annualize_damages(
        fluvialu_damages_2d_sim_stacked[:, j, 7],
        'fluvialu', 'backyard', options)

pluvial_damages_2d_sim_stacked = np.stack(
    [df for df in pluvial_damages_2d_sim.values()])
pluvial_formal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_formal_structure_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 0],
        'pluvial', 'formal', options)
pluvial_subsidized_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_subsidized_structure_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 1],
        'pluvial', 'subsidized', options)
pluvial_informal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_informal_structure_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 2],
        'pluvial', 'informal', options)
pluvial_backyard_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_backyard_structure_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 3],
        'pluvial', 'backyard', options)
pluvial_formal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_formal_content_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 4],
        'pluvial', 'formal', options)
pluvial_subsidized_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_subsidized_content_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 5],
        'pluvial', 'subsidized', options)
pluvial_informal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_informal_content_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 6],
        'pluvial', 'informal', options)
pluvial_backyard_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    pluvial_backyard_content_2d_sim[j] = outfld.annualize_damages(
        pluvial_damages_2d_sim_stacked[:, j, 7],
        'pluvial', 'backyard', options)

coastal_damages_2d_sim_stacked = np.stack(
    [df for df in coastal_damages_2d_sim.values()])
coastal_formal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_formal_structure_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 0],
        'coastal', 'formal', options)
coastal_subsidized_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_subsidized_structure_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 1],
        'coastal', 'subsidized', options)
coastal_informal_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_informal_structure_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 2],
        'coastal', 'informal', options)
coastal_backyard_structure_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_backyard_structure_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 3],
        'coastal', 'backyard', options)
coastal_formal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_formal_content_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 4],
        'coastal', 'formal', options)
coastal_subsidized_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_subsidized_content_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 5],
        'coastal', 'subsidized', options)
coastal_informal_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_informal_content_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 6],
        'coastal', 'informal', options)
coastal_backyard_content_2d_sim = np.zeros(24014)
for j in np.arange(24014):
    coastal_backyard_content_2d_sim[j] = outfld.annualize_damages(
        coastal_damages_2d_sim_stacked[:, j, 7],
        'coastal', 'backyard', options)

list_annualized_2d_damages = [
    fluviald_backyard_structure_2d_data, fluviald_backyard_structure_2d_sim,
    fluviald_backyard_content_2d_data, fluviald_backyard_content_2d_sim,
    fluviald_subsidized_structure_2d_data,
    fluviald_subsidized_structure_2d_sim,
    fluviald_subsidized_content_2d_data, fluviald_subsidized_content_2d_sim,
    fluviald_informal_structure_2d_data, fluviald_informal_structure_2d_sim,
    fluviald_informal_content_2d_data, fluviald_informal_content_2d_sim,
    fluviald_formal_structure_2d_data, fluviald_formal_structure_2d_sim,
    fluviald_formal_content_2d_data, fluviald_formal_content_2d_sim,
    fluvialu_backyard_structure_2d_data, fluvialu_backyard_structure_2d_sim,
    fluvialu_backyard_content_2d_data, fluvialu_backyard_content_2d_sim,
    fluvialu_subsidized_structure_2d_data,
    fluvialu_subsidized_structure_2d_sim,
    fluvialu_subsidized_content_2d_data, fluvialu_subsidized_content_2d_sim,
    fluvialu_informal_structure_2d_data, fluvialu_informal_structure_2d_sim,
    fluvialu_informal_content_2d_data, fluvialu_informal_content_2d_sim,
    fluvialu_formal_structure_2d_data, fluvialu_formal_structure_2d_sim,
    fluvialu_formal_content_2d_data, fluvialu_formal_content_2d_sim,
    pluvial_backyard_structure_2d_data, pluvial_backyard_structure_2d_sim,
    pluvial_backyard_content_2d_data, pluvial_backyard_content_2d_sim,
    pluvial_subsidized_structure_2d_data, pluvial_subsidized_structure_2d_sim,
    pluvial_subsidized_content_2d_data, pluvial_subsidized_content_2d_sim,
    pluvial_informal_structure_2d_data, pluvial_informal_structure_2d_sim,
    pluvial_informal_content_2d_data, pluvial_informal_content_2d_sim,
    pluvial_formal_structure_2d_data, pluvial_formal_structure_2d_sim,
    pluvial_formal_content_2d_data, pluvial_formal_content_2d_sim,
    coastal_backyard_structure_2d_data, coastal_backyard_structure_2d_sim,
    coastal_backyard_content_2d_data, coastal_backyard_content_2d_sim,
    coastal_subsidized_structure_2d_data, coastal_subsidized_structure_2d_sim,
    coastal_subsidized_content_2d_data, coastal_subsidized_content_2d_sim,
    coastal_informal_structure_2d_data, coastal_informal_structure_2d_sim,
    coastal_informal_content_2d_data, coastal_informal_content_2d_sim,
    coastal_formal_structure_2d_data, coastal_formal_structure_2d_sim,
    coastal_formal_content_2d_data, coastal_formal_content_2d_sim]

list_annualized_2d_damages_formal = [
    fluviald_formal_structure_2d_data, fluviald_formal_structure_2d_sim,
    fluviald_formal_content_2d_data, fluviald_formal_content_2d_sim,
    fluvialu_formal_structure_2d_data, fluvialu_formal_structure_2d_sim,
    fluvialu_formal_content_2d_data, fluvialu_formal_content_2d_sim,
    pluvial_formal_structure_2d_data, pluvial_formal_structure_2d_sim,
    pluvial_formal_content_2d_data, pluvial_formal_content_2d_sim,
    coastal_formal_structure_2d_data, coastal_formal_structure_2d_sim,
    coastal_formal_content_2d_data, coastal_formal_content_2d_sim]

list_annualized_2d_damages_informal = [
    fluviald_informal_structure_2d_data, fluviald_informal_structure_2d_sim,
    fluviald_informal_content_2d_data, fluviald_informal_content_2d_sim,
    fluvialu_informal_structure_2d_data, fluvialu_informal_structure_2d_sim,
    fluvialu_informal_content_2d_data, fluvialu_informal_content_2d_sim,
    pluvial_informal_structure_2d_data, pluvial_informal_structure_2d_sim,
    pluvial_informal_content_2d_data, pluvial_informal_content_2d_sim,
    coastal_informal_structure_2d_data, coastal_informal_structure_2d_sim,
    coastal_informal_content_2d_data, coastal_informal_content_2d_sim]

list_annualized_2d_damages_backyard = [
    fluviald_backyard_structure_2d_data, fluviald_backyard_structure_2d_sim,
    fluviald_backyard_content_2d_data, fluviald_backyard_content_2d_sim,
    fluvialu_backyard_structure_2d_data, fluvialu_backyard_structure_2d_sim,
    fluvialu_backyard_content_2d_data, fluvialu_backyard_content_2d_sim,
    pluvial_backyard_structure_2d_data, pluvial_backyard_structure_2d_sim,
    pluvial_backyard_content_2d_data, pluvial_backyard_content_2d_sim,
    coastal_backyard_structure_2d_data, coastal_backyard_structure_2d_sim,
    coastal_backyard_content_2d_data, coastal_backyard_content_2d_sim]

list_annualized_2d_damages_subsidized = [
    fluviald_subsidized_structure_2d_data,
    fluviald_subsidized_structure_2d_sim,
    fluviald_subsidized_content_2d_data, fluviald_subsidized_content_2d_sim,
    fluvialu_subsidized_structure_2d_data,
    fluvialu_subsidized_structure_2d_sim,
    fluvialu_subsidized_content_2d_data, fluvialu_subsidized_content_2d_sim,
    pluvial_subsidized_structure_2d_data, pluvial_subsidized_structure_2d_sim,
    pluvial_subsidized_content_2d_data, pluvial_subsidized_content_2d_sim,
    coastal_subsidized_structure_2d_data, coastal_subsidized_structure_2d_sim,
    coastal_subsidized_content_2d_data, coastal_subsidized_content_2d_sim]

# NB: need to retrieve name of item
for item in list_annualized_2d_damages:
    try:
        outexp.export_map(item, grid, geo_grid,
                          path_plots, outexp.retrieve_name(item, -1), "",
                          path_tables,
                          ubnd=np.quantile(item[item > 0], 0.9),
                          lbnd=np.min(item[item > 0]))
    except IndexError:
        pass

# We could plot fraction of capital destroyed separately for each
# flood type, but it would be similar due to bath-tub model
# NB: note that content damage function is the same for all housing types
for col in fraction_capital_destroyed.columns:
    value = fraction_capital_destroyed[col]
    outexp.export_map(value, grid, geo_grid,
                      path_plots, col + '_fract_K_destroyed', "",
                      path_tables,
                      ubnd=1)

# Graphs with share of annual income destroyed (by income group
# and return period)?
# TODO: also average/distribution graph?

# TODO: are duplicates a problem?
# NB: note that share can be bigger than 1 (which is just a cap)

selected_net_income_formal = np.empty(24014)
cond = np.argmax(initial_state_households[0, :, :], axis=0)
for j in np.arange(24014):
    selected_net_income_formal[j] = (
        income_net_of_commuting_costs[cond[j], j])

for item in list_annualized_2d_damages_formal:
    new_item = item / selected_net_income_formal
    try:
        outexp.export_map(new_item, grid, geo_grid,
                          path_plots,
                          outexp.retrieve_name(item, -1) + '_shareinc', "",
                          path_tables,
                          ubnd=1)
    except IndexError:
        pass

selected_net_income_rdp = np.empty(24014)
cond = np.argmax(initial_state_households[3, :, :], axis=0)
for j in np.arange(24014):
    selected_net_income_rdp[j] = (
        income_net_of_commuting_costs[cond[j], j])

for item in list_annualized_2d_damages_subsidized:
    new_item = item / selected_net_income_rdp
    try:
        outexp.export_map(new_item, grid, geo_grid,
                          path_plots,
                          outexp.retrieve_name(item, -1) + '_shareinc', "",
                          path_tables,
                          ubnd=1)
    except IndexError:
        pass

selected_net_income_backyard = np.empty(24014)
cond = np.argmax(initial_state_households[1, :, :], axis=0)
for j in np.arange(24014):
    selected_net_income_backyard[j] = (
        income_net_of_commuting_costs[cond[j], j])

for item in list_annualized_2d_damages_backyard:
    new_item = item / selected_net_income_backyard
    try:
        outexp.export_map(new_item, grid, geo_grid,
                          path_plots,
                          outexp.retrieve_name(item, -1) + '_shareinc', "",
                          path_tables,
                          ubnd=1)
    except IndexError:
        pass

selected_net_income_informal = np.empty(24014)
cond = np.argmax(initial_state_households[2, :, :], axis=0)
for j in np.arange(24014):
    selected_net_income_informal[j] = (
        income_net_of_commuting_costs[cond[j], j])

for item in list_annualized_2d_damages_informal:
    new_item = item / selected_net_income_informal
    try:
        outexp.export_map(new_item, grid, geo_grid,
                          path_plots,
                          outexp.retrieve_name(item, -1) + '_shareinc', "",
                          path_tables,
                          ubnd=1)
    except IndexError:
        pass


# TODO: Does it really make sense to take share of net income for formal
# where households do not bear structural costs?
# In that case, we should just superimpose fraction of capital destroyed
# over capital map (for formal sector) or exogenous value (for other sectors)


# DYNAMICS
# TODO: plot scenarios (to track evolution of time-moving variables)!
# NB: we can also run the code for each year in simulation arrays
# (no validation needed)

# TODO: Do aggregate damage graphs!

years_simul = np.arange(2011, 2011 + 30)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(years_simul, simulation_utility[:, 0],
        color="maroon", label="Poor")
ax.plot(years_simul, simulation_utility[:, 1],
        color="red", label="Mid-poor")
ax.plot(years_simul, simulation_utility[:, 2],
        color="darkorange", label="Mid-rich")
ax.plot(years_simul, simulation_utility[:, 3],
        color="gold", label="Rich")
ax.set_ylim(0)
ax.yaxis.set_major_formatter(
    mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
plt.tick_params(labelbottom=True)
plt.ylabel("Utility levels", labelpad=15)
plt.savefig(path_plots + 'evol_util_levels.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(years_simul, np.nansum(simulation_households_center, 2)[:, 0],
        color="maroon", label="Poor")
ax.plot(years_simul, np.nansum(simulation_households_center, 2)[:, 1],
        color="red", label="Mid-poor")
ax.plot(years_simul, np.nansum(simulation_households_center, 2)[:, 2],
        color="darkorange", label="Mid-rich")
ax.plot(years_simul, np.nansum(simulation_households_center, 2)[:, 3],
        color="gold", label="Rich")
ax.set_ylim(0)
ax.yaxis.set_major_formatter(
    mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
plt.tick_params(labelbottom=True)
plt.ylabel("Total number of households per income group", labelpad=15)
plt.savefig(path_plots + 'evol_nb_households_incgroup.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(years_simul, np.nansum(simulation_households_housing_type, 2)[:, 0],
        color="gold", label="Formal")
ax.plot(years_simul, np.nansum(simulation_households_housing_type, 2)[:, 1],
        color="darkorange", label="Backyard")
ax.plot(years_simul, np.nansum(simulation_households_housing_type, 2)[:, 2],
        color="red", label="Informal")
ax.plot(years_simul, np.nansum(simulation_households_housing_type, 2)[:, 3],
        color="maroon", label="Subsidized")
ax.set_ylim(0)
ax.yaxis.set_major_formatter(
    mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
plt.tick_params(labelbottom=True)
plt.ylabel("Total number of households per housing type", labelpad=15)
plt.savefig(path_plots + 'evol_nb_households_htype.png')
plt.close()

# TODO: also do 2D variations from 2011 to 2040

# NB: Where do aggregate flood damage estimates come from?

# COMPARISONS
