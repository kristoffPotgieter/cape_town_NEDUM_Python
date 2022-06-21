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
import scipy

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
average_income = np.load(
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
    sim_nb_households_tot, grid, path_plots + 'total_sim',
    "Total number of households (simulation)",
    path_tables, path_data,
    ubnd=5000)
# Note that we lack households in Mitchell's Plain
# TODO: should we correct it?
data_nb_households_tot = np.nansum(housing_types[
    ["informal_grid", "backyard_informal_grid", "formal_grid"]
    ], 1)
total_data = outexp.export_map(
    data_nb_households_tot, grid, path_plots + 'total_data',
    "Total number of households (data)",
    path_tables, path_data,
    ubnd=5000)

#  Per housing type
sim_nb_households_formal = initial_state_households_housing_types[0, :]
formal_sim = outexp.export_map(
    sim_nb_households_formal, grid, path_plots + 'formal_sim',
    "Number of households in formal private (simulation)",
    path_tables, path_data,
    ubnd=1000)
data_nb_households_formal = (housing_types["formal_grid"]
                             - initial_state_households_housing_types[3, :])
formal_data = outexp.export_map(
    data_nb_households_formal, grid, path_plots + 'formal_data',
    "Number of households in formal private (data)",
    path_tables, path_data,
    ubnd=1000)

sim_nb_households_backyard = initial_state_households_housing_types[1, :]
backyard_sim = outexp.export_map(
    sim_nb_households_backyard, grid, path_plots + 'backyard_sim',
    "Number of households in informal backyard (simulation)",
    path_tables, path_data,
    ubnd=1000)
data_nb_households_backyard = housing_types["backyard_informal_grid"]
backyard_data = outexp.export_map(
    data_nb_households_backyard, grid, path_plots + 'backyard_data',
    "Number of households in informal backyard (data)",
    path_tables, path_data,
    ubnd=1000)

sim_nb_households_informal = initial_state_households_housing_types[2, :]
informal_sim = outexp.export_map(
    sim_nb_households_informal, grid, path_plots + 'informal_sim',
    "Number of households in informal settlements (simulation)",
    path_tables, path_data,
    ubnd=3000)
data_nb_households_informal = housing_types["informal_grid"]
informal_data = outexp.export_map(
    data_nb_households_informal, grid, path_plots + 'informal_data',
    "Number of households in informal settlements (data)",
    path_tables, path_data,
    ubnd=3000)

data_nb_households_rdp = initial_state_households_housing_types[3, :]
rdp_sim = outexp.export_map(
    data_nb_households_rdp, grid, path_plots + 'rdp_sim',
    "Number of households in formal subsidized (data)",
    path_tables, path_data,
    ubnd=1800)

#  Per income group
#  NB: validation data is disaggregated from SP, hence the smooth appearance,
#  not necessarily corresponding to reality (we do not plot it)
sim_nb_households_poor = initial_state_household_centers[0, :]
poor_sim = outexp.export_map(
    sim_nb_households_poor, grid, path_plots + 'poor_sim',
    "Number of poor households (simulation)",
    path_tables, path_data,
    ubnd=5000)
sim_nb_households_midpoor = initial_state_household_centers[1, :]
midpoor_sim = outexp.export_map(
    sim_nb_households_midpoor, grid, path_plots + 'midpoor_sim',
    "Number of mid-poor households (simulation)",
    path_tables, path_data,
    ubnd=2000)
sim_nb_households_midrich = initial_state_household_centers[2, :]
midrich_sim = outexp.export_map(
    sim_nb_households_midrich, grid, path_plots + 'midrich_sim',
    "Number of mid-rich households (simulation)",
    path_tables, path_data,
    ubnd=1000)
sim_nb_households_rich = initial_state_household_centers[3, :]
rich_sim = outexp.export_map(
    sim_nb_households_rich, grid, path_plots + 'rich_sim',
    "Number of rich households (simulation)",
    path_tables, path_data,
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
    hsupply_tot, grid, path_plots + 'hsupply_2d_sim',
    "Total housing supply (in m²)",
    path_tables, path_data,
    ubnd=50000)
FAR = np.nansum(housing_supply, 0) / (0.25 * 1000000)
FAR_2d_sim = outexp.export_map(
    FAR, grid, path_plots + 'FAR_2d_sim',
    "Overall floor-area ratio",
    path_tables, path_data,
    ubnd=0.3)

hsupply_formal = housing_supply[0, :]
hsupply_formal_2d_sim = outexp.export_map(
    hsupply_formal, grid, path_plots + 'hsupply_formal_2d_sim',
    "Total housing supply in private formal (in m²)",
    path_tables, path_data,
    ubnd=35000)
FAR_formal = housing_supply[0, :] / (0.25 * 1000000)
FAR_formal_2d_sim = outexp.export_map(
    FAR_formal, grid, path_plots + 'FAR_formal_2d_sim',
    "Floor-area ratio in formal private",
    path_tables, path_data,
    ubnd=0.15)

# Pb of validation in hyper-centre is also reflected in price
sim_HFA_dens_formal = initial_state_housing_supply[0, :] / 1000000
HFA_dens_formal_2d_sim = outexp.export_map(
    sim_HFA_dens_formal, grid, path_plots + 'HFA_dens_formal_2d_sim',
    "Households density in formal private HFA (simulation)",
    path_tables, path_data,
    ubnd=1)
grid_formal_density_HFA[np.isnan(grid_formal_density_HFA)] = 0
data_HFA_dens_formal = grid_formal_density_HFA
HFA_dens_formal_2d_data = outexp.export_map(
    data_HFA_dens_formal, grid, path_plots + 'HFA_dens_formal_2d_data',
    "Households density in formal private HFA (data)",
    path_tables, path_data,
    ubnd=1)

hsupply_backyard = housing_supply[1, :]
hsupply_backyard_2d_sim = outexp.export_map(
    hsupply_backyard, grid, path_plots + 'hsupply_backyard_2d_sim',
    "Total housing supply in informal backyards (in m²)",
    path_tables, path_data,
    ubnd=30000)
FAR_backyard = housing_supply[1, :] / (0.25 * 1000000)
FAR_backyard_2d_sim = outexp.export_map(
    FAR_backyard, grid, path_plots + 'FAR_backyard_2d_sim',
    "Floor-area ratio in informal backyards",
    path_tables, path_data,
    ubnd=0.10)

hsupply_informal = housing_supply[2, :]
hsupply_informal_2d_sim = outexp.export_map(
    hsupply_informal, grid, path_plots + 'hsupply_informal_2d_sim',
    "Total housing supply in informal settlements (in m²)",
    path_tables, path_data,
    ubnd=70000)
FAR_informal = housing_supply[2, :] / (0.25 * 1000000)
FAR_informal_2d_sim = outexp.export_map(
    FAR_informal, grid, path_plots + 'FAR_informal_2d_sim',
    "Floor-area ratio in informal settlements",
    path_tables, path_data,
    ubnd=0.30)

hsupply_rdp = housing_supply[3, :]
hsupply_rdp_2d_sim = outexp.export_map(
    hsupply_rdp, grid, path_plots + 'hsupply_rdp_2d_sim',
    "Total housing supply in formal subsidized (in m²)",
    path_tables, path_data,
    ubnd=25000)
FAR_rdp = housing_supply[3, :] / (0.25 * 1000000)
FAR_rdp_2d_sim = outexp.export_map(
    FAR_rdp, grid, path_plots + 'FAR_rdp_2d_sim',
    "Floor-area ratio in formal subsidized",
    path_tables, path_data,
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
    netincome_poor, grid, path_plots + 'netincome_poor_2d_sim',
    "Estimated income net of commuting costs (poor)",
    path_tables, path_data,
    ubnd=25000, lbnd=-15000, cmap='bwr')
netincome_midpoor = income_net_of_commuting_costs[1, :]
netincome_midpoor_2d_sim = outexp.export_map(
    netincome_midpoor, grid, path_plots + 'netincome_midpoor_2d_sim',
    "Estimated income net of commuting costs (mid-poor)",
    path_tables, path_data,
    ubnd=70000, lbnd=-20000, cmap='bwr')
netincome_midrich = income_net_of_commuting_costs[2, :]
netincome_midrich_2d_sim = outexp.export_map(
    netincome_midrich, grid, path_plots + 'netincome_midrich_2d_sim',
    "Estimated income net of commuting costs (mid-rich)",
    path_tables, path_data,
    ubnd=200000, lbnd=25000)
netincome_rich = income_net_of_commuting_costs[3, :]
netincome_rich_2d_sim = outexp.export_map(
    netincome_rich, grid, path_plots + 'netincome_rich_2d_sim',
    "Estimated income net of commuting costs (rich)",
    path_tables, path_data,
    ubnd=850000, lbnd=250000)

(avg_income_net_of_commuting_1d
 ) = outexp.plot_income_net_of_commuting_costs(
     grid, income_net_of_commuting_costs, path_plots, path_tables)


#  Average income

avgincome_poor = average_income[0, :]
avgincome_poor_2d_sim = outexp.export_map(
    avgincome_poor, grid, path_plots + 'avgincome_poor_2d_sim',
    "Estimated average income (poor)",
    path_tables, path_data,
    ubnd=25000, lbnd=10000)
avgincome_midpoor = average_income[1, :]
avgincome_midpoor_2d_sim = outexp.export_map(
    avgincome_midpoor, grid, path_plots + 'avgincome_midpoor_2d_sim',
    "Estimated average income (mid-poor)",
    path_tables, path_data,
    ubnd=70000, lbnd=25000)
avgincome_midrich = average_income[2, :]
avgincome_midrich_2d_sim = outexp.export_map(
    avgincome_midrich, grid, path_plots + 'avgincome_midrich_2d_sim',
    "Estimated average income (mid-rich)",
    path_tables, path_data,
    ubnd=200000, lbnd=100000)
avgincome_rich = average_income[3, :]
avgincome_rich_2d_sim = outexp.export_map(
    avgincome_rich, grid, path_plots + 'avgincome_rich_2d_sim',
    "Estimated average income (rich)",
    path_tables, path_data,
    ubnd=850000, lbnd=550000)

(avg_income_1d
 ) = outexp.plot_average_income(
     grid, average_income, path_plots, path_tables)

# We also conduct validation with overall average income
# (fit in 1D is OK, cf. calibration)
overall_avg_income = (average_income
                      * initial_state_household_centers
                      / np.nansum(initial_state_household_centers, 0))
overall_avg_income[np.isnan(overall_avg_income)] = 0
overall_avg_income = np.nansum(overall_avg_income, 0)
data_avg_income = data['gridAverageIncome'][0][0].squeeze()
data_avg_income[np.isnan(data_avg_income)] = 0
avgincome_all_2d_sim = outexp.export_map(
    overall_avg_income, grid, path_plots + 'avgincome_all_2d_sim',
    "Estimated average income (all income groups)",
    path_tables, path_data,
    ubnd=850000)
avgincome_all_2d_sim = outexp.export_map(
    overall_avg_income, grid, path_plots + 'avgincome_all_2d_sim',
    "Estimated average income (all income groups)",
    path_tables, path_data,
    ubnd=850000)
     
     
     