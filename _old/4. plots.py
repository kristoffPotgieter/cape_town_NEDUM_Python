# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:40:37 2022.

@author: vincentviguie
"""


# %% Preamble


# IMPORT PACKAGES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os

import inputs.data as inpdt
import inputs.parameters_and_options as inpprm

import equilibrium.functions_dynamic as eqdyn

import outputs.export_outputs as outexp
import outputs.flood_outputs as outfld
import outputs.export_outputs_floods as outval


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/2. Data/'
path_precalc_inp = path_folder + '0. Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_folder + 'data_Cape_Town/Scenarios/'
path_outputs = path_code + '/4. Output/'
path_floods = path_folder + "FATHOM/"


# IMPORT PARAMETERS AND OPTIONS

options = inpprm.import_options()
param = inpprm.import_param(path_precalc_inp, path_outputs)

# TODO: meant to stick with original specification
options["agents_anticipate_floods"] = 1
options["informal_land_constrained"] = 0
options["convert_sal_data"] = 0
options["compute_net_income"] = 0
options["actual_backyards"] = 0
options["unempl_reweight"] = 1
# implicit_empl_rate = 0.74/0.99/0.98/0.99
options["correct_agri_rent"] = 1

options["pluvial"] = 1
options["correct_pluvial"] = 1
options["coastal"] = 1
# This is in line with the DEM used in FATHOM data for fluvial and pluvial
options["dem"] = "MERITDEM"
options["slr"] = 1

name = ('allfloods_precal_modif')
# name = ('floods' + str(options["agents_anticipate_floods"]) + '_'
#         + 'informal' + str(options["informal_land_constrained"]) + '_'
#         + 'actual_backyards1' + '_' + 'pockets1')
plot_repo = name + '/plots/'


# %% Load data


# BASIC GEOGRAPHIC DATA

grid, center = inpdt.import_grid(path_data)
amenities = inpdt.import_amenities(path_precalc_inp)


# HOUSEHOLDS AND INCOME DATA

income_class_by_housing_type = inpdt.import_hypothesis_housing_type()

(mean_income, households_per_income_class, average_income, income_mult,
 income_2011, households_per_income_and_housing
 ) = inpdt.import_income_classes_data(param, path_data)

param["income_year_reference"] = mean_income

#  Import income net of commuting costs, as calibrated in Pfeiffer et al.
#  (see part 3.1 or appendix C3)
income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')
# income_net_of_commuting_costs_29 = np.load(
#     path_precalc_transp + 'GRID_incomeNetofCommuting_29.npy')

# average_income_2011 = np.load(
#     path_precalc_transp + 'average_income_year_0.npy')
# average_income_2040 = np.load(
#     path_precalc_transp + 'average_income_year_28.npy')

# We convert income distribution data (at SP level) to grid dimensions for use
# in income calibration: long to run, uncomment only if needed
# income_distribution_grid = inpdt.convert_income_distribution(
#     income_distribution, grid, path_data, data_sp)
income_distribution_grid = np.load(path_data + "income_distrib_grid.npy")


(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(path_precalc_inp)

#  Import population density per pixel, by housing type
#  Note that there is no RDP, but both formal and informal backyard
housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')
# Replace missing values by zero
housing_types[np.isnan(housing_types)] = 0


# MACRO DATA

(interest_rate, population, housing_type_data, total_RDP
 ) = inpdt.import_macro_data(param, path_scenarios)


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

coeff_land_28 = inpdt.import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal,
    spline_land_RDP, param, 28)

#  We update land use parameters at baseline (relies on data)

housing_limit = inpdt.import_housing_limit(grid, param)

(param, minimum_housing_supply, agricultural_rent
 ) = inpprm.import_construction_parameters(
    param, grid, housing_types_sp, data_sp["dwelling_size"],
    mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land,
    interest_rate, options
    )

# FLOOD DATA
# param = inpdt.infer_WBUS2_depth(housing_types, param, path_floods)
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

pluvial_100yr = pd.read_excel(path_floods + "P_100yr" + ".xlsx")


# LOAD EQUILIBRIUM DATA (from main.py)

initial_state_households = np.load(
    path_outputs + name + '/initial_state_households.npy')
initial_state_households_housing_types = np.load(
    path_outputs + name + '/initial_state_households_housing_types.npy')
initial_state_household_centers = np.load(
    path_outputs + name + '/initial_state_household_centers.npy')
initial_state_housing_supply = np.load(
    path_outputs + name + '/initial_state_housing_supply.npy')
initial_state_rent = np.load(
    path_outputs + name + '/initial_state_rent.npy')
initial_state_dwelling_size = np.load(
    path_outputs + name + '/initial_state_dwelling_size.npy')

# LOAD SIMULATION DATA (from main.py)

# simulation_households_center = np.load(
#     path_outputs + name + '/simulation_households_center.npy')
# simulation_dwelling_size = np.load(
#     path_outputs + name + '/simulation_dwelling_size.npy')
# simulation_rent = np.load(path_outputs + name + '/simulation_rent.npy')
# simulation_households_housing_type = np.load(
#     path_outputs + name + '/simulation_households_housing_type.npy')
# simulation_households = np.load(
#     path_outputs + name + '/simulation_households.npy')
# simulation_utility = np.load(path_outputs + name + '/simulation_utility.npy')

(spline_agricultural_rent, spline_interest_rate,
 spline_population_income_distribution, spline_inflation,
 spline_income_distribution, spline_population,
 spline_income, spline_minimum_housing_supply, spline_fuel
 ) = eqdyn.import_scenarios(income_2011, param, grid, path_scenarios)


construction_coeff = ((spline_income(0) / param["income_year_reference"])
                      ** (-param["coeff_b"])
                      * param["coeff_A"])
construction_coeff_28 = ((spline_income(28) / param["income_year_reference"])
                         ** (-param["coeff_b"])
                         * param["coeff_A"])

inflation = spline_inflation(28) / spline_inflation(0)


# %% Validation: draw maps and figures

# GENERAL VALIDATION

# TODO: discuss need to use reweighting

try:
    os.mkdir(path_outputs + plot_repo)
except OSError as error:
    print(error)

# TODO: How can macro data stick more to simulation than income data on which
# it was optimized? Because of coeff_land? Should we use weights?
# TODO: Why not use SP data for validation?
# TODO: need to reweight?
outexp.export_housing_types(
    initial_state_households_housing_types, initial_state_household_centers,
    housing_type_data, households_per_income_class, 'Simulation', 'Data',
    path_outputs + plot_repo
    )

# Note that we use no reweighting in validation, hence the gap due to formal
# backyards in income data (?)

outexp.export_households(
    initial_state_households, households_per_income_and_housing, 'Simulation',
    'Data', path_outputs + plot_repo)

# TODO: RDP? Formal backyard? Does not change much...
# TODO: Does drop have to do with Mitchell's Plain? Overshoot with
# disamenity parameters?
# TODO: Absurd number far from the center?
outexp.validation_density(
    grid, initial_state_households_housing_types, housing_types,
    path_outputs + plot_repo, coeff_land, land_constraint=0
    )

# TODO: pb seems to come from formal sector essentially
outexp.validation_density_housing_types(
    grid, initial_state_households_housing_types, housing_types, 0,
    path_outputs + plot_repo
    )
outexp.validation_density_housing_types(
    grid, initial_state_households_housing_types, housing_types, 1,
    path_outputs + plot_repo
    )
# TODO: More precisely, pb seems to come from midpoor households at 20km!
# Also midrich at 0 and 20, and rich at 10...
# But then, is validation data trustworthy? Use griddata?
# Also note reweighting pb
outexp.validation_density_income_groups(
    grid, initial_state_household_centers, income_distribution_grid, 0,
    path_outputs + plot_repo
    )
# TODO: switch to SP for more precision?
outexp.validation_density_income_groups(
    grid, initial_state_household_centers, income_distribution_grid, 1,
    path_outputs + plot_repo
    )

outexp.plot_housing_supply(grid, initial_state_housing_supply, coeff_land,
                           0, path_outputs + plot_repo)
outexp.plot_housing_supply(grid, initial_state_housing_supply, coeff_land,
                           1, path_outputs + plot_repo)

# TODO: Also do breakdown for poorest across housing types and backyards
# vs. RDP

# test = (initial_state_households_housing_types[1, :]
#         / initial_state_households_housing_types[3, :])
# test = test[test > 0]

# TODO: does it matter if price distribution is shifted to the right?
outexp.validation_housing_price(
    grid, initial_state_rent, interest_rate, param, center, path_precalc_inp,
    path_outputs + plot_repo
    )

# TODO: how can dwelling size be so big?
outexp.plot_housing_demand(grid, center, initial_state_dwelling_size,
                           path_precalc_inp, path_outputs + plot_repo)

outexp.export_map(initial_state_households_housing_types[0, :], grid,
                  path_outputs + plot_repo + 'formal_sim', 1200)
outexp.export_map(np.nansum(initial_state_households_housing_types,0), grid,
                  path_outputs + plot_repo + 'sim', 4000)

outexp.export_map(
    housing_types.formal_grid - initial_state_households_housing_types[3, :],
    grid, path_outputs + plot_repo + 'formal_data', 1200)
outexp.export_map(
    housing_types.formal_grid + housing_types.informal_grid
    + housing_types.backyard_formal_grid
    + housing_types.backyard_informal_grid,
    grid, path_outputs + plot_repo + 'data', 4000)

#%% More maps for Claus

informal_risks_medium = pd.read_csv(
    path_folder + 'Land occupation/informal_settlements_risk_MEDIUM.csv',
    sep=',')
informal_risks_short = pd.read_csv(
    path_folder + 'Land occupation/informal_settlements_risk_SHORT.csv',
    sep=',')

outexp.export_map(
    informal_risks_medium["area"] / 250000,
    grid, path_outputs + plot_repo + 'informal_risks_medium', 1)
outexp.export_map(
    informal_risks_short["area"] / 250000,
    grid, path_outputs + plot_repo + 'informal_risks_short', 1)

polygon_medium_timing = pd.read_excel(
    path_folder + 'Land occupation/polygon_medium_timing.xlsx',
    header=None)
for item in list(polygon_medium_timing.squeeze()):
    informal_risks_medium.loc[
        informal_risks_medium["grid.data.ID"] == item,
        "area"
        ] = informal_risks_short.loc[
            informal_risks_short["grid.data.ID"] == item,
            "area"
            ]
    informal_risks_short.loc[
        informal_risks_short["grid.data.ID"] == item,
        "area"] = 0

outexp.export_map(
    informal_risks_medium["area"] / 250000,
    grid, path_outputs + plot_repo + 'informal_risks_medium_correct', 1)
outexp.export_map(
    informal_risks_short["area"] / 250000,
    grid, path_outputs + plot_repo + 'informal_risks_short_correct', 1)

grid["dummy"] = 0
for pixel in polygon_medium_timing.iloc[:, 0]:
    grid.dummy[grid.id == pixel] = 1

outexp.export_map(
    grid["dummy"],
    grid, path_outputs + plot_repo + 'polygon_medium_timing', 1)

#%%

# TODO: Is this function still useful?
outexp.plot_diagnosis_map_informl(
    grid, coeff_land, initial_state_households_housing_types,
    path_outputs + plot_repo
    )


# TODO: FLOOD VALIDATION

#  Stats per housing type (flood depth and households in flood-prone areas)
fluvial_floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr',
                  'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
pluvial_floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr', 'P_100yr',
                  'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr']

count_formal = (
    housing_types.formal_grid
    - (number_properties_RDP * total_RDP / sum(number_properties_RDP))
    )
count_formal[count_formal < 0] = 0

stats_per_housing_type_data_fluvial = outfld.compute_stats_per_housing_type(
    fluvial_floods, path_floods, count_formal,
    (number_properties_RDP * total_RDP / sum(number_properties_RDP)),
    housing_types.informal_grid,
    housing_types.backyard_formal_grid + housing_types.backyard_informal_grid,
    options, param, 0.01)

stats_per_housing_type_fluvial = outfld.compute_stats_per_housing_type(
    fluvial_floods, path_floods, initial_state_households_housing_types[0, :],
    initial_state_households_housing_types[3, :],
    initial_state_households_housing_types[2, :],
    initial_state_households_housing_types[1, :],
    options, param, 0.01)

outval.validation_flood
(name, stats_per_housing_type_data_fluvial, stats_per_housing_type_fluvial,
 'Data', 'Simul', 'fluvial')

if options["pluvial"] == 1:
    (stats_per_housing_type_data_pluvial
     ) = outfld.compute_stats_per_housing_type(
         pluvial_floods, path_floods, count_formal,
         (number_properties_RDP * total_RDP / sum(number_properties_RDP)),
         housing_types.informal_grid, housing_types.backyard_formal_grid
         + housing_types.backyard_informal_grid, options, param, 0.01)
    stats_per_housing_type_pluvial = outfld.compute_stats_per_housing_type(
        pluvial_floods, path_floods,
        initial_state_households_housing_types[0, :],
        initial_state_households_housing_types[3, :],
        initial_state_households_housing_types[2, :],
        initial_state_households_housing_types[1, :],
        options, param, 0.01)
    outval.validation_flood(
        name, stats_per_housing_type_data_pluvial,
        stats_per_housing_type_pluvial, 'Data', 'Simul', 'pluvial')

# %% Housing type scenarios

data = pd.DataFrame(
    {'2011': np.nansum(simulation_households_housing_type[0, :, :], 1),
     '2020': np.nansum(simulation_households_housing_type[9, :, :], 1),
     '2030': np.nansum(simulation_households_housing_type[19, :, :], 1),
     '2040': np.nansum(simulation_households_housing_type[28, :, :], 1)},
    index=["Formal private", "Informal in \n backyards",
           "Informal \n settlements", "Formal subsidized"]
    )
data.plot(kind="bar")
plt.tick_params(labelbottom=True)
plt.xticks(rotation='horizontal')
plt.ylabel("Number of households")
plt.ylim(0, 880000)


hh_2011 = simulation_households[0, :, :, :]
hh_2020 = simulation_households[9, :, :, :]
hh_2030 = simulation_households[19, :, :, :]
hh_2040 = simulation_households[29, :, :, :]

np.nansum(hh_2011, 2)
np.nansum(hh_2020, 2)
np.nansum(hh_2030, 2)
np.nansum(hh_2040, 2)


#  Formal
class_income = 3
income_class_2011 = np.argmax(simulation_households[29, :, :, :], 1)
subset = income_class_2011[0, :] == class_income
q = simulation_dwelling_size[29, 0, :][subset]
r = simulation_rent[29, 0, :][subset]
Y = income_net_of_commuting_costs_29[class_income, :][subset]
B = 1
z = (
     (Y - q*r)
     / (1
        + (fraction_capital_destroyed.contents_formal[subset]
           * param["fraction_z_dwellings"]))
     )
U = (z ** (1 - param["beta"])) * (q ** param["beta"]) * amenities[subset] * B

sns.histplot(U)
Umed_formal = np.nanmedian(U)


# Informal

class_income = 0
income_class_2011 = np.argmax(simulation_households[0, :, :, :], 1)
subset = income_class_2011[2, :] == class_income
q = simulation_dwelling_size[0, 2, :][subset]
r = simulation_rent[0, 2, :][subset]
Y = income_net_of_commuting_costs_29[class_income, :][subset]
B = np.load(path_precalc_inp + 'param_pockets.npy')

z = (
     Y - (q*r)
     - (fraction_capital_destroyed.structure_informal_settlements[subset]
        * (param["informal_structure_value"]
           * (spline_inflation(29) / spline_inflation(0))))
     ) / (1 + (fraction_capital_destroyed.contents_informal[subset]
               * param["fraction_z_dwellings"]))
U = ((z ** (1 - param["beta"])) * (q ** param["beta"]) * amenities[subset]
     * B[subset])

sns.histplot(U)
Umed_informal = np.nanmedian(U)


# Backyards

class_income = 1
income_class_2011 = np.argmax(simulation_households[0, :, :, :], 1)
subset = income_class_2011[1, :] == class_income
q = simulation_dwelling_size[0, 1, :][subset]
r = simulation_rent[0, 1, :][subset]
Y = income_net_of_commuting_costs_29[class_income, :][subset]
B = np.load(path_precalc_inp + 'param_backyards.npy')

z = (
     Y - (q*r)
     - (fraction_capital_destroyed.structure_backyards[subset]
        * (param["informal_structure_value"]
           * (spline_inflation(29) / spline_inflation(0))))
     ) / (1 + (fraction_capital_destroyed.contents_backyard[subset]
               * param["fraction_z_dwellings"]))
U = ((z ** (1 - param["beta"])) * (q ** param["beta"]) * amenities[subset]
     * B[subset])

sns.histplot(U)
Umed_backyard = np.nanmedian(U)


# %% Flood prone area scenarios


# FLUVIAL

stats_per_housing_type_2011_fluvial = outfld.compute_stats_per_housing_type(
    fluvial_floods, path_floods, simulation_households_housing_type[0, 0, :],
    simulation_households_housing_type[0, 3, :],
    simulation_households_housing_type[0, 2, :],
    simulation_households_housing_type[0, 1, :],
    options, param, 0.01)
stats_per_housing_type_2040_fluvial = outfld.compute_stats_per_housing_type(
    fluvial_floods, path_floods, simulation_households_housing_type[28, 0, :],
    simulation_households_housing_type[28, 3, :],
    simulation_households_housing_type[28, 2, :],
    simulation_households_housing_type[28, 1, :],
    options, param, 0.01)

label = ["Formal private", "Formal subsidized", "Informal \n settlements",
         "Informal \n in backyards"]

stats_2011_1 = [
    stats_per_housing_type_2011_fluvial[
        'fraction_formal_in_flood_prone_area'][2],
    stats_per_housing_type_2011_fluvial[
        'fraction_subsidized_in_flood_prone_area'][2],
    stats_per_housing_type_2011_fluvial[
        'fraction_informal_in_flood_prone_area'][2],
    stats_per_housing_type_2011_fluvial[
        'fraction_backyard_in_flood_prone_area'][2]
    ]
stats_2011_2 = [
    stats_per_housing_type_2011_fluvial[
        'fraction_formal_in_flood_prone_area'][3],
    stats_per_housing_type_2011_fluvial[
        'fraction_subsidized_in_flood_prone_area'][3],
    stats_per_housing_type_2011_fluvial[
        'fraction_informal_in_flood_prone_area'][3],
    stats_per_housing_type_2011_fluvial[
        'fraction_backyard_in_flood_prone_area'][3]
    ]
stats_2011_3 = [
    stats_per_housing_type_2011_fluvial[
        'fraction_formal_in_flood_prone_area'][5],
    stats_per_housing_type_2011_fluvial[
        'fraction_subsidized_in_flood_prone_area'][5],
    stats_per_housing_type_2011_fluvial[
        'fraction_informal_in_flood_prone_area'][5],
    stats_per_housing_type_2011_fluvial[
        'fraction_backyard_in_flood_prone_area'][5]
    ]
stats_2040_1 = [
    stats_per_housing_type_2040_fluvial[
        'fraction_formal_in_flood_prone_area'][2],
    stats_per_housing_type_2040_fluvial[
        'fraction_subsidized_in_flood_prone_area'][2],
    stats_per_housing_type_2040_fluvial[
        'fraction_informal_in_flood_prone_area'][2],
    stats_per_housing_type_2040_fluvial[
        'fraction_backyard_in_flood_prone_area'][2]
    ]
stats_2040_2 = [
    stats_per_housing_type_2040_fluvial[
        'fraction_formal_in_flood_prone_area'][3],
    stats_per_housing_type_2040_fluvial[
        'fraction_subsidized_in_flood_prone_area'][3],
    stats_per_housing_type_2040_fluvial[
        'fraction_informal_in_flood_prone_area'][3],
    stats_per_housing_type_2040_fluvial[
        'fraction_backyard_in_flood_prone_area'][3]
    ]
stats_2040_3 = [
    stats_per_housing_type_2040_fluvial[
        'fraction_formal_in_flood_prone_area'][5],
    stats_per_housing_type_2040_fluvial[
        'fraction_subsidized_in_flood_prone_area'][5],
    stats_per_housing_type_2040_fluvial[
        'fraction_informal_in_flood_prone_area'][5],
    stats_per_housing_type_2040_fluvial[
        'fraction_backyard_in_flood_prone_area'][5]
    ]

colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
r = np.arange(4)
barWidth = 0.25

plt.figure(figsize=(10, 7))
plt.bar(r, stats_2011_1, color=colors[0], edgecolor='white', width=barWidth,
        label="20 years")
plt.bar(r, np.array(stats_2011_2) - np.array(stats_2011_1),
        bottom=np.array(stats_2011_1), color=colors[1], edgecolor='white',
        width=barWidth, label='50 years')
plt.bar(r, np.array(stats_2011_3) - (np.array(stats_2011_2)),
        bottom=(np.array(stats_2011_2)), color=colors[2], edgecolor='white',
        width=barWidth, label='100 years')
plt.bar(r + 0.25, np.array(stats_2040_1), color=colors[0], edgecolor='white',
        width=barWidth)
plt.bar(r + 0.25, np.array(stats_2040_2) - np.array(stats_2040_1),
        bottom=np.array(stats_2040_1), color=colors[1], edgecolor='white',
        width=barWidth)
plt.bar(r + 0.25, np.array(stats_2040_3) - np.array(stats_2040_2),
        bottom=np.array(stats_2040_2), color=colors[2], edgecolor='white',
        width=barWidth)

plt.legend(loc='upper right')
plt.xticks(r, label)
plt.ylim(0, 75000)
plt.text(
    r[0] - 0.1,
    stats_per_housing_type_2011_fluvial[
        'fraction_formal_in_flood_prone_area'][5]
    + 0.005,
    "2011")
plt.text(
    r[1] - 0.1,
    stats_per_housing_type_2011_fluvial[
        'fraction_subsidized_in_flood_prone_area'][5]
    + 0.005,
    "2011")
plt.text(
    r[2] - 0.1,
    stats_per_housing_type_2011_fluvial[
        'fraction_informal_in_flood_prone_area'][5]
    + 0.005,
    "2011")
plt.text(
    r[3] - 0.1,
    stats_per_housing_type_2011_fluvial[
        'fraction_backyard_in_flood_prone_area'][5]
    + 0.005,
    "2011")
plt.text(
    r[0] + 0.15,
    stats_per_housing_type_2040_fluvial[
        'fraction_formal_in_flood_prone_area'][5]
    + 0.005,
    '2040')
plt.text(
    r[1] + 0.15,
    stats_per_housing_type_2040_fluvial[
        'fraction_subsidized_in_flood_prone_area'][5]
    + 0.005,
    '2040')
plt.text(
    r[2] + 0.15,
    stats_per_housing_type_2040_fluvial[
        'fraction_informal_in_flood_prone_area'][5]
    + 0.005,
    '2040')
plt.text(
    r[3] + 0.15,
    stats_per_housing_type_2040_fluvial[
        'fraction_backyard_in_flood_prone_area'][5]
    + 0.005,
    '2040')
plt.tick_params(labelbottom=True)
plt.ylabel("Dwellings in flood-prone areas")
plt.show()

# v2 HARRIS

(stats_per_housing_type_2011_fluvial["tot"]
 ) = (
      stats_per_housing_type_2011_fluvial.fraction_formal_in_flood_prone_area
      + np.array(stats_per_housing_type_2011_fluvial[
          'fraction_backyard_in_flood_prone_area'])
      + stats_per_housing_type_2011_fluvial[
          'fraction_subsidized_in_flood_prone_area']
      + stats_per_housing_type_2011_fluvial[
          'fraction_informal_in_flood_prone_area']
      )

(stats_per_housing_type_2040_fluvial["tot"]
 ) = (
      stats_per_housing_type_2040_fluvial.fraction_formal_in_flood_prone_area
      + np.array(stats_per_housing_type_2040_fluvial[
          'fraction_backyard_in_flood_prone_area'])
      + stats_per_housing_type_2040_fluvial[
          'fraction_subsidized_in_flood_prone_area']
      + stats_per_housing_type_2040_fluvial[
          'fraction_informal_in_flood_prone_area']
      )

plt.figure(figsize=(10, 7))
barWidth = 0.25

(vec_2011_formal
 ) = (stats_per_housing_type_2011_fluvial.fraction_formal_in_flood_prone_area
      / stats_per_housing_type_2011_fluvial["tot"])
(vec_2011_formal
 ) = [np.nansum(simulation_households_housing_type[0, :, :], 1)[0]
      / sum(np.nansum(simulation_households_housing_type[0, :, :], 1)),
      vec_2011_formal[2], vec_2011_formal[3], vec_2011_formal[5]]
(vec_2011_subsidized
 ) = (stats_per_housing_type_2011_fluvial[
     'fraction_subsidized_in_flood_prone_area']
      / stats_per_housing_type_2011_fluvial["tot"])
(vec_2011_subsidized
 ) = [np.nansum(simulation_households_housing_type[0, :, :], 1)[3]
      / sum(np.nansum(simulation_households_housing_type[0, :, :], 1)),
      vec_2011_subsidized[2], vec_2011_subsidized[3], vec_2011_subsidized[5]]
(vec_2011_informal
 ) = (stats_per_housing_type_2011_fluvial.fraction_informal_in_flood_prone_area
      / stats_per_housing_type_2011_fluvial["tot"])
(vec_2011_informal
 ) = [np.nansum(simulation_households_housing_type[0, :, :], 1)[2]
      / sum(np.nansum(simulation_households_housing_type[0, :, :], 1)),
      vec_2011_informal[2], vec_2011_informal[3], vec_2011_informal[5]]
(vec_2011_backyard
 ) = (stats_per_housing_type_2011_fluvial.fraction_backyard_in_flood_prone_area
      / stats_per_housing_type_2011_fluvial["tot"])
(vec_2011_backyard
 ) = [np.nansum(simulation_households_housing_type[0, :, :], 1)[1]
      / sum(np.nansum(simulation_households_housing_type[0, :, :], 1)),
      vec_2011_backyard[2], vec_2011_backyard[3], vec_2011_backyard[5]]
(vec_2040_formal
 ) = (stats_per_housing_type_2040_fluvial.fraction_formal_in_flood_prone_area
      / stats_per_housing_type_2040_fluvial["tot"])
(vec_2040_formal
 ) = [np.nansum(simulation_households_housing_type[28, :, :], 1)[0]
      / sum(np.nansum(simulation_households_housing_type[28, :, :], 1)),
      vec_2040_formal[2], vec_2040_formal[3], vec_2040_formal[5]]
(vec_2040_subsidized
 ) = (stats_per_housing_type_2040_fluvial[
     'fraction_subsidized_in_flood_prone_area']
      / stats_per_housing_type_2040_fluvial["tot"])
(vec_2040_subsidized
 ) = [np.nansum(simulation_households_housing_type[28, :, :], 1)[3]
      / sum(np.nansum(simulation_households_housing_type[28, :, :], 1)),
      vec_2040_subsidized[2], vec_2040_subsidized[3], vec_2040_subsidized[5]]
(vec_2040_informal
 ) = (stats_per_housing_type_2040_fluvial.fraction_informal_in_flood_prone_area
      / stats_per_housing_type_2040_fluvial["tot"])
(vec_2040_informal
 ) = [np.nansum(simulation_households_housing_type[28, :, :], 1)[2]
      / sum(np.nansum(simulation_households_housing_type[28, :, :], 1)),
      vec_2040_informal[2], vec_2040_informal[3], vec_2040_informal[5]]
(vec_2040_backyard
 ) = (stats_per_housing_type_2040_fluvial.fraction_backyard_in_flood_prone_area
      / stats_per_housing_type_2040_fluvial["tot"])
(vec_2040_backyard
 ) = [np.nansum(simulation_households_housing_type[28, :, :], 1)[1]
      / sum(np.nansum(simulation_households_housing_type[28, :, :], 1)),
      vec_2040_backyard[2], vec_2040_backyard[3], vec_2040_backyard[5]]

plt.ylim(0, 1.3)
plt.ylabel("Fraction of dwellings of each housing type")
label = ["Over the city", "In 20-year \n return period \n flood zones",
         "In 50-year \n return period \n flood zones",
         "In 100-year \n return period \n flood zones"]

plt.bar(np.arange(4), vec_2011_formal, color=colors[0], edgecolor='white',
        width=barWidth, label="Formal private")
plt.bar(np.arange(4), vec_2011_subsidized, bottom=vec_2011_formal,
        color=colors[1], edgecolor='white', width=barWidth,
        label="Formal subsidized")
plt.bar(np.arange(4), vec_2011_informal,
        bottom=np.array(vec_2011_formal) + np.array(vec_2011_subsidized),
        color=colors[2], edgecolor='white', width=barWidth,
        label="Informal settlements")
plt.bar(np.arange(4), vec_2011_backyard,
        bottom=np.array(vec_2011_formal) + np.array(vec_2011_subsidized)
        + np.array(vec_2011_informal), color=colors[3], edgecolor='white',
        width=barWidth, label="Informal in backyards")
plt.bar(np.arange(4) + 0.25, vec_2040_formal, color=colors[0],
        edgecolor='white', width=barWidth)
plt.bar(np.arange(4) + 0.25, vec_2040_subsidized, bottom=vec_2040_formal,
        color=colors[1], edgecolor='white', width=barWidth)
plt.bar(np.arange(4) + 0.25, vec_2040_informal,
        bottom=np.array(vec_2040_formal) + np.array(vec_2040_subsidized),
        color=colors[2], edgecolor='white', width=barWidth)
plt.bar(np.arange(4) + 0.25, vec_2040_backyard,
        bottom=np.array(vec_2040_formal) + np.array(vec_2040_subsidized)
        + np.array(vec_2040_informal), color=colors[3], edgecolor='white',
        width=barWidth)

plt.legend(loc='upper left')
plt.xticks(np.arange(4), label)
plt.text(r[0] - 0.1, 1.005, "2011")
plt.text(r[1] - 0.1, 1.005, "2011")
plt.text(r[2] - 0.1, 1.005, "2011")
plt.text(r[3] - 0.1, 1.005, "2011")
plt.text(r[0] + 0.15, 1.005, '2040')
plt.text(r[1] + 0.15, 1.005, '2040')
plt.text(r[2] + 0.15, 1.005, '2040')
plt.text(r[3] + 0.15, 1.005, '2040')


# PLUVIAL
stats_per_housing_type_2011_pluvial = outfld.compute_stats_per_housing_type(
    pluvial_floods, path_floods, simulation_households_housing_type[0, 0, :],
    simulation_households_housing_type[0, 3, :],
    simulation_households_housing_type[0, 2, :],
    simulation_households_housing_type[0, 1, :],
    options, param, 0.01)
stats_per_housing_type_2040_pluvial = outfld.compute_stats_per_housing_type(
    pluvial_floods, path_floods, simulation_households_housing_type[28, 0, :],
    simulation_households_housing_type[28, 3, :],
    simulation_households_housing_type[28, 2, :],
    simulation_households_housing_type[28, 1, :],
    options, param, 0.01)

label = ["Formal private", "Formal subsidized", "Informal \n settlements",
         "Informal \n in backyards"]
stats_2011_1 = [
    stats_per_housing_type_2011_pluvial.fraction_formal_in_flood_prone_area[2],
    stats_per_housing_type_2011_pluvial[
        'fraction_subsidized_in_flood_prone_area'][2],
    stats_per_housing_type_2011_pluvial[
        'fraction_informal_in_flood_prone_area'][2],
    stats_per_housing_type_2011_pluvial[
        'fraction_backyard_in_flood_prone_area'][2]
    ]
stats_2011_2 = [
    stats_per_housing_type_2011_pluvial.fraction_formal_in_flood_prone_area[3],
    stats_per_housing_type_2011_pluvial[
        'fraction_subsidized_in_flood_prone_area'][3],
    stats_per_housing_type_2011_pluvial[
        'fraction_informal_in_flood_prone_area'][3],
    stats_per_housing_type_2011_pluvial[
        'fraction_backyard_in_flood_prone_area'][3]
    ]
stats_2011_3 = [
    stats_per_housing_type_2011_pluvial.fraction_formal_in_flood_prone_area[5],
    stats_per_housing_type_2011_pluvial[
        'fraction_subsidized_in_flood_prone_area'][5],
    stats_per_housing_type_2011_pluvial[
        'fraction_informal_in_flood_prone_area'][5],
    stats_per_housing_type_2011_pluvial[
        'fraction_backyard_in_flood_prone_area'][5]
    ]
stats_2040_1 = [
    stats_per_housing_type_2040_pluvial.fraction_formal_in_flood_prone_area[2],
    stats_per_housing_type_2040_pluvial[
        'fraction_subsidized_in_flood_prone_area'][2],
    stats_per_housing_type_2040_pluvial[
        'fraction_informal_in_flood_prone_area'][2],
    stats_per_housing_type_2040_pluvial[
        'fraction_backyard_in_flood_prone_area'][2]
    ]
stats_2040_2 = [
    stats_per_housing_type_2040_pluvial.fraction_formal_in_flood_prone_area[3],
    stats_per_housing_type_2040_pluvial[
        'fraction_subsidized_in_flood_prone_area'][3],
    stats_per_housing_type_2040_pluvial[
        'fraction_informal_in_flood_prone_area'][3],
    stats_per_housing_type_2040_pluvial[
        'fraction_backyard_in_flood_prone_area'][3]
    ]
stats_2040_3 = [
    stats_per_housing_type_2040_pluvial.fraction_formal_in_flood_prone_area[5],
    stats_per_housing_type_2040_pluvial[
        'fraction_subsidized_in_flood_prone_area'][5],
    stats_per_housing_type_2040_pluvial[
        'fraction_informal_in_flood_prone_area'][5],
    stats_per_housing_type_2040_pluvial[
        'fraction_backyard_in_flood_prone_area'][5]
    ]

colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
r = np.arange(4)
barWidth = 0.25

plt.figure(figsize=(10, 7))
plt.bar(r, stats_2011_1, color=colors[0], edgecolor='white', width=barWidth,
        label="20 years")
plt.bar(r, np.array(stats_2011_2) - np.array(stats_2011_1),
        bottom=np.array(stats_2011_1), color=colors[1], edgecolor='white',
        width=barWidth, label='50 years')
plt.bar(r, np.array(stats_2011_3) - (np.array(stats_2011_2)),
        bottom=(np.array(stats_2011_2)), color=colors[2], edgecolor='white',
        width=barWidth, label='100 years')
plt.bar(r + 0.25, np.array(stats_2040_1), color=colors[0], edgecolor='white',
        width=barWidth)
plt.bar(r + 0.25, np.array(stats_2040_2) - np.array(stats_2040_1),
        bottom=np.array(stats_2040_1), color=colors[1], edgecolor='white',
        width=barWidth)
plt.bar(r + 0.25, np.array(stats_2040_3) - np.array(stats_2040_2),
        bottom=np.array(stats_2040_2), color=colors[2], edgecolor='white',
        width=barWidth)

plt.legend(loc='upper right')
plt.xticks(r, label)
plt.ylim(0, 290000)
plt.text(
    r[0] - 0.1,
    stats_per_housing_type_2011_pluvial.fraction_formal_in_flood_prone_area[5]
    + 0.005,
    "2011")
plt.text(
    r[1] - 0.1,
    stats_per_housing_type_2011_pluvial[
        'fraction_subsidized_in_flood_prone_area'][5]
    + 0.005,
    "2011")
plt.text(
    r[2] - 0.1,
    stats_per_housing_type_2011_pluvial[
        'fraction_informal_in_flood_prone_area'][5]
    + 0.005,
    "2011")
plt.text(
    r[3] - 0.1,
    stats_per_housing_type_2011_pluvial[
        'fraction_backyard_in_flood_prone_area'][5]
    + 0.005,
    "2011")
plt.text(
    r[0] + 0.15,
    stats_per_housing_type_2040_pluvial.fraction_formal_in_flood_prone_area[5]
    + 0.005,
    '2040')
plt.text(
    r[1] + 0.15,
    stats_per_housing_type_2040_pluvial[
        'fraction_subsidized_in_flood_prone_area'][5]
    + 0.005,
    '2040')
plt.text(
    r[2] + 0.15,
    stats_per_housing_type_2040_pluvial[
        'fraction_informal_in_flood_prone_area'][5]
    + 0.005,
    '2040')
plt.text(
    r[3] + 0.15,
    stats_per_housing_type_2040_pluvial[
        'fraction_backyard_in_flood_prone_area'][5]
    + 0.005,
    '2040')
plt.tick_params(labelbottom=True)
plt.ylabel("Dwellings in flood-prone areas")
plt.show()


# v2 HARRIS

(stats_per_housing_type_2011_pluvial["tot"]
 ) = (stats_per_housing_type_2011_pluvial.fraction_formal_in_flood_prone_area
      + np.array(stats_per_housing_type_2011_pluvial[
          'fraction_backyard_in_flood_prone_area'])
      + stats_per_housing_type_2011_pluvial[
          'fraction_subsidized_in_flood_prone_area']
      + stats_per_housing_type_2011_pluvial[
          'fraction_informal_in_flood_prone_area'])

(stats_per_housing_type_2040_pluvial["tot"]
 ) = (stats_per_housing_type_2040_pluvial.fraction_formal_in_flood_prone_area
      + np.array(stats_per_housing_type_2040_pluvial[
          'fraction_backyard_in_flood_prone_area'])
      + stats_per_housing_type_2040_pluvial[
          'fraction_subsidized_in_flood_prone_area']
      + stats_per_housing_type_2040_pluvial[
          'fraction_informal_in_flood_prone_area'])

plt.figure(figsize=(10, 7))
barWidth = 0.25

(vec_2011_formal
 ) = (stats_per_housing_type_2011_pluvial.fraction_formal_in_flood_prone_area
      / stats_per_housing_type_2011_pluvial["tot"])
(vec_2011_formal
 ) = [np.nansum(simulation_households_housing_type[0, :, :], 1)[0]
      / sum(np.nansum(simulation_households_housing_type[0, :, :], 1)),
      vec_2011_formal[2], vec_2011_formal[3], vec_2011_formal[5]]
(vec_2011_subsidized
 ) = (stats_per_housing_type_2011_pluvial[
     'fraction_subsidized_in_flood_prone_area']
      / stats_per_housing_type_2011_pluvial["tot"])
(vec_2011_subsidized
 ) = [np.nansum(simulation_households_housing_type[0, :, :], 1)[3]
      / sum(np.nansum(simulation_households_housing_type[0, :, :], 1)),
      vec_2011_subsidized[2], vec_2011_subsidized[3], vec_2011_subsidized[5]]
(vec_2011_informal
 ) = (stats_per_housing_type_2011_pluvial.fraction_informal_in_flood_prone_area
      / stats_per_housing_type_2011_pluvial["tot"])
(vec_2011_informal
 ) = [np.nansum(simulation_households_housing_type[0, :, :], 1)[2]
      / sum(np.nansum(simulation_households_housing_type[0, :, :], 1)),
      vec_2011_informal[2], vec_2011_informal[3], vec_2011_informal[5]]
(vec_2011_backyard
 ) = (stats_per_housing_type_2011_pluvial.fraction_backyard_in_flood_prone_area
      / stats_per_housing_type_2011_pluvial["tot"])
(vec_2011_backyard
 ) = [np.nansum(simulation_households_housing_type[0, :, :], 1)[1]
      / sum(np.nansum(simulation_households_housing_type[0, :, :], 1)),
      vec_2011_backyard[2], vec_2011_backyard[3], vec_2011_backyard[5]]
(vec_2040_formal
 ) = (stats_per_housing_type_2040_pluvial.fraction_formal_in_flood_prone_area
      / stats_per_housing_type_2040_pluvial["tot"])
(vec_2040_formal
 ) = [np.nansum(simulation_households_housing_type[28, :, :], 1)[0]
      / sum(np.nansum(simulation_households_housing_type[28, :, :], 1)),
      vec_2040_formal[2], vec_2040_formal[3], vec_2040_formal[5]]
(vec_2040_subsidized
 ) = (stats_per_housing_type_2040_pluvial[
     'fraction_subsidized_in_flood_prone_area']
      / stats_per_housing_type_2040_pluvial["tot"])
(vec_2040_subsidized
 ) = [np.nansum(simulation_households_housing_type[28, :, :], 1)[3]
      / sum(np.nansum(simulation_households_housing_type[28, :, :], 1)),
      vec_2040_subsidized[2], vec_2040_subsidized[3], vec_2040_subsidized[5]]
(vec_2040_informal
 ) = (stats_per_housing_type_2040_pluvial.fraction_informal_in_flood_prone_area
      / stats_per_housing_type_2040_pluvial["tot"])
(vec_2040_informal
 ) = [np.nansum(simulation_households_housing_type[28, :, :], 1)[2]
      / sum(np.nansum(simulation_households_housing_type[28, :, :], 1)),
      vec_2040_informal[2], vec_2040_informal[3], vec_2040_informal[5]]
(vec_2040_backyard
 ) = (stats_per_housing_type_2040_pluvial.fraction_backyard_in_flood_prone_area
      / stats_per_housing_type_2040_pluvial["tot"])
(vec_2040_backyard
 ) = [np.nansum(simulation_households_housing_type[28, :, :], 1)[1]
      / sum(np.nansum(simulation_households_housing_type[28, :, :], 1)),
      vec_2040_backyard[2], vec_2040_backyard[3], vec_2040_backyard[5]]

plt.ylim(0, 1.3)
label = ["Over the city", "In 20-year \n return period \n flood zones",
         "In 50-year \n return period \n flood zones",
         "In 100-year \n return period \n flood zones"]

plt.bar(np.arange(4), vec_2011_formal, color=colors[0], edgecolor='white',
        width=barWidth, label="Formal private")
plt.bar(np.arange(4), vec_2011_subsidized, bottom=vec_2011_formal,
        color=colors[1], edgecolor='white', width=barWidth,
        label="Formal subsidized")
plt.bar(np.arange(4), vec_2011_informal,
        bottom=np.array(vec_2011_formal) + np.array(vec_2011_subsidized),
        color=colors[2], edgecolor='white', width=barWidth,
        label="Informal settlements")
plt.bar(np.arange(4), vec_2011_backyard,
        bottom=np.array(vec_2011_formal) + np.array(vec_2011_subsidized)
        + np.array(vec_2011_informal), color=colors[3], edgecolor='white',
        width=barWidth, label="Informal in backyards")
plt.bar(np.arange(4) + 0.25, vec_2040_formal, color=colors[0],
        edgecolor='white', width=barWidth)
plt.bar(np.arange(4) + 0.25, vec_2040_subsidized, bottom=vec_2040_formal,
        color=colors[1], edgecolor='white', width=barWidth)
plt.bar(np.arange(4) + 0.25, vec_2040_informal,
        bottom=np.array(vec_2040_formal) + np.array(vec_2040_subsidized),
        color=colors[2], edgecolor='white', width=barWidth)
plt.bar(np.arange(4) + 0.25, vec_2040_backyard,
        bottom=np.array(vec_2040_formal) + np.array(vec_2040_subsidized)
        + np.array(vec_2040_informal), color=colors[3], edgecolor='white',
        width=barWidth)

plt.legend(loc='upper left')
plt.ylabel("Fraction of dwellings of each housing type")
plt.xticks(np.arange(4), label)
plt.text(r[0] - 0.1, 1.005, "2011")
plt.text(r[1] - 0.1, 1.005, "2011")
plt.text(r[2] - 0.1, 1.005, "2011")
plt.text(r[3] - 0.1, 1.005, "2011")
plt.text(r[0] + 0.15, 1.005, '2040')
plt.text(r[1] + 0.15, 1.005, '2040')
plt.text(r[2] + 0.15, 1.005, '2040')
plt.text(r[3] + 0.15, 1.005, '2040')


# %% Damages by income group scenarios

# RUN ONLY IF FLOODS IN THE MODEL!

formal_structure_cost_2011 = outfld.compute_formal_structure_cost_method2(
    simulation_rent[0, :, :],
    param,
    eqdyn.interpolate_interest_rate(spline_interest_rate, 0),
    coeff_land,
    simulation_households_housing_type[0, :, :],
    construction_coeff
    )

content_cost_2011 = outfld.compute_content_cost(
    simulation_households_center[0, :, :],
    income_net_of_commuting_costs,
    param,
    fraction_capital_destroyed,
    simulation_rent[0, :, :],
    simulation_dwelling_size[0, :, :],
    eqdyn.interpolate_interest_rate(spline_interest_rate, 0)
    )

formal_structure_cost_2040 = outfld.compute_formal_structure_cost_method2(
    simulation_rent[28, :, :],
    param,
    eqdyn.interpolate_interest_rate(spline_interest_rate, 28),
    coeff_land_28,
    simulation_households_housing_type[28, :, :],
    construction_coeff_28
    )

content_cost_2040 = outfld.compute_content_cost(
    simulation_households_center[28, :, :],
    income_net_of_commuting_costs_29,
    param,
    fraction_capital_destroyed,
    simulation_rent[28, :, :],
    simulation_dwelling_size[28, :, :],
    eqdyn.interpolate_interest_rate(spline_interest_rate, 28)
    )

# To be set dynamically?
item = 'FD_100yr'
option = "percent"

df2011 = pd.DataFrame()
df2040 = pd.DataFrame()
# type_flood = copy.deepcopy(item)
data_flood = np.squeeze(pd.read_excel(path_floods + item + ".xlsx"))

formal_damages = structural_damages_type4a(data_flood['flood_depth'])
(formal_damages[simulation_dwelling_size[0, 0, :] > param["threshold"]]
 ) = structural_damages_type4b(data_flood.flood_depth[
     simulation_dwelling_size[0, 0, :] > param["threshold"]])

subsidized_damages = structural_damages_type4a(data_flood['flood_depth'])
(subsidized_damages[simulation_dwelling_size[0, 3, :] > param["threshold"]]
 ) = structural_damages_type4b(data_flood.flood_depth[
     simulation_dwelling_size[0, 3, :] > param["threshold"]])

df2011['formal_structure_damages'] = (
    formal_structure_cost_2011 * formal_damages)
df2011['subsidized_structure_damages'] = (
    param["subsidized_structure_value_ref"]
    * subsidized_damages)
df2011['informal_structure_damages'] = (
    param["informal_structure_value_ref"]
    * structural_damages_type2(data_flood['flood_depth'])
    )
df2011['backyard_structure_damages'] = (
    ((16216
      * (param["informal_structure_value_ref"]
         * structural_damages_type2(data_flood['flood_depth'])))
     + (74916
        * (param["informal_structure_value_ref"]
           * structural_damages_type3a(data_flood['flood_depth']))))
    / (74916 + 16216)
    )

df2011['formal_content_damages'] = (
    content_cost_2011.formal * content_damages(data_flood['flood_depth']))
df2011['subsidized_content_damages'] = (
    content_cost_2011.subsidized * content_damages(data_flood['flood_depth']))
df2011['informal_content_damages'] = (
    content_cost_2011.informal * content_damages(data_flood['flood_depth']))
df2011['backyard_content_damages'] = (
    content_cost_2011.backyard * content_damages(data_flood['flood_depth']))

formal_damages = structural_damages_type4a(data_flood['flood_depth'])
(formal_damages[simulation_dwelling_size[28, 0, :] > param["threshold"]]
 ) = structural_damages_type4b(
     data_flood.flood_depth[simulation_dwelling_size[28, 0, :]
                            > param["threshold"]])
subsidized_damages = structural_damages_type4a(data_flood['flood_depth'])
(subsidized_damages[simulation_dwelling_size[28, 3, :] > param["threshold"]]
 ) = structural_damages_type4b(
     data_flood.flood_depth[simulation_dwelling_size[28, 3, :]
                            > param["threshold"]])

df2040['formal_structure_damages'] = (
    formal_structure_cost_2040 * formal_damages)
df2040['subsidized_structure_damages'] = (
    param["subsidized_structure_value_ref"]
    * (spline_inflation(28) / spline_inflation(0)) * subsidized_damages)
df2040['informal_structure_damages'] = (
    param["informal_structure_value_ref"]
    * (spline_inflation(28) / spline_inflation(0))
    * structural_damages_type2(data_flood['flood_depth'])
    )
df2040['backyard_structure_damages'] = (
    ((16216
      * (param["informal_structure_value_ref"]
         * (spline_inflation(28) / spline_inflation(0))
         * structural_damages_type2(data_flood['flood_depth'])))
     + (74916
        * (param["informal_structure_value_ref"]
           * (spline_inflation(28) / spline_inflation(0))
           * structural_damages_type3a(data_flood['flood_depth']))))
    / (74916 + 16216)
    )

df2040['formal_content_damages'] = (
    content_cost_2040.formal * content_damages(data_flood['flood_depth']))
df2040['subsidized_content_damages'] = (
    content_cost_2040.subsidized * content_damages(data_flood['flood_depth']))
df2040['informal_content_damages'] = (
    content_cost_2040.informal * content_damages(data_flood['flood_depth']))
df2040['backyard_content_damages'] = (
    content_cost_2040.backyard * content_damages(data_flood['flood_depth']))

df2011["formal_pop_flood_prone"] = (
    simulation_households_housing_type[0, 0, :]
    * data_flood["prop_flood_prone"]
    )
df2011["backyard_pop_flood_prone"] = (
    simulation_households_housing_type[0, 1, :]
    * data_flood["prop_flood_prone"]
    )
df2011["informal_pop_flood_prone"] = (
    simulation_households_housing_type[0, 2, :]
    * data_flood["prop_flood_prone"]
    )
df2011["subsidized_pop_flood_prone"] = (
    simulation_households_housing_type[0, 3, :]
    * data_flood["prop_flood_prone"]
    )

df2040["formal_pop_flood_prone"] = (
    simulation_households_housing_type[28, 0, :]
    * data_flood["prop_flood_prone"]
    )
df2040["backyard_pop_flood_prone"] = (
    simulation_households_housing_type[28, 1, :]
    * data_flood["prop_flood_prone"]
    )
df2040["informal_pop_flood_prone"] = (
    simulation_households_housing_type[28, 2, :]
    * data_flood["prop_flood_prone"]
    )
df2040["subsidized_pop_flood_prone"] = (
    simulation_households_housing_type[28, 3, :]
    * data_flood["prop_flood_prone"]
    )

df2011["formal_damages"] = (
    df2011['formal_structure_damages'] + df2011['formal_content_damages']
    )
df2011["informal_damages"] = (
    df2011['informal_structure_damages'] + df2011['informal_content_damages']
    )
df2011["subsidized_damages"] = (
    df2011['subsidized_structure_damages']
    + df2011['subsidized_content_damages']
    )
df2011["backyard_damages"] = (
    df2011['backyard_structure_damages'] + df2011['backyard_content_damages']
    )

df2040["formal_damages"] = (
    df2040['formal_structure_damages'] + df2040['formal_content_damages']
    )
df2040["informal_damages"] = (
    df2040['informal_structure_damages'] + df2040['informal_content_damages']
    )
df2040["subsidized_damages"] = (
    df2040['subsidized_structure_damages']
    + df2040['subsidized_content_damages']
    )
df2040["backyard_damages"] = (
    df2040['backyard_structure_damages'] + df2040['backyard_content_damages']
    )

subset = df2011[
    (~np.isnan(df2011.formal_damages)) & (df2011.formal_pop_flood_prone > 0)
    ]

sns.displot(subset.formal_damages, hist=True, kde=False,
             hist_kws={'weights': subset.formal_pop_flood_prone})


# %% EAD scenarios


formal_structure_cost_2011 = outfld.compute_formal_structure_cost_method2(
    simulation_rent[0, :, :],
    param,
    eqdyn.interpolate_interest_rate(spline_interest_rate, 0),
    coeff_land,
    simulation_households_housing_type[0, :, :],
    construction_coeff
    )

content_cost_2011 = outfld.compute_content_cost(
    simulation_households_center[0, :, :],
    income_net_of_commuting_costs,
    param,
    fraction_capital_destroyed,
    simulation_rent[0, :, :],
    simulation_dwelling_size[0, :, :],
    eqdyn.interpolate_interest_rate(spline_interest_rate, 0))

damages_fluvial_2011 = outfld.compute_damages(
    fluvial_floods,
    path_floods,
    param,
    content_cost_2011,
    simulation_households_housing_type[0, 0, :],
    simulation_households_housing_type[0, 3, :],
    simulation_households_housing_type[0, 2, :],
    simulation_households_housing_type[0, 1, :],
    simulation_dwelling_size[0, :, :],
    formal_structure_cost_2011,
    content_damages,
    structural_damages_type4b,
    structural_damages_type4a,
    structural_damages_type2,
    structural_damages_type3a,
    options, spline_inflation, 0)

formal_structure_cost_2040 = outfld.compute_formal_structure_cost_method2(
    simulation_rent[28, :, :],
    param,
    eqdyn.interpolate_interest_rate(spline_interest_rate, 28),
    coeff_land_28,
    simulation_households_housing_type[28, :, :],
    construction_coeff_28
    )

content_cost_2040 = outfld.compute_content_cost(
    simulation_households_center[28, :, :],
    income_net_of_commuting_costs_29,
    param,
    fraction_capital_destroyed,
    simulation_rent[28, :, :],
    simulation_dwelling_size[28, :, :],
    eqdyn.interpolate_interest_rate(spline_interest_rate, 28)
    )

damages_fluvial_2040 = outfld.compute_damages(
    fluvial_floods, path_floods, param, content_cost_2040,
    simulation_households_housing_type[28, 0, :],
    simulation_households_housing_type[28, 3, :],
    simulation_households_housing_type[28, 2, :],
    simulation_households_housing_type[28, 1, :],
    simulation_dwelling_size[28, :, :],
    formal_structure_cost_2040, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a,
    options, spline_inflation, 28)

damages_fluvial_2011.backyard_damages = (
    damages_fluvial_2011.backyard_content_damages
    + damages_fluvial_2011.backyard_structure_damages
    )
damages_fluvial_2011.informal_damages = (
    damages_fluvial_2011.informal_content_damages
    + damages_fluvial_2011.informal_structure_damages
    )
damages_fluvial_2011.subsidized_damages = (
    damages_fluvial_2011.subsidized_content_damages
    + damages_fluvial_2011.subsidized_structure_damages
    )
damages_fluvial_2011.formal_damages = (
    damages_fluvial_2011.formal_content_damages
    + damages_fluvial_2011.formal_structure_damages
    )

damages_fluvial_2040.backyard_damages = (
    damages_fluvial_2040.backyard_content_damages
    + damages_fluvial_2040.backyard_structure_damages
    )
damages_fluvial_2040.informal_damages = (
    damages_fluvial_2040.informal_content_damages
    + damages_fluvial_2040.informal_structure_damages
    )
damages_fluvial_2040.subsidized_damages = (
    damages_fluvial_2040.subsidized_content_damages
    + damages_fluvial_2040.subsidized_structure_damages
    )
damages_fluvial_2040.formal_damages = (
    damages_fluvial_2040.formal_content_damages
    + damages_fluvial_2040.formal_structure_damages
    )

formal_structure_cost_2011 = outfld.compute_formal_structure_cost_method2(
    simulation_rent[0, :, :],
    param,
    eqdyn.interpolate_interest_rate(spline_interest_rate, 0),
    coeff_land,
    simulation_households_housing_type[0, :, :],
    construction_coeff
    )

content_cost_2011 = outfld.compute_content_cost(
    simulation_households_center[0, :, :],
    income_net_of_commuting_costs,
    param,
    fraction_capital_destroyed,
    simulation_rent[0, :, :],
    simulation_dwelling_size[0, :, :],
    eqdyn.interpolate_interest_rate(spline_interest_rate, 0)
    )

damages_pluvial_2011 = outfld.compute_damages(
    pluvial_floods, path_floods, param, content_cost_2011,
    simulation_households_housing_type[0, 0, :],
    simulation_households_housing_type[0, 3, :],
    simulation_households_housing_type[0, 2, :],
    simulation_households_housing_type[0, 1, :],
    simulation_dwelling_size[0, :, :],
    formal_structure_cost_2011, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a,
    options, spline_inflation, 0)

formal_structure_cost_2040 = outfld.compute_formal_structure_cost_method2(
    simulation_rent[28, :, :],
    param,
    eqdyn.interpolate_interest_rate(spline_interest_rate, 28),
    coeff_land_28,
    simulation_households_housing_type[28, :, :],
    construction_coeff_28
    )

content_cost_2040 = outfld.compute_content_cost(
    simulation_households_center[28, :, :],
    income_net_of_commuting_costs_29,
    param,
    fraction_capital_destroyed,
    simulation_rent[28, :, :],
    simulation_dwelling_size[28, :, :],
    eqdyn.interpolate_interest_rate(spline_interest_rate, 28)
    )

damages_pluvial_2040 = outfld.compute_damages(
    pluvial_floods, path_floods, param, content_cost_2040,
    simulation_households_housing_type[28, 0, :],
    simulation_households_housing_type[28, 3, :],
    simulation_households_housing_type[28, 2, :],
    simulation_households_housing_type[28, 1, :],
    simulation_dwelling_size[28, :, :],
    formal_structure_cost_2040, content_damages,
    structural_damages_type4b, structural_damages_type4a,
    structural_damages_type2, structural_damages_type3a,
    options, spline_inflation, 28
    )

damages_pluvial_2011.backyard_damages = (
    damages_pluvial_2011.backyard_content_damages
    + damages_pluvial_2011.backyard_structure_damages
    )
damages_pluvial_2011.informal_damages = (
    damages_pluvial_2011.informal_content_damages
    + damages_pluvial_2011.informal_structure_damages
    )
damages_pluvial_2011.subsidized_damages = (
    damages_pluvial_2011.subsidized_content_damages
    + damages_pluvial_2011.subsidized_structure_damages
    )
damages_pluvial_2011.formal_damages = (
    damages_pluvial_2011.formal_content_damages
    + damages_pluvial_2011.formal_structure_damages
    )

damages_pluvial_2040.backyard_damages = (
    damages_pluvial_2040.backyard_content_damages
    + damages_pluvial_2040.backyard_structure_damages
    )
damages_pluvial_2040.informal_damages = (
    damages_pluvial_2040.informal_content_damages
    + damages_pluvial_2040.informal_structure_damages
    )
damages_pluvial_2040.subsidized_damages = (
    damages_pluvial_2040.subsidized_content_damages
    + damages_pluvial_2040.subsidized_structure_damages
    )
damages_pluvial_2040.formal_damages = (
    damages_pluvial_2040.formal_content_damages
    + damages_pluvial_2040.formal_structure_damages
    )

damages_pluvial_2011.formal_damages[0:3] = 0
damages_pluvial_2040.formal_damages[0:3] = 0
damages_pluvial_2011.backyard_damages[0:2] = 0
damages_pluvial_2040.backyard_damages[0:2] = 0
damages_pluvial_2011.subsidized_damages[0:2] = 0
damages_pluvial_2040.subsidized_damages[0:2] = 0

damages_2011 = pd.DataFrame()
damages_2011["backyard_damages"] = (
    damages_pluvial_2011.backyard_damages
    + damages_fluvial_2011.backyard_damages
    )
damages_2011["informal_damages"] = (
    damages_pluvial_2011.informal_damages
    + damages_fluvial_2011.informal_damages
    )
damages_2011["subsidized_damages"] = (
    damages_pluvial_2011.subsidized_damages
    + damages_fluvial_2011.subsidized_damages
    )
damages_2011["formal_damages"] = (
    damages_pluvial_2011.formal_damages +
    damages_fluvial_2011.formal_damages
    )

damages_2040 = pd.DataFrame()
damages_2040["backyard_damages"] = (
    damages_pluvial_2040.backyard_damages
    + damages_fluvial_2040.backyard_damages
    )
damages_2040["informal_damages"] = (
    damages_pluvial_2040.informal_damages
    + damages_fluvial_2040.informal_damages
    )
damages_2040["subsidized_damages"] = (
    damages_pluvial_2040.subsidized_damages
    + damages_fluvial_2040.subsidized_damages
    )
damages_2040["formal_damages"] = (
    damages_pluvial_2040.formal_damages + damages_fluvial_2040.formal_damages
    )

label = ["2011", "2040", "2040 (deflated)"]
stats_2011_formal = [
    outfld.annualize_damages(damages_2011.formal_damages),
    outfld.annualize_damages(damages_2040.formal_damages),
    outfld.annualize_damages(damages_2040.formal_damages) / inflation]
stats_2011_subsidized = [
    outfld.annualize_damages(damages_2011.subsidized_damages),
    outfld.annualize_damages(damages_2040.subsidized_damages),
    outfld.annualize_damages(damages_2040.subsidized_damages) / inflation]
stats_2011_informal = [
    outfld.annualize_damages(damages_2011.informal_damages),
    outfld.annualize_damages(damages_2040.informal_damages),
    outfld.annualize_damages(damages_2040.informal_damages) / inflation]
stats_2011_backyard = [
    outfld.annualize_damages(damages_2011.backyard_damages),
    outfld.annualize_damages(damages_2040.backyard_damages),
    outfld.annualize_damages(damages_2040.backyard_damages) / inflation]

colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
r = np.arange(3)
barWidth = 0.5
plt.figure(figsize=(10, 7))
plt.bar(r, stats_2011_formal, color=colors[0], edgecolor='white',
        width=barWidth, label="formal")
plt.bar(r, np.array(stats_2011_subsidized), bottom=np.array(stats_2011_formal),
        color=colors[1], edgecolor='white', width=barWidth, label='subsidized')
plt.bar(r, np.array(stats_2011_informal),
        bottom=(np.array(stats_2011_subsidized) + np.array(stats_2011_formal)),
        color=colors[2], edgecolor='white', width=barWidth, label='informal')
plt.bar(r, np.array(stats_2011_backyard),
        bottom=(np.array(stats_2011_informal) + np.array(stats_2011_subsidized)
                + np.array(stats_2011_formal)), color=colors[3],
        edgecolor='white', width=barWidth, label='backyard')
plt.legend()
plt.xticks(r, label)
plt.ylim(0, 600000000)
plt.tick_params(labelbottom=True)
plt.ylabel("Estimated annual damages (R)")
plt.show()


# %% Data for maps

pluvial_100yr["flood_prone"] = (
    (pluvial_100yr.flood_depth > 0.05) & (pluvial_100yr.prop_flood_prone > 0.8)
    )

# 30% of the grid cell is prone to floods of at least 1 cm every 100 yrs

pop_2011 = np.nansum(simulation_households_housing_type[0, :, :], 0)
pop_2040 = np.nansum(simulation_households_housing_type[28, :, :], 0)
red_2011 = ((pop_2011 > 10) & (pluvial_100yr["flood_prone"] == 1))
grey_2011 = ((pop_2011 > 10) & (pluvial_100yr["flood_prone"] == 0))
blue_2011 = ((pop_2011 < 10) & (pluvial_100yr["flood_prone"] == 1))
red_2040 = ((pop_2040 > 10) & (pluvial_100yr["flood_prone"] == 1))
grey_2040 = ((pop_2040 > 10) & (pluvial_100yr["flood_prone"] == 0))
blue_2040 = ((pop_2040 < 10) & (pluvial_100yr["flood_prone"] == 1))

df = pd.DataFrame()
df["pop_2011"] = pop_2011
df["pop_2040"] = pop_2040
(df["fraction_capital_destroyed"]
 ) = fraction_capital_destroyed["structure_formal_2"]
df.to_excel(path_outputs + name + "/map_data.xlsx")


# %% Plot damages per household

# 0. Flood damages per household

floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr',
          'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
# floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr', 'P_100yr',
#           'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr']
option = "percent"
# option = "absolu"

formal_structure_cost_2011 = outfld.compute_formal_structure_cost_method2(
    simulation_rent[0, :, :],
    param,
    eqdyn.interpolate_interest_rate(spline_interest_rate, 0),
    coeff_land,
    simulation_households_housing_type[0, :, :],
    construction_coeff)
content_cost_2011 = outfld.compute_content_cost(
    simulation_households_center[0, :, :],
    income_net_of_commuting_costs,
    param,
    fraction_capital_destroyed,
    simulation_rent[0, :, :],
    simulation_dwelling_size[0, :, :],
    eqdyn.interpolate_interest_rate(spline_interest_rate, 0))
formal_structure_cost_2040 = outfld.compute_formal_structure_cost_method2(
    simulation_rent[28, :, :],
    param,
    eqdyn.interpolate_interest_rate(spline_interest_rate, 28),
    coeff_land_28,
    simulation_households_housing_type[28, :, :],
    construction_coeff_28)
content_cost_2040 = outfld.compute_content_cost(
    simulation_households_center[28, :, :],
    income_net_of_commuting_costs_29,
    param,
    fraction_capital_destroyed,
    simulation_rent[28, :, :],
    simulation_dwelling_size[28, :, :],
    eqdyn.interpolate_interest_rate(spline_interest_rate, 28))


for item in floods:

    param["subsidized_structure_value_ref"] = 150000
    param["informal_structure_value_ref"] = 4000
    df2011 = pd.DataFrame()
    df2040 = pd.DataFrame()
    type_flood = copy.deepcopy(item)
    data_flood = np.squeeze(pd.read_excel(path_floods + item + ".xlsx"))

    formal_damages = structural_damages_type4a(data_flood['flood_depth'])
    (formal_damages[simulation_dwelling_size[0, 0, :] > param["threshold"]]
     ) = structural_damages_type4b(
         data_flood.flood_depth[
             simulation_dwelling_size[0, 0, :] > param["threshold"]]
         )
    subsidized_damages = structural_damages_type4a(data_flood['flood_depth'])
    (subsidized_damages[simulation_dwelling_size[0, 3, :] > param["threshold"]]
     ) = structural_damages_type4b(
         data_flood.flood_depth[
             simulation_dwelling_size[0, 3, :] > param["threshold"]]
         )

    df2011['formal_structure_damages'] = (
        formal_structure_cost_2011 * formal_damages)
    df2011['subsidized_structure_damages'] = (
        param["subsidized_structure_value_ref"] * subsidized_damages)
    df2011['informal_structure_damages'] = param["informal_structure_value_ref"] * \
        structural_damages_type2(data_flood['flood_depth'])
    df2011['backyard_structure_damages'] = (
        16216 * param["informal_structure_value_ref"]
        * structural_damages_type2(data_flood['flood_depth'])
        + 74916 * param["informal_structure_value_ref"]
        * structural_damages_type3a(data_flood['flood_depth'])
        ) / (74916 + 16216)

    df2011['formal_content_damages'] = content_cost_2011.formal * \
        content_damages(data_flood['flood_depth'])
    df2011['subsidized_content_damages'] = content_cost_2011.subsidized * \
        content_damages(data_flood['flood_depth'])
    df2011['informal_content_damages'] = content_cost_2011.informal * \
        content_damages(data_flood['flood_depth'])
    df2011['backyard_content_damages'] = content_cost_2011.backyard * \
        content_damages(data_flood['flood_depth'])

    formal_damages = structural_damages_type4a(data_flood['flood_depth'])
    (formal_damages[simulation_dwelling_size[28, 0, :] > param["threshold"]]
     ) = structural_damages_type4b(
         data_flood.flood_depth[
             simulation_dwelling_size[28, 0, :] > param["threshold"]]
         )
    subsidized_damages = structural_damages_type4a(data_flood['flood_depth'])
    subsidized_damages[
        simulation_dwelling_size[28, 3, :] > param["threshold"]
        ] = structural_damages_type4b(
            data_flood.flood_depth[
                simulation_dwelling_size[28, 3, :] > param["threshold"]]
            )

    df2040['formal_structure_damages'] = (
        formal_structure_cost_2040 * formal_damages)
    df2040['subsidized_structure_damages'] = (
        param["subsidized_structure_value_ref"]
        * (spline_inflation(28) / spline_inflation(0)) * subsidized_damages
        )
    df2040['informal_structure_damages'] = (
        param["informal_structure_value_ref"]
        * (spline_inflation(28) / spline_inflation(0))
        * structural_damages_type2(data_flood['flood_depth'])
        )
    df2040['backyard_structure_damages'] = (
        16216 * param["informal_structure_value_ref"]
        * (spline_inflation(28) / spline_inflation(0))
        * structural_damages_type2(data_flood['flood_depth'])
        + 74916 * param["informal_structure_value_ref"]
        * (spline_inflation(28) / spline_inflation(0))
        * structural_damages_type3a(data_flood['flood_depth'])
        ) / (74916 + 16216)

    df2040['formal_content_damages'] = (
        content_cost_2040.formal * content_damages(data_flood['flood_depth']))
    df2040['subsidized_content_damages'] = (
        content_cost_2040.subsidized
        * content_damages(data_flood['flood_depth']))
    df2040['informal_content_damages'] = (
        content_cost_2040.informal
        * content_damages(data_flood['flood_depth']))
    df2040['backyard_content_damages'] = (
        content_cost_2040.backyard * content_damages(data_flood['flood_depth'])
        )

    df2011["formal_pop_flood_prone"] = (
        simulation_households_housing_type[0, 0, :]
        * data_flood["prop_flood_prone"])
    df2011["backyard_pop_flood_prone"] = (
        simulation_households_housing_type[0, 1, :]
        * data_flood["prop_flood_prone"])
    df2011["informal_pop_flood_prone"] = (
        simulation_households_housing_type[0, 2, :]
        * data_flood["prop_flood_prone"])
    df2011["subsidized_pop_flood_prone"] = (
        simulation_households_housing_type[0, 3, :]
        * data_flood["prop_flood_prone"])

    df2040["formal_pop_flood_prone"] = (
        simulation_households_housing_type[28, 0, :]
        * data_flood["prop_flood_prone"])
    df2040["backyard_pop_flood_prone"] = (
        simulation_households_housing_type[28, 1, :]
        * data_flood["prop_flood_prone"])
    df2040["informal_pop_flood_prone"] = (
        simulation_households_housing_type[28, 2, :]
        * data_flood["prop_flood_prone"])
    df2040["subsidized_pop_flood_prone"] = (
        simulation_households_housing_type[28, 3, :]
        * data_flood["prop_flood_prone"])

    df2011["formal_damages"] = (
        df2011['formal_structure_damages'] + df2011['formal_content_damages'])
    df2011["informal_damages"] = (
        df2011['informal_structure_damages']
        + df2011['informal_content_damages'])
    df2011["subsidized_damages"] = (
        df2011['subsidized_structure_damages']
        + df2011['subsidized_content_damages'])
    df2011["backyard_damages"] = (
        df2011['backyard_structure_damages']
        + df2011['backyard_content_damages'])

    df2040["formal_damages"] = (
        df2040['formal_structure_damages'] + df2040['formal_content_damages'])
    df2040["informal_damages"] = (
        df2040['informal_structure_damages']
        + df2040['informal_content_damages'])
    df2040["subsidized_damages"] = (
        df2040['subsidized_structure_damages']
        + df2040['subsidized_content_damages'])
    df2040["backyard_damages"] = (
        df2040['backyard_structure_damages']
        + df2040['backyard_content_damages'])

    if item == "P_20yr":
        df2011["formal_damages"] = 0
        df2040["formal_damages"] = 0
        df2011["formal_pop_flood_prone"] = 0
        df2040["formal_pop_flood_prone"] = 0
    elif ((item == "P_5yr") | (item == "P_10yr")):
        df2011["formal_damages"] = 0
        df2040["formal_damages"] = 0
        df2011["subsidized_damages"] = 0
        df2040["subsidized_damages"] = 0
        df2011["backyard_damages"] = 0
        df2040["backyard_damages"] = 0
        df2011["formal_pop_flood_prone"] = 0
        df2040["formal_pop_flood_prone"] = 0
        df2011["backyard_pop_flood_prone"] = 0
        df2040["backyard_pop_flood_prone"] = 0
        df2011["subsidized_pop_flood_prone"] = 0
        df2040["subsidized_pop_flood_prone"] = 0
    writer = pd.ExcelWriter(
        path_outputs + name + '/damages_' + str(item) + '_2011.xlsx')
    df2011.to_excel(excel_writer=writer)
    writer.save()
    writer = pd.ExcelWriter(
        path_outputs + name + '/damages_' + str(item) + '_2040.xlsx')

    df2040.to_excel(excel_writer=writer)
    writer.save()

damages_5yr_2011 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_5yr' + '_2011.xlsx')
damages_10yr_2011 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_10yr' + '_2011.xlsx')
damages_20yr_2011 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_20yr' + '_2011.xlsx')
damages_50yr_2011 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_50yr' + '_2011.xlsx')
damages_75yr_2011 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_75yr' + '_2011.xlsx')
damages_100yr_2011 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_100yr' + '_2011.xlsx')
damages_200yr_2011 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_200yr' + '_2011.xlsx')
damages_250yr_2011 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_250yr' + '_2011.xlsx')
damages_500yr_2011 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_500yr' + '_2011.xlsx')
damages_1000yr_2011 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_1000yr' + '_2011.xlsx')

damages_5yr_2040 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_5yr' + '_2040.xlsx')
damages_10yr_2040 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_10yr' + '_2040.xlsx')
damages_20yr_2040 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_20yr' + '_2040.xlsx')
damages_50yr_2040 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_50yr' + '_2040.xlsx')
damages_75yr_2040 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_75yr' + '_2040.xlsx')
damages_100yr_2040 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_100yr' + '_2040.xlsx')
damages_200yr_2040 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_200yr' + '_2040.xlsx')
damages_250yr_2040 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_250yr' + '_2040.xlsx')
damages_500yr_2040 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_500yr' + '_2040.xlsx')
damages_1000yr_2040 = pd.read_excel(
    path_outputs + name + '/damages_' + 'FD_1000yr' + '_2040.xlsx')

damages_10yr_2011.iloc[:, 9:13] = (
    damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13])
damages_20yr_2011.iloc[:, 9:13] = (
    damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13]
    - damages_5yr_2011.iloc[:, 9:13])
damages_50yr_2011.iloc[:, 9:13] = (
    damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13]
    - damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13])
damages_75yr_2011.iloc[:, 9:13] = (
    damages_75yr_2011.iloc[:, 9:13] - damages_50yr_2011.iloc[:, 9:13]
    - damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13]
    - damages_5yr_2011.iloc[:, 9:13])
damages_100yr_2011.iloc[:, 9:13] = (
    damages_100yr_2011.iloc[:, 9:13] - damages_75yr_2011.iloc[:, 9:13]
    - damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13]
    - damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13])
damages_200yr_2011.iloc[:, 9:13] = (
    damages_200yr_2011.iloc[:, 9:13] - damages_100yr_2011.iloc[:, 9:13]
    - damages_75yr_2011.iloc[:, 9:13] - damages_50yr_2011.iloc[:, 9:13]
    - damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13]
    - damages_5yr_2011.iloc[:, 9:13])
damages_250yr_2011.iloc[:, 9:13] = (
    damages_250yr_2011.iloc[:, 9:13] - damages_200yr_2011.iloc[:, 9:13]
    - damages_100yr_2011.iloc[:, 9:13] - damages_75yr_2011.iloc[:, 9:13]
    - damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13]
    - damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13])
damages_500yr_2011.iloc[:, 9:13] = (
    damages_500yr_2011.iloc[:, 9:13] - damages_250yr_2011.iloc[:, 9:13]
    - damages_200yr_2011.iloc[:, 9:13] - damages_100yr_2011.iloc[:, 9:13]
    - damages_75yr_2011.iloc[:, 9:13] - damages_50yr_2011.iloc[:, 9:13]
    - damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13]
    - damages_5yr_2011.iloc[:, 9:13])
damages_1000yr_2011.iloc[:, 9:13] = (
    damages_1000yr_2011.iloc[:, 9:13] - damages_500yr_2011.iloc[:, 9:13]
    - damages_250yr_2011.iloc[:, 9:13] - damages_200yr_2011.iloc[:, 9:13]
    - damages_100yr_2011.iloc[:, 9:13] - damages_75yr_2011.iloc[:, 9:13]
    - damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13]
    - damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13])

damages_10yr_2040.iloc[:, 9:13] = (
    damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13])
damages_20yr_2040.iloc[:, 9:13] = (
    damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13]
    - damages_5yr_2040.iloc[:, 9:13])
damages_50yr_2040.iloc[:, 9:13] = (
    damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13]
    - damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13])
damages_75yr_2040.iloc[:, 9:13] = (
    damages_75yr_2040.iloc[:, 9:13] - damages_50yr_2040.iloc[:, 9:13]
    - damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13]
    - damages_5yr_2040.iloc[:, 9:13])
damages_100yr_2040.iloc[:, 9:13] = (
    damages_100yr_2040.iloc[:, 9:13] - damages_75yr_2040.iloc[:, 9:13]
    - damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13]
    - damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13])
damages_200yr_2040.iloc[:, 9:13] = (
    damages_200yr_2040.iloc[:, 9:13] - damages_100yr_2040.iloc[:, 9:13]
    - damages_75yr_2040.iloc[:, 9:13] - damages_50yr_2040.iloc[:, 9:13]
    - damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13]
    - damages_5yr_2040.iloc[:, 9:13])
damages_250yr_2040.iloc[:, 9:13] = (
    damages_250yr_2040.iloc[:, 9:13] - damages_200yr_2040.iloc[:, 9:13]
    - damages_100yr_2040.iloc[:, 9:13] - damages_75yr_2040.iloc[:, 9:13]
    - damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13]
    - damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13])
damages_500yr_2040.iloc[:, 9:13] = (
    damages_500yr_2040.iloc[:, 9:13] - damages_250yr_2040.iloc[:, 9:13]
    - damages_200yr_2040.iloc[:, 9:13] - damages_100yr_2040.iloc[:, 9:13]
    - damages_75yr_2040.iloc[:, 9:13] - damages_50yr_2040.iloc[:, 9:13]
    - damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13]
    - damages_5yr_2040.iloc[:, 9:13])
damages_1000yr_2040.iloc[:, 9:13] = (
    damages_1000yr_2040.iloc[:, 9:13] - damages_500yr_2040.iloc[:, 9:13]
    - damages_250yr_2040.iloc[:, 9:13] - damages_200yr_2040.iloc[:, 9:13]
    - damages_100yr_2040.iloc[:, 9:13] - damages_75yr_2040.iloc[:, 9:13]
    - damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13]
    - damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13]
    )

damages_5yr_2011.iloc[:, 13:17] = outfld.annualize_damages(
    [damages_5yr_2011.iloc[:, 13:17], damages_10yr_2011.iloc[:, 13:17],
     damages_20yr_2011.iloc[:, 13:17], damages_50yr_2011.iloc[:, 13:17],
     damages_75yr_2011.iloc[:, 13:17], damages_100yr_2011.iloc[:, 13:17],
     damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17],
     damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_10yr_2011.iloc[:, 13:17] = outfld.annualize_damages(
    [0, damages_10yr_2011.iloc[:, 13:17], damages_20yr_2011.iloc[:, 13:17],
     damages_50yr_2011.iloc[:, 13:17], damages_75yr_2011.iloc[:, 13:17],
     damages_100yr_2011.iloc[:, 13:17], damages_200yr_2011.iloc[:, 13:17],
     damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17],
     damages_1000yr_2011.iloc[:, 13:17]])
damages_20yr_2011.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, damages_20yr_2011.iloc[:, 13:17], damages_50yr_2011.iloc[:, 13:17],
     damages_75yr_2011.iloc[:, 13:17], damages_100yr_2011.iloc[:, 13:17],
     damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17],
     damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_50yr_2011.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, damages_50yr_2011.iloc[:, 13:17],
     damages_75yr_2011.iloc[:, 13:17], damages_100yr_2011.iloc[:, 13:17],
     damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17],
     damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_75yr_2011.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, damages_75yr_2011.iloc[:, 13:17],
     damages_100yr_2011.iloc[:, 13:17], damages_200yr_2011.iloc[:, 13:17],
     damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17],
     damages_1000yr_2011.iloc[:, 13:17]])
damages_100yr_2011.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, 0, damages_100yr_2011.iloc[:, 13:17],
     damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17],
     damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_200yr_2011.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, 0, 0, damages_200yr_2011.iloc[:, 13:17],
     damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17],
     damages_1000yr_2011.iloc[:, 13:17]])
damages_250yr_2011.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, 0, 0, 0, damages_250yr_2011.iloc[:, 13:17],
     damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_500yr_2011.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, 0, 0, 0, 0, damages_500yr_2011.iloc[:, 13:17],
     damages_1000yr_2011.iloc[:, 13:17]])
damages_1000yr_2011.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, damages_1000yr_2011.iloc[:, 13:17]])

damages_5yr_2040.iloc[:, 13:17] = outfld.annualize_damages(
    [damages_5yr_2040.iloc[:, 13:17], damages_10yr_2040.iloc[:, 13:17],
     damages_20yr_2040.iloc[:, 13:17], damages_50yr_2040.iloc[:, 13:17],
     damages_75yr_2040.iloc[:, 13:17], damages_100yr_2040.iloc[:, 13:17],
     damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17],
     damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_10yr_2040.iloc[:, 13:17] = outfld.annualize_damages(
    [0, damages_10yr_2040.iloc[:, 13:17], damages_20yr_2040.iloc[:, 13:17],
     damages_50yr_2040.iloc[:, 13:17], damages_75yr_2040.iloc[:, 13:17],
     damages_100yr_2040.iloc[:, 13:17], damages_200yr_2040.iloc[:, 13:17],
     damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17],
     damages_1000yr_2040.iloc[:, 13:17]])
damages_20yr_2040.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, damages_20yr_2040.iloc[:, 13:17], damages_50yr_2040.iloc[:, 13:17],
     damages_75yr_2040.iloc[:, 13:17], damages_100yr_2040.iloc[:, 13:17],
     damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17],
     damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_50yr_2040.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, damages_50yr_2040.iloc[:, 13:17],
     damages_75yr_2040.iloc[:, 13:17], damages_100yr_2040.iloc[:, 13:17],
     damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17],
     damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_75yr_2040.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, damages_75yr_2040.iloc[:, 13:17],
     damages_100yr_2040.iloc[:, 13:17], damages_200yr_2040.iloc[:, 13:17],
     damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17],
     damages_1000yr_2040.iloc[:, 13:17]])
damages_100yr_2040.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, 0, damages_100yr_2040.iloc[:, 13:17],
     damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17],
     damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_200yr_2040.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, 0, 0, damages_200yr_2040.iloc[:, 13:17],
     damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17],
     damages_1000yr_2040.iloc[:, 13:17]])
damages_250yr_2040.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, 0, 0, 0, damages_250yr_2040.iloc[:, 13:17],
     damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_500yr_2040.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, 0, 0, 0, 0, damages_500yr_2040.iloc[:, 13:17],
     damages_1000yr_2040.iloc[:, 13:17]])
damages_1000yr_2040.iloc[:, 13:17] = outfld.annualize_damages(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, damages_1000yr_2040.iloc[:, 13:17]])

damages_5yr_2011 = damages_5yr_2011.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_10yr_2011 = damages_10yr_2011.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_20yr_2011 = damages_20yr_2011.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_50yr_2011 = damages_50yr_2011.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_75yr_2011 = damages_75yr_2011.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_100yr_2011 = damages_100yr_2011.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_200yr_2011 = damages_200yr_2011.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_250yr_2011 = damages_250yr_2011.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_500yr_2011 = damages_500yr_2011.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_1000yr_2011 = damages_1000yr_2011.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']

damages_5yr_2040 = damages_5yr_2040.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_10yr_2040 = damages_10yr_2040.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_20yr_2040 = damages_20yr_2040.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_50yr_2040 = damages_50yr_2040.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_75yr_2040 = damages_75yr_2040.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_100yr_2040 = damages_100yr_2040.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_200yr_2040 = damages_200yr_2040.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_250yr_2040 = damages_250yr_2040.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_500yr_2040 = damages_500yr_2040.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']
damages_1000yr_2040 = damages_1000yr_2040.loc[
    :, 'formal_pop_flood_prone':'backyard_damages']


income_class_2011 = np.argmax(simulation_households[0, :, :, :], 1)
income_class_2040 = np.argmax(simulation_households[28, :, :, :], 1)

real_income_2011 = np.empty((24014, 4))
for i in range(0, 24014):
    for j in range(0, 4):
        print(i)
        real_income_2011[i, j] = (
            average_income_2011[np.array(income_class_2011)[j, i], i]
            )

real_income_2040 = np.empty((24014, 4))
for i in range(0, 24014):
    for j in range(0, 4):
        print(i)
        real_income_2040[i, j] = (
            average_income_2040[np.array(income_class_2040)[j, i], i]
            )

real_income_2011 = np.matlib.repmat(real_income_2011, 10, 1).squeeze()
real_income_2040 = np.matlib.repmat(real_income_2040, 10, 1).squeeze()

total_2011 = np.vstack(
    [damages_5yr_2011, damages_10yr_2011, damages_20yr_2011, damages_50yr_2011,
     damages_75yr_2011, damages_100yr_2011, damages_200yr_2011,
     damages_250yr_2011, damages_500yr_2011, damages_1000yr_2011])
total_2040 = np.vstack(
    [damages_5yr_2040, damages_10yr_2040, damages_20yr_2040, damages_50yr_2040,
     damages_75yr_2040, damages_100yr_2040, damages_200yr_2040,
     damages_250yr_2040, damages_500yr_2040, damages_1000yr_2040])

total_2011[:, 4] = (total_2011[:, 4] / real_income_2011[:, 0]) * 100
total_2011[:, 5] = (total_2011[:, 5] / real_income_2011[:, 2]) * 100
total_2011[:, 6] = (total_2011[:, 6] / real_income_2011[:, 3]) * 100
total_2011[:, 7] = (total_2011[:, 7] / real_income_2011[:, 1]) * 100

total_2040[:, 4] = (total_2040[:, 4] / real_income_2040[:, 0]) * 100
total_2040[:, 5] = (total_2040[:, 5] / real_income_2040[:, 2]) * 100
total_2040[:, 6] = (total_2040[:, 6] / real_income_2040[:, 3]) * 100
total_2040[:, 7] = (total_2040[:, 7] / real_income_2040[:, 1]) * 100

# Reshape
formal_2011 = total_2011[:, [0, 4]]
backyard_2011 = total_2011[:, [1, 7]]
informal_2011 = total_2011[:, [2, 5]]
subsidized_2011 = total_2011[:, [3, 6]]

formal_2040 = total_2040[:, [0, 4]]
backyard_2040 = total_2040[:, [1, 7]]
informal_2040 = total_2040[:, [2, 5]]
subsidized_2040 = total_2040[:, [3, 6]]


# Now we subset by income class

income_class_2011_reshape = np.matlib.repmat(
    income_class_2011, 1, 10).squeeze()
income_class_2040_reshape = np.matlib.repmat(
    income_class_2040, 1, 10).squeeze()

formal_2011_class1 = formal_2011[income_class_2011_reshape[0, :] == 0]
formal_2011_class2 = formal_2011[income_class_2011_reshape[0, :] == 1]
formal_2011_class3 = formal_2011[income_class_2011_reshape[0, :] == 2]
formal_2011_class4 = formal_2011[income_class_2011_reshape[0, :] == 3]

formal_2040_class1 = formal_2040[income_class_2040_reshape[0, :] == 0]
formal_2040_class2 = formal_2040[income_class_2040_reshape[0, :] == 1]
formal_2040_class3 = formal_2040[income_class_2040_reshape[0, :] == 2]
formal_2040_class4 = formal_2040[income_class_2040_reshape[0, :] == 3]

subsidized_2011_class1 = subsidized_2011[income_class_2011_reshape[3, :] == 0]
subsidized_2011_class2 = subsidized_2011[income_class_2011_reshape[3, :] == 1]
subsidized_2011_class3 = subsidized_2011[income_class_2011_reshape[3, :] == 2]
subsidized_2011_class4 = subsidized_2011[income_class_2011_reshape[3, :] == 3]

subsidized_2040_class1 = subsidized_2040[income_class_2040_reshape[3, :] == 0]
subsidized_2040_class2 = subsidized_2040[income_class_2040_reshape[3, :] == 1]
subsidized_2040_class3 = subsidized_2040[income_class_2040_reshape[3, :] == 2]
subsidized_2040_class4 = subsidized_2040[income_class_2040_reshape[3, :] == 3]

backyard_2011_class1 = backyard_2011[income_class_2011_reshape[1, :] == 0]
backyard_2011_class2 = backyard_2011[income_class_2011_reshape[1, :] == 1]
backyard_2011_class3 = backyard_2011[income_class_2011_reshape[1, :] == 2]
backyard_2011_class4 = backyard_2011[income_class_2011_reshape[1, :] == 3]

backyard_2040_class1 = backyard_2040[income_class_2040_reshape[1, :] == 0]
backyard_2040_class2 = backyard_2040[income_class_2040_reshape[1, :] == 1]
backyard_2040_class3 = backyard_2040[income_class_2040_reshape[1, :] == 2]
backyard_2040_class4 = backyard_2040[income_class_2040_reshape[1, :] == 3]

informal_2011_class1 = informal_2011[income_class_2011_reshape[2, :] == 0]
informal_2011_class2 = informal_2011[income_class_2011_reshape[2, :] == 1]
informal_2011_class3 = informal_2011[income_class_2011_reshape[2, :] == 2]
informal_2011_class4 = informal_2011[income_class_2011_reshape[2, :] == 3]

informal_2040_class1 = informal_2040[income_class_2040_reshape[2, :] == 0]
informal_2040_class2 = informal_2040[income_class_2040_reshape[2, :] == 1]
informal_2040_class3 = informal_2040[income_class_2040_reshape[2, :] == 2]
informal_2040_class4 = informal_2040[income_class_2040_reshape[2, :] == 3]

# Total

array_2011 = np.vstack(
    [formal_2011, backyard_2011, informal_2011, subsidized_2011])
subset_2011 = array_2011[~np.isnan(array_2011[:, 1])]
array_2040 = np.vstack(
    [formal_2040, backyard_2040, informal_2040, subsidized_2040])
subset_2040 = array_2040[~np.isnan(array_2040[:, 1])]
sns.displot(
    subset_2011[:, 1], bins=np.arange(0, 0.7, 0.01), hist=True, kde=False,
    hist_kws={'weights': subset_2011[:, 0]}, color='black', label="2011")
sns.displot(
    subset_2040[:, 1], bins=np.arange(0, 0.7, 0.01), hist=True, kde=False,
    hist_kws={'weights': subset_2040[:, 0]}, label="2040")
plt.legend()
# plt.ylim(0, 320000)
# plt.ylim(0, 50000)
plt.xlabel("Share of the annual income destroyed by floods - annualized (%)")
plt.ylabel("Number of households")

# Class 1
array_2011 = np.vstack([formal_2011_class1, backyard_2011_class1,
                       informal_2011_class1, subsidized_2011_class1])
subset_2011 = array_2011[~np.isnan(array_2011[:, 1])]
array_2040 = np.vstack([formal_2040_class1, backyard_2040_class1,
                       informal_2040_class1, subsidized_2040_class1])
subset_2040 = array_2040[~np.isnan(array_2040[:, 1])]
sns.displot(
    subset_2011[:, 1], bins=np.arange(0, 0.7, 0.01), hist=True, kde=False,
    hist_kws={'weights': subset_2011[:, 0]}, color='black', label="2011")
sns.displot(
    subset_2040[:, 1], bins=np.arange(0, 0.7, 0.01), hist=True, kde=False,
    hist_kws={'weights': subset_2040[:, 0]}, label="2040")
plt.legend()
plt.xlabel("Share of the annual income destroyed by floods - annualized (%)")
plt.ylabel("Number of households")

# Class 2
array_2011 = np.vstack([formal_2011_class2, backyard_2011_class2,
                       informal_2011_class2, subsidized_2011_class2])
subset_2011 = array_2011[~np.isnan(array_2011[:, 1])]
array_2040 = np.vstack([formal_2040_class2, backyard_2040_class2,
                       informal_2040_class2, subsidized_2040_class2])
subset_2040 = array_2040[~np.isnan(array_2040[:, 1])]
sns.displot(
    subset_2011[:, 1], bins=np.arange(0, 0.7, 0.01), hist=True, kde=False,
    hist_kws={'weights': subset_2011[:, 0]}, color='black', label="2011")
sns.displot(
    subset_2040[:, 1], bins=np.arange(0, 0.7, 0.01), hist=True, kde=False,
    hist_kws={'weights': subset_2040[:, 0]}, label="2040")
plt.legend()
plt.xlabel("Share of the annual income destroyed by floods - annualized (%)")
plt.ylabel("Number of households")

# Class 3
array_2011 = np.vstack([formal_2011_class3, backyard_2011_class3,
                       informal_2011_class3, subsidized_2011_class3])
subset_2011 = array_2011[~np.isnan(array_2011[:, 1])]
array_2040 = np.vstack([formal_2040_class3, backyard_2040_class3,
                       informal_2040_class3, subsidized_2040_class3])
subset_2040 = array_2040[~np.isnan(array_2040[:, 1])]
sns.displot(
    subset_2011[:, 1], bins=np.arange(0, 0.7, 0.01), hist=True, kde=False,
    hist_kws={'weights': subset_2011[:, 0]}, color='black', label="2011")
sns.displot(
    subset_2040[:, 1], bins=np.arange(0, 0.7, 0.01), hist=True, kde=False,
    hist_kws={'weights': subset_2040[:, 0]}, label="2040")
plt.legend()
plt.xlabel("Share of the annual income destroyed by floods - annualized (%)")
plt.ylabel("Number of households")

# Class 4
array_2011 = np.vstack([formal_2011_class4, backyard_2011_class4,
                       informal_2011_class4, subsidized_2011_class4])
subset_2011 = array_2011[~np.isnan(array_2011[:, 1])]
array_2040 = np.vstack([formal_2040_class4, backyard_2040_class4,
                       informal_2040_class4, subsidized_2040_class4])
subset_2040 = array_2040[~np.isnan(array_2040[:, 1])]
sns.displot(
    subset_2011[:, 1], bins=np.arange(0, 0.7, 0.01), hist=True, kde=False,
    hist_kws={'weights': subset_2011[:, 0]}, color='black', label="2011")
sns.displot(
    subset_2040[:, 1], bins=np.arange(0, 0.7, 0.01), hist=True, kde=False,
    hist_kws={'weights': subset_2040[:, 0]}, label="2040")
plt.legend()
plt.xlabel("Share of the annual income destroyed by floods - annualized (%)")
plt.ylabel("Number of households")
