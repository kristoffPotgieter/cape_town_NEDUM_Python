# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:31:00 2020.

@author: Charlotte Liotta
"""

# %% Preamble


# IMPORT PACKAGES

import pandas as pd
import numpy as np
import numpy.matlib
import scipy.io
from sklearn.linear_model import LinearRegression
import os
import math
import seaborn as sns

import inputs.parameters_and_options as inpprm
import inputs.data as inpdt

import equilibrium.functions_dynamic as eqdyn

import calibration.compute_income as calcmp
import calibration.import_employment_data as calemp
import calibration.estimate_parameters_by_scanning as calscan
import calibration.estimate_parameters_by_optimization as calopt
import calibration.import_amenities as calam

import outputs.export_outputs as outexp


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/2. Data/'
path_precalc_inp = path_folder + '0. Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_folder + 'data_Cape_Town/Scenarios/'
path_outputs = path_code + '/4. Sorties/'
path_floods = path_folder + "FATHOM/"


# %% Import parameters and options


# IMPORT DEFAULT PARAMETERS AND OPTIONS

options = inpprm.import_options()
param = inpprm.import_param(path_precalc_inp, path_outputs)

# OPTIONS FOR THIS SIMULATION

options["agents_anticipate_floods"] = 0
options["informal_land_constrained"] = 0
options["convert_sal_data"] = 0
options["compute_net_income"] = 0
options["actual_backyards"] = 0
options["unempl_reweight"] = 1
options["correct_agri_rent"] = 1

t = (0)


# %% Load data

# BASIC GEOGRAPHIC DATA

grid, center = inpdt.import_grid(path_data)
amenities = inpdt.import_amenities(path_precalc_inp)


# MACRO DATA

(interest_rate, population, housing_type_data, total_RDP
 ) = inpdt.import_macro_data(param, path_scenarios)


# HOUSEHOLDS AND INCOME DATA

income_class_by_housing_type = inpdt.import_hypothesis_housing_type()

# See appendix A1
(mean_income, households_per_income_class, average_income, income_mult,
 income_2011, households_per_income_and_housing
 ) = inpdt.import_income_classes_data(param, path_data)

#  We create this parameter to maintain money illusion in simulations
#  (see eqsim.run_simulation)
#  TODO: Set as a variable, not a parameter
param["income_year_reference"] = mean_income

(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(path_precalc_inp)

#  Import nb of households per pixel, by housing type
#  Note that RDP is included in foraml, and there are both formal and informal
#  backyards

if options["convert_sal_data"] == 1:
    housing_types = inpdt.import_sal_data(grid, path_folder, path_data,
                                          housing_type_data)

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
    interest_rate, options
    )

# FLOOD DATA (takes some time when agents anticipate floods)
#  TODO: create a new variable instead of storing in param
#  TODO: check if WBUS2 data is indeed deprecated
#  param = inpdt.infer_WBUS2_depth(housing_types, param, path_floods)
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

#  Import income net of commuting costs, as calibrated in Pfeiffer et al.
#  (see part 3.1 or appendix C3)

if options["compute_net_income"] == 1:
    for t_temp in t:
        print(t_temp)
        (incomeNetOfCommuting, modalShares, ODflows, averageIncome
         ) = inpdt.import_transport_data(
             grid, param, t_temp, households_per_income_class, average_income,
             spline_inflation, spline_fuel,
             spline_population_income_distribution, spline_income_distribution,
             path_precalc_inp, path_precalc_transp, 'SP')

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'SP_incomeNetOfCommuting_0.npy')


# %% Estimation of coefficients of construction function (Cobb-Douglas)

# We associate income group to each census block according to average income
# TODO: is it the right way to define dominant income group? Shouldn't we take
# the most numerous group instead?
# data_income_group = np.zeros(len(data_sp["income"]))
# for j in range(0, 3):
#     data_income_group[data_sp["income"] >
#                       threshold_income_distribution[j]] = j+1
data_income_group = np.zeros(len(income_distribution))
for i in range(0, len(income_distribution)):
    data_income_group[i] = np.argmax(income_distribution[i])

# We get the number of formal housing units per SP
# TODO: should we substract RDP?

grid_intersect = pd.read_csv(path_data + 'grid_SP_intersect.csv', sep=';')
# grid_intersect = grid_intersect.set_index('ID_grille')
# grid_intersect = grid_intersect.sort_index()
grid_intersect = grid_intersect.groupby('ID_grille').max('Area')
data_rdp["ID_grille"] = data_rdp.index
data_rdp["ID_grille"] = data_rdp["ID_grille"] + 1

rdp_grid = pd.merge(data_rdp, grid_intersect, on="ID_grille", how="outer")
rdp_sp = rdp_grid.groupby('SP_CODE')['count'].sum()
rdp_sp = rdp_sp.reset_index()
rdp_sp = rdp_sp.rename(columns={'SP_CODE': 'sp_code'})
rdp_sp_fill = pd.merge(rdp_sp, data_sp['sp_code'], on="sp_code", how="outer")
rdp_sp_fill['count'] = rdp_sp_fill['count'].fillna(0)
rdp_sp_fill = rdp_sp_fill.sort_values(by='sp_code')

data_number_formal = (
    housing_types_sp.total_dwellings_SP_2011
    - housing_types_sp.backyard_SP_2011
    - housing_types_sp.informal_SP_2011
    - rdp_sp_fill['count'])

# data_number_formal = (
#     housing_types_sp.total_dwellings_SP_2011
#     - housing_types_sp.backyard_SP_2011
#     - housing_types_sp.informal_SP_2011
#     )

# We select the data points we are going to use.
# As Cobb-Douglas log-linear relation is only true for the formal sector, we
# exclude SPs in the bottom quintile of property prices and for which more
# than 5% of dwellings are reported to live in "informal" housing. We also
# exclude "rural" SPs (i.e., those that are large, with a small share than can
# be urbanized).

# NB: we also add other criteria compared to the working paper, namely we
# exclude poorest income group (which is in effect crowded out from the formal
# sector), as well as Mitchell's Plain (as its housing market is very specific)
# and far-away land (for which we have few observations)
# TODO: should we really exclude dominant income group?

# TODO: makes sense to calibrate without specific neighborhoods to avoid
# overfitting?

# selected_density = (
#     (data_sp["price"] > np.nanquantile(data_sp["price"], 0.2))
#     & (data_number_formal > 0.95 * housing_types_sp.total_dwellings_SP_2011)
#     & (data_sp["unconstrained_area"]
#         < np.nanquantile(data_sp["unconstrained_area"], 0.8))
#     & (data_sp["unconstrained_area"] > 0.6 * 1000000 * data_sp["area"])
#     & (data_income_group > 0)
#     & (data_sp["mitchells_plain"] == 0)
#     & (data_sp["distance"] < 40)
#     )

selected_density = (
    (data_sp["price"] > np.nanquantile(data_sp["price"], 0.2))
    & (data_number_formal > 0.95 * housing_types_sp.total_dwellings_SP_2011)
    & (data_sp["unconstrained_area"]
        < np.nanquantile(data_sp["unconstrained_area"], 0.8))
    & (data_sp["unconstrained_area"] > 0.6 * 1000000 * data_sp["area"])
    )

# We run regression from apppendix C2

y = np.log(data_number_formal[selected_density])

# Note that we use data_sp["unconstrained_area"] (which is accurate data at SP
# level) rather than coeff_land (which is an estimate at grid level)
X = np.transpose(
    np.array([np.log(data_sp["price"][selected_density]),
              np.log(param["max_land_use"]
                     * data_sp["unconstrained_area"][selected_density]),
              np.log(data_sp["dwelling_size"][selected_density])])
    )

model_construction = LinearRegression().fit(X, y)
# model_construction.score(X, y)
# model_construction.coef_
# model_construction.intercept_

# We export outputs of the model
coeff_b = model_construction.coef_[0]
coeff_a = 1 - coeff_b
# Comes from zero profit condition combined with footnote 16 from optimization
# TODO: correct typo in paper
coeffKappa = ((1 / (coeff_b / coeff_a) ** coeff_b)
              * np.exp(model_construction.intercept_))
# coeffKappa = ((1 / (coeff_b) ** coeff_b)
#               * np.exp(model_construction.intercept_))

try:
    os.mkdir(path_precalc_inp)
except OSError as error:
    print(error)

np.save(path_precalc_inp + 'calibratedHousing_b.npy', coeff_b)
np.save(path_precalc_inp + 'calibratedHousing_kappa.npy', coeffKappa)


# %% Estimation of incomes and commuting parameters

# TODO: refine choice of inputs!

# We input a range of values that we would like to test for lambda (gravity
# parameter)

# The value range is set by trial and error: the wider the range you want to
# test, the longer
# listLambda = [4.027, 0]
# list_lambda = 10 ** np.arange(0.6, 0.65, 0.01)
list_lambda = 10 ** np.arange(0.6, 0.61, 0.005)

# Note that number of employees per income group imported from job center data
# is rescaled to match aggregate income distribution in census data
job_centers = calemp.import_employment_data(
    households_per_income_class, param, path_data)

# Note that we reason at the SP level here. Also note that we are considering
# round trips and households made of two representative individuals

(timeOutput, distanceOutput, monetaryCost, costTime
 ) = calcmp.import_transport_costs(grid, param, 0, households_per_income_class,
                                   spline_inflation, spline_fuel,
                                   spline_population_income_distribution,
                                   spline_income_distribution,
                                   path_precalc_inp, path_precalc_transp, 'SP')
# (timeOutput, distanceOutput, monetaryCost, costTime
#  ) = calcmpt.import_transport_costs(
#      income_2011, param, grid, path_precalc_inp, path_scenarios)


# Note that this is long to run
# Here again, we are considering rescaled income data

# TODO: how to explain poor convergence with doubling of transport times?
incomeCenters, distanceDistribution = calcmp.EstimateIncome(
    param, timeOutput, distanceOutput[:, :, 0], monetaryCost, costTime,
    job_centers, average_income, income_distribution, list_lambda)
# incomeCenters, distanceDistribution = calcmpt.EstimateIncome(
#     param, timeOutput, distanceOutput, monetaryCost, costTime, job_centers,
#     average_income, income_distribution, list_lambda)


# Gives aggregate statistics for % of commuters per distance bracket
# TODO: where from?
# data_modal_shares = np.array(
#     [7.8, 14.8, 39.5+0.7, 16, 8]) / (7.8+14.8+39.5+0.7+16+8) * 100
# data_time_distribution = np.array(
#     [18.3, 32.7, 35.0, 10.5, 3.4]) / (18.3+32.7+35.0+10.5+3.4)
data_distance_distribution = np.array(
    [45.6174222, 18.9010734, 14.9972971, 9.6725616, 5.9425438, 2.5368754,
     0.9267125, 0.3591011, 1.0464129])

# Compute accessibility index
# NB1: Bhattacharyya distance measures the similarity of two probability
# distributions (here, data vs. simulated % of commuters)
# NB2: Mahalanobis distance is a particular case of the Bhattacharyya distance
# when the standard deviations of the two classes are the same
bhattacharyyaDistances = (
    - np.log(np.nansum(np.sqrt(data_distance_distribution[:, None]
                               / 100 * distanceDistribution), 0))
    )
whichLambda = np.argmin(bhattacharyyaDistances)

# Hence, we keep the lambda that minimizes the distance and the associated
# income vector
# TODO: correct typo in paper
lambdaKeep = list_lambda[whichLambda]
# modalSharesKeep = modalShares[:, whichLambda]
# timeDistributionKeep = timeDistribution[:, whichLambda]
distanceDistributionKeep = distanceDistribution[:, whichLambda]
incomeCentersKeep = incomeCenters[:, :, whichLambda]

# Note that income is set to -inf for job centers and income groups in which
# it could not be calibrated

np.save(path_precalc_inp + 'incomeCentersKeep.npy', incomeCentersKeep)
# 4.027 as in paper
np.save(path_precalc_inp + 'lambdaKeep.npy', lambdaKeep)


# Plot difference with initial input from Matlab
incomeCentersKeep_mat = scipy.io.loadmat(
    path_precalc_inp + 'incomeCentersKeep.mat')['incomeCentersKeep']
sns.distplot(
    np.abs((
        (incomeCentersKeep - incomeCentersKeep_mat) / incomeCentersKeep_mat
        ) * 100)
    )

# lambdaKeep = 10 ** 0.605

# NB: what about equilibrium condition (v)?
# It holds by construction for SP, but what about simulated pixels?

# TODO: note that we are actually comparing incomes with wages?

incomeCentersKeep[incomeCentersKeep < 0] = math.nan
cal_avg_income = np.nanmean(incomeCentersKeep, 0)

incomeCentersKeep_mat[incomeCentersKeep_mat < 0] = math.nan
cal_avg_income_mat = np.nanmean(incomeCentersKeep_mat, 0)

# Actually our simulation works better!


# %% Calibration of utility function parameters

# We select in which areas we actually measure the likelihood
# NB: We remove the areas where there is informal housing, because dwelling
# size data is not reliable
# TODO: why not use same criteria as for selected_density?

selectedSP = (
    (data_number_formal > 0.90 * housing_types_sp.total_dwellings_SP_2011)
    & (data_income_group > 0)
    )

# Coefficients of the model for simulations (arbitrary)
listBeta = np.arange(0.2, 0.35, 0.05)
listBasicQ = np.arange(2, 7, 1)

# Coefficient for spatial autocorrelation
# TODO: how would this work if implemented?
listRho = 0

# Utilities for simulations (arbitrary)
# TODO: why not start with same set as in compute_equilibrium?
# utilityTarget = np.array([300, 1000, 3000, 10000])
utilityTarget = np.array([1501, 4819, 16947, 79809])

# TODO: meaning?
listVariation = np.arange(0.8, 1.3, 0.2)
initUti2 = utilityTarget[1]
listUti3 = utilityTarget[2] * listVariation
listUti4 = utilityTarget[3] * listVariation

# Cf. inversion of footnote 16
# TODO: should we use param["interest_rate"]?
# TODO: correct typo in paper
dataRent = (
    data_sp["price"] ** (coeff_a)
    * (param["depreciation_rate"]
       + interest_rate)
    / (coeffKappa * coeff_b ** coeff_b * coeff_a ** coeff_a)
    )
# dataRent = (
#     data_sp["price"] ** (coeff_a)
#     * (param["depreciation_rate"]
#        + eqdyn.interpolate_interest_rate(spline_interest_rate, 0))
#     / (coeffKappa * coeff_b ** coeff_b * coeff_a ** coeff_a)
#     )

# We get the built density in associated buildable land (per km²)
data_density = (
    data_number_formal
    / (data_sp["unconstrained_area"] * param["max_land_use"] / 1000000)
    )


# Import amenities at the SP level

amenities_sp = calam.import_amenities(path_data, path_precalc_inp, 'SP')
# We select amenity variables to be used in regression from table C5
# NB: choice has to do with relevance and exogenity of variables
variables_regression = [
    'distance_ocean', 'distance_ocean_2_4', 'slope_1_5', 'slope_5',
    'airport_cone2', 'distance_distr_parks', 'distance_biosphere_reserve',
    'distance_train', 'distance_urban_herit']

# Note that this is long to run as it depends on the combination of all inputs

(parametersScan, scoreScan, parametersAmenitiesScan, modelAmenityScan,
 parametersHousing, _) = calscan.EstimateParametersByScanning(
     incomeNetOfCommuting, dataRent, data_sp["dwelling_size"],
     data_income_group, data_density, selected_density,
     housing_types_sp["x_sp"], housing_types_sp["y_sp"], selectedSP,
     amenities_sp, variables_regression, listRho, listBeta, listBasicQ,
     initUti2, listUti3, listUti4)

# Now run the optimization algo with identified value of the parameters:
# corresponds to interior-point algorithm

# Note that we are not equal to the paper: what values to keep?
initBeta = parametersScan[0]
# TODO: meaning?
# initBasicQ = max(parametersScan[1], 5.1)
initBasicQ = parametersScan[1]

# Utilities
initUti3 = parametersScan[2]
initUti4 = parametersScan[3]

# TODO: should we run it with true values of parametersScan
(parameters, scoreTot, parametersAmenities, modelAmenity, parametersHousing,
 selectedSPRent) = calopt.EstimateParametersByOptimization(
     incomeNetOfCommuting, dataRent, data_sp["dwelling_size"],
     data_income_group, data_density, selected_density,
     housing_types_sp["x_sp"], housing_types_sp["y_sp"], selectedSP,
     amenities_sp, variables_regression, listRho, initBeta, initBasicQ,
     initUti2, initUti3, initUti4)

# TODO: we do not get same parameters as in paper / mat files, how to validate?

# Exporting and saving

#  Generating the map of amenities?
#  TODO: Note that this a problem with dummies

amenities_grid = calam.import_amenities(path_data, path_precalc_inp, 'grid')
predictors_grid = amenities_grid.loc[:, variables_regression]
predictors_grid = np.vstack(
    [np.ones(predictors_grid.shape[0]),
     predictors_grid.T]
    ).T

cal_amenities = np.exp(np.nansum(predictors_grid * parametersAmenities, 1))
calw_amenities = cal_amenities / np.nanmean(cal_amenities)
outexp.export_map(calw_amenities, grid, path_outputs + 'amenity_map', 1.3, 0.8)

modelAmenity.save(path_precalc_inp + 'modelAmenity')
# utilitiesCorrected = parameters[3:] / np.exp(parametersAmenities[1])
calibratedUtility_beta = parameters[0]
calibratedUtility_q0 = parameters[1]

# 0.1 vs. 0.25
np.save(path_precalc_inp + 'calibratedUtility_beta', calibratedUtility_beta)
# 3.0 vs. 4.1
np.save(path_precalc_inp + 'calibratedUtility_q0', calibratedUtility_q0)
# Hard to tell according to map
np.save(path_precalc_inp + 'calibratedAmenities', cal_amenities)

# Other tests

outexp.export_map(amenities, grid, path_outputs + 'precalc_amenity_map',
                  1.3, 0.8)
outexp.export_map(amenities, grid, path_outputs + 'precalc_amenity_map',
                  1.3, 0.8)

income_centers_precalc = scipy.io.loadmat(
    path_precalc_inp + 'incomeCentersKeep.mat')['incomeCentersKeep']
income_centers_precalc[income_centers_precalc == -np.inf] = np.nan
income_centers_precalc_w = (
    income_centers_precalc / np.nanmean(income_centers_precalc))
income_centers = np.load(path_precalc_inp + 'incomeCentersKeep.npy')
income_centers[income_centers == -np.inf] = np.nan
income_centers_w = income_centers / np.nanmean(income_centers)

# outexp.export_map(income_centers_precalc_w[:, 0], grid,
#                   path_outputs + 'precalc_income_rich_map',
#                   0, 5.5)
# outexp.export_map(income_centers_w[:, 0], grid,
#                   path_outputs + 'income_rich_map',
#                   0, 5.5)

outexp.validation_cal_income(
    path_data, path_outputs, center, income_centers[:, 0],
    income_centers_precalc[:, 0])
