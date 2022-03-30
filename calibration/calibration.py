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
import copy
import scipy.io
from sklearn.linear_model import LinearRegression
import seaborn as sns

import inputs.parameters_and_options as inpprm
import inputs.data as inpdt
# import inputs.WBUS2_depth as inpwbu

import equilibrium.compute_equilibrium as eqcmp
import equilibrium.functions_dynamic as eqdyn
import equilibrium.run_simulations as eqsim

import outputs.flood_outputs as outfld
import outputs.export_outputs_floods as outval
import outputs.export_outputs as outexp

import calibration.compute_income as calcmp


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/2. Data/'
path_precalc_inp = path_folder + '0. Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_folder + 'data_Cape_Town/Scenarios/'
path_outputs = path_code + '/4. Sorties/'
path_floods = path_folder + "FATHOM/"


print("\n*** NEDUM-Cape-Town - Construction function calibration ***\n")


# %% Import parameters and options

print("\n*** Load parameters and options ***\n")


# IMPORT DEFAULT PARAMETERS AND OPTIONS

options = inpprm.import_options()
param = inpprm.import_param(path_precalc_inp, path_outputs)

# TODO: Set timeline for what?
t = np.arange(0, 1)

# OPTIONS FOR THIS SIMULATION
# TODO: check if still relevant
# options["coeff_land"] = 'old'
# options["WBUS2"] = 1


# %% Load data

print("\n*** Load data ***\n")


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

#  TODO: Why do we need this correction?
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
incomeNetOfCommuting = np.load(
    path_precalc_transp + 'incomeNetOfCommuting_0.npy')


# %% Estimation of coefficient of construction function

# We associate income group to each census block according to average income
data_income_group = np.zeros(len(data_sp["income"]))
for j in range(0, 3):
    data_income_group[data_sp["income"] >
                      threshold_income_distribution[j]] = j+1

# We get the number of formal housing units per SP
data_number_formal = (
    housing_types_sp.total_dwellings_SP_2011
    - housing_types_sp.backyard_SP_2011
    - housing_types_sp.informal_SP_2011)

# We then deduce the built density in associated buildable land (per km²)
data_density = (
    data_number_formal
    / (data_sp["unconstrained_area"] * param["max_land_use"] / 1000000)
    )

# We select the data points we are going to use.
# As Cobb-Douglas log-linear relation is only true for the formal sector, we
# exclude SPs in the bottom quintile of property prices and for which more
# than 5% of dwellings are reported to live in informal housing. We also
# exclude rural SPs (i.e., those that are large, with a small share than can
# be urbanized)
# TODO: Does this correspond?
selected_density = (
    (data_sp["unconstrained_area"] > 0.6 * 1000000 * data_sp["area"])
    & (data_income_group > 0)
    & (data_sp["mitchells_plain"] == 0)
    & (data_sp["distance"] < 40)
    & (data_sp["price"] > np.nanquantile(data_sp["price"], 0.2))
    & (data_sp["unconstrained_area"]
       < np.nanquantile(data_sp["unconstrained_area"], 0.8))
    )

# We run regression from apppendix C2

y = np.log(data_number_formal[selected_density])

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
coeffKappa = ((1 / (coeff_b / coeff_a) ** coeff_b)
              * np.exp(model_construction.intercept_))

# TODO: What about CES parameters?
# TODO: save results!

# # Correcting data for rents
# dataRent = (
#     dataPrice ** (coeff_a)
#     * (param["depreciation_rate"]
#        + InterpolateInterestRateEvolution(macro_data, t[0]))
#     / (coeffKappa * coeff_b ** coeff_b)
#     )
# interestRate = (param["depreciation_rate"]
#                 + InterpolateInterestRateEvolution(macro_data, t[0]))

# # Cobb-Douglas:
# simulHousing_CD = (
#     coeffKappa ** (1/coeff_a) * (coeff_b/interestRate) ** (coeff_b/coeff_a)
#     * (dataRent) ** (coeff_b/coeff_a)
#     )

# f1 = fit(data.sp2011Distance(selectedDensity),
#          data.spFormalDensityHFA(selectedDensity), 'poly5')
# f2 = fit(data.sp2011Distance(~isnan(simulHousing_CD)),
#          simulHousing_CD(~isnan(simulHousing_CD)), 'poly5')


# %% Estimation of incomes and commuting parameters

# We input a range of values that we would like to test for lambda (gravity
# parameter)
# TODO: how arbitray is it?

# listLambda = [4.027, 0]
# list_lambda = 10 ** np.arange(0.6, 0.65, 0.01)
list_lambda = 10 ** np.arange(0.6, 0.605, 0.005)

(timeOutput, distanceOutput, monetaryCost, costTime
 ) = calcmp.import_transport_costs(
     income_2011, param, grid, 0, path_precalc_inp, spline_inflation,
     spline_fuel)

job_centers = calcmp.import_employment_data(
    households_per_income_class, param, path_data)

incomeCenters, distanceDistribution = calcmp.EstimateIncome(
    param, timeOutput, distanceOutput, monetaryCost, costTime, job_centers,
    average_income, income_distribution, list_lambda)

###

data_modal_shares = np.array(
    [7.8, 14.8, 39.5+0.7, 16, 8]) / (7.8+14.8+39.5+0.7+16+8) * 100
data_time_distribution = np.array(
    [18.3, 32.7, 35.0, 10.5, 3.4]) / (18.3+32.7+35.0+10.5+3.4)
data_distance_distribution = np.array(
    [45.6174222, 18.9010734, 14.9972971, 9.6725616, 5.9425438, 2.5368754,
     0.9267125, 0.3591011, 1.0464129])

# Compute accessibility index
bhattacharyyaDistances = (
    - np.log(np.nansum(np.sqrt(data_distance_distribution[:, None]
                               / 100 * distanceDistribution), 0))
    )
whichLambda = np.argmin(bhattacharyyaDistances)

lambdaKeep = list_lambda[whichLambda]
# modalSharesKeep = modalShares[:, whichLambda]
# timeDistributionKeep = timeDistribution[:, whichLambda]
distanceDistributionKeep = distanceDistribution[:, whichLambda]
incomeCentersKeep = incomeCenters[:, :, whichLambda]
incomeCentersKeep_mat = scipy.io.loadmat('incomeCentersKeep.mat')[
    'incomeCentersKeep']
sns.distplot(
    np.abs((
        (incomeCentersKeep - incomeCentersKeep_mat) / incomeCentersKeep_mat
        ) * 100)
    )

# lamdbaKeep = 10 ** 0.605


# %% Calibration utility function

# In which areas we actually measure the likelihood
#  NB: We remove the areas where there is informal housing, because dwelling
#  size data is not reliable
selectedSP = (
    ((housing_types_sp.backyard_SP_2011 + housing_types_sp.informal_SP_2011)
     / housing_types_sp.total_dwellings_SP_2011 < 0.1)
    & (data_income_group > 0)
    )

# Coefficients of the model
listBeta = np.arange(0.1, 0.55, 0.2)
listBasicQ = np.arange(5, 16, 5)

# Utilities
utilityTarget = np.array([300, 1000, 3000, 10000])
listVariation = np.arange(0.5, 2, 0.3)
initUti2 = utilityTarget[1]
listUti3 = utilityTarget[2] * listVariation
listUti4 = utilityTarget[3] * listVariation

dataRent = (
    data_sp["price"] ** (coeff_a)
    * (param["depreciation_rate"]
       + interpolate_interest_rate(spline_interest_rate, 0))
    / (coeffKappa * coeff_b ** coeff_b)
    )

(parametersScan, scoreScan, parametersAmenitiesScan, modelAmenityScan,
 parametersHousing, _) = EstimateParametersByScanning(
     incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup,
     dataDensity, selectedDensity, xData, yData, selectedSPForEstimation,
     tableAmenities, variablesRegression, listRho, listBeta, listBasicQ,
     initUti2, listUti3, listUti4)

# Now run the optimization algo with identified value of the parameters
initBeta = parametersScan[0]
initBasicQ = max(parametersScan[1], 5.1)

# Utilities
initUti3 = parametersScan[2]
initUti4 = parametersScan[3]

(parameters, scoreTot, parametersAmenities, modelAmenity, parametersHousing,
 selectedSPRent) = EstimateParametersByOptimization(
     incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup,
     dataDensity, selectedDensity, xData, yData, selectedSPForEstimation,
     tableAmenities, variablesRegression, listRho, initBeta, initBasicQ,
     initUti2, initUti3, initUti4)

# Generating the map of amenities

modelAmenity.save(path_precalc_inp + 'modelAmenity')
# ImportAmenitiesGrid
# amenities = exp(parametersAmenities[2:]
#                 * table2array(tableAmenitiesGrid[:, variablesRegression]))

# Exporting and saving
utilitiesCorrected = parameters[3:] / np.exp(parametersAmenities[1])
calibratedUtility_beta = parameters(1)
calibratedUtility_q0 = parameters(2)










# TODO: Put in side script


def import_amenities_SP():
    """Import relevant amenity data at SP level."""
    # Load aemnity files
    amenity_data = pd.read_csv(path_data + 'SP_amenities.csv', sep=',')

    # We replace values for airport cone amenities with a dummy for being
    # located in the airport cone
    airport_cone = copy.deepcopy(amenity_data.airport_cone)
    airport_cone[airport_cone == 55] = 1
    airport_cone[airport_cone == 60] = 1
    airport_cone[airport_cone == 65] = 1
    airport_cone[airport_cone == 70] = 1
    airport_cone[airport_cone == 75] = 1

    # Load distance to RDP house dummies
    SP_distance_RDP = scipy.io.loadmat(
        path_precalc_inp + 'SPdistanceRDP.mat')["SP_distance_RDP"].squeeze()

    # We store relevant data in an output table
    # NB: we only consider dummies for amenity data crossing some thresholds.
    # This is done by trial and error to simplify calibration process
    table_amenities = pd.DataFrame(
        data=np.transpose(np.array(
            [amenity_data.SP_CODE,
             amenity_data.distance_distr_parks < 2,
             amenity_data.distance_ocean < 2,
             ((amenity_data.distance_ocean > 2)
              & (amenity_data.distance_ocean < 4)),
             amenity_data.distance_world_herit < 2,
             ((amenity_data.distance_world_herit > 2)
              & (amenity_data.distance_world_herit < 4)),
             amenity_data.distance_urban_herit < 2,
             amenity_data.distance_UCT < 2,
             airport_cone,
             ((amenity_data.slope > 1) & (amenity_data.slope < 5)),
             amenity_data.slope > 5,
             amenity_data.distance_train < 2,
             amenity_data.distance_protected_envir < 2,
             ((amenity_data.distance_protected_envir > 2)
              & (amenity_data.distance_protected_envir < 4)),
             SP_distance_RDP,
             amenity_data.distance_power_station < 2,
             amenity_data.distance_biosphere_reserve < 2])
            ),
        columns=['SP_CODE', 'distance_distr_parks', 'distance_ocean',
                 'distance_ocean_2_4', 'distance_world_herit',
                 'distance_world_herit_2_4', 'distance_urban_herit',
                 'distance_UCT', 'airport_cone2', 'slope_1_5', 'slope_5',
                 'distance_train', 'distance_protected_envir',
                 'distance_protected_envir_2_4', 'RDP_proximity',
                 'distance_power_station', 'distance_biosphere_reserve']
        )

    return table_amenities


# Import amenities at the SP level
amenities_sp = import_amenities_SP()
# We select amenity variables to be used in regression from table C5
# NB: choice has to do with relevance and exogenity of variables
variables_regression = [
    'distance_ocean', 'distance_ocean_2_4', 'slope_1_5', 'slope_5',
    'airport_cone2', 'distance_distr_parks', 'distance_biosphere_reserve',
    'distance_train', 'distance_urban_herit']
