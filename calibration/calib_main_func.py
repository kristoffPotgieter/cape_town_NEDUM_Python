# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:26:37 2022.

@author: monni
"""

# import pandas as pd
import numpy as np
import numpy.matlib
# import scipy.io
from sklearn.linear_model import LinearRegression
import os
import math

# import inputs.parameters_and_options as inpprm
# import inputs.data as inpdt

# import equilibrium.functions_dynamic as eqdyn

import calibration.sub.compute_income as calcmp
import calibration.sub.import_employment_data as calemp
import calibration.sub.estimate_parameters_by_scanning as calscan
import calibration.sub.estimate_parameters_by_optimization as calopt
import calibration.sub.import_amenities as calam

# import equilibrium.functions_dynamic as eqdyn

import outputs.export_outputs as outexp


def estim_construct_func_param(options, param, data_sp,
                               threshold_income_distribution,
                               income_distribution, data_rdp, housing_types_sp,
                               data_number_formal, data_income_group,
                               selected_density,
                               path_data, path_precalc_inp, path_folder):
    """Estimate coefficients of construction function (Cobb-Douglas)."""
    # We run regression from apppendix C2

    y = np.log(data_number_formal[selected_density])
    # Note that we use data_sp["unconstrained_area"] (which is accurate data at
    # SP level) rather than coeff_land (which is an estimate at grid level)
    X = np.transpose(
        np.array([np.log(data_sp["price"][selected_density]),
                  np.log(param["max_land_use"]
                         * data_sp["unconstrained_area"][selected_density]),
                  np.log(data_sp["dwelling_size"][selected_density])])
        )
    # NB: Our data set for dwelling sizes only provides the average (not
    # median) dwelling size at the Sub-Place level, aggregating formal and
    # informal housing

    model_construction = LinearRegression().fit(X, y)

    # We export outputs of the model
    coeff_b = model_construction.coef_[0]
    coeff_a = 1 - coeff_b
    # Comes from zero profit condition combined with footnote 16 from
    # optimization
    # TODO: correct typo in paper
    if options["correct_kappa"] == 1:
        coeffKappa = ((1 / (coeff_b / coeff_a) ** coeff_b)
                      * np.exp(model_construction.intercept_))
    elif options["correct_kappa"] == 0:
        coeffKappa = ((1 / (coeff_b) ** coeff_b)
                      * np.exp(model_construction.intercept_))

    try:
        os.mkdir(path_precalc_inp)
    except OSError as error:
        print(error)

    # TODO: why capital and land elasticities seem to be reversed compared
    # to the literature?
    np.save(path_precalc_inp + 'calibratedHousing_b.npy', coeff_b)
    np.save(path_precalc_inp + 'calibratedHousing_kappa.npy', coeffKappa)

    if options["reverse_elasticities"] == 1:
        coeff_a = coeff_b
        coeff_b = 1 - coeff_a
        if options["correct_kappa"] == 1:
            coeffKappa = ((1 / (coeff_b / coeff_a) ** coeff_b)
                          * np.exp(model_construction.intercept_))
        elif options["correct_kappa"] == 0:
            coeffKappa = ((1 / (coeff_b) ** coeff_b)
                          * np.exp(model_construction.intercept_))

    return coeff_b, coeff_a, coeffKappa


def estim_incomes_and_gravity(param, grid, list_lambda,
                              households_per_income_class,
                              average_income, income_distribution,
                              spline_inflation, spline_fuel,
                              spline_population_income_distribution,
                              spline_income_distribution,
                              path_data, path_precalc_inp,
                              path_precalc_transp, options):
    """Estimate incomes per job center and group and commuting parameter."""
    # We import number of workers in each selected job center
    # Note that it is rescaled to match aggregate income distribution in census
    job_centers = calemp.import_employment_data(
        households_per_income_class, param, path_data)

    # We import transport cost data.
    # Note that we reason at the SP level here. Also note that we are
    # considering round trips and households made of two representative
    # individuals
    (timeOutput, distanceOutput, monetaryCost, costTime
     ) = calcmp.import_transport_costs(
         grid, param, 0, households_per_income_class,
         spline_inflation, spline_fuel, spline_population_income_distribution,
         spline_income_distribution,
         path_precalc_inp, path_precalc_transp, 'SP', options)

    # Note that this is long to run
    # Here again, we are considering rescaled income data

    # TODO: get the errors
    (incomeCenters, distanceDistribution, scoreMatrix, errorMatrix
     ) = calcmp.EstimateIncome(
        param, timeOutput, distanceOutput[:, :, 0], monetaryCost, costTime,
        job_centers, average_income, income_distribution, list_lambda, options)

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
    # NB2: Mahalanobis distance is a particular case of the Bhattacharyya
    # distance when the standard deviations of the two classes are the same
    bhattacharyyaDistances = (
        - np.log(np.nansum(np.sqrt(data_distance_distribution[:, None]
                                   / 100 * distanceDistribution), 0))
        )
    whichLambda = np.argmin(bhattacharyyaDistances)

    # Hence, we keep the lambda that minimizes the distance and the associated
    # income vector
    lambdaKeep = list_lambda[whichLambda]
    # modalSharesKeep = modalShares[:, whichLambda]
    # timeDistributionKeep = timeDistribution[:, whichLambda]
    # distanceDistributionKeep = distanceDistribution[:, whichLambda]
    incomeCentersKeep = incomeCenters[:, :, whichLambda]

    scoreKeep = scoreMatrix[whichLambda, :]
    errorKeep = errorMatrix[whichLambda, :]

    # Note that income is set to -inf for job centers and income groups in
    # which it could not be calibrated

    np.save(path_precalc_inp + 'incomeCentersKeep.npy', incomeCentersKeep)
    np.save(path_precalc_inp + 'lambdaKeep.npy', lambdaKeep)

    # Validate with initial input from Matlab
    # incomeCentersKeep_mat = scipy.io.loadmat(
    #     path_precalc_inp + 'incomeCentersKeep.mat')['incomeCentersKeep']

    # Note that it is unclear whether "average" income from data includes
    # unemployment or not: a priori, it does for short spells (less than one
    # year) and should therefore be slightly bigger than calibrated income
    incomeCentersKeep[incomeCentersKeep < 0] = math.nan
    cal_avg_income = np.nanmean(incomeCentersKeep, 0)
    # incomeCentersKeep_mat[incomeCentersKeep_mat < 0] = math.nan
    # cal_avg_income_mat = np.nanmean(incomeCentersKeep_mat, 0)

    return incomeCentersKeep, lambdaKeep, cal_avg_income, scoreKeep, errorKeep


def estim_util_func_param(data_number_formal, data_income_group,
                          housing_types_sp, data_sp, grid,
                          coeff_a, coeff_b, coeffKappa, interest_rate,
                          incomeNetOfCommuting, selected_density,
                          path_data, path_precalc_inp, path_plots,
                          options, param):
    """Calibrate utility function parameters."""
    # We select in which areas we actually measure the likelihood
    # NB: We remove the areas where there is informal housing, because
    # dwelling size data is not reliable. This will be used to select
    # appropriate rents and dwelling sizes. Criterion is less stringent than
    # for density as added criteria are more specific to the use of
    # construction technology.
    # TODO: add option to change the selection
    selectedSP = (
        (data_number_formal > 0.90 * housing_types_sp.total_dwellings_SP_2011)
        & (data_income_group > 0)
        )

    # Coefficients of the model for simulations: set by trial and error
    #  This naturally tends to 0.2 without griddata, hence the reduced range
    # listBeta = np.arange(0.18, 0.221, 0.01)
    #  This tends to 0.12 with griddata (neighbors=10)
    # listBeta = np.arange(0.10, 0.141, 0.01)
    #  This tends to 0.13 with griddata (neighbors=100)
    listBeta = np.arange(0.12, 0.141, 0.01)

    #  This naturally tends to 0 without griddata, hence the floor which will
    #  be updated in later optimization
    # listBasicQ = np.arange(0.01, 0.1, 0.01)
    #  This tends to 2.8 with griddata (neighbors=10)
    # listBasicQ = np.arange(2.6, 3.01, 0.1)
    #  This tends to 3.4 with griddata (neighbors=100)
    listBasicQ = np.arange(3.3, 3.51, 0.1)

    # Coefficient for spatial autocorrelation
    # TODO: how would this work if implemented?
    listRho = 0

    # Utilities for simulations: we start with levels close from what we expect
    # in equilibrium
    # TODO: to be updated with values close to what we obtain in equilibrium
    # (to speed up convergence)
    utilityTarget = np.array([1500, 5000, 17000, 80000])

    # We scan varying values of utility targets
    #  This is also set by trial and error
    #  This converges towards 1.3 for U_3 and 1.1 for U_4 without griddata
    # listVariation = np.arange(1.0, 1.51, 0.1)
    #  This tends to 1.2 for U_3 and 1.1 for U_4 with griddata (neighbors=10)
    # listVariation = np.arange(0.9, 1.31, 0.1)
    #  This tends to 1.1 for U_3 and 1 for U_4 with griddata (neighbors=100)
    listVariation = np.arange(0.9, 1.21, 0.1)
    # Note that the poorest income group is not taken into account as it is
    # excluded from the analysis.
    # Then, we do not vary the first income group as the magnitude is not that
    # large.
    initUti2 = utilityTarget[1]
    # However, we do so for the two richest groups.
    listUti3 = utilityTarget[2] * listVariation
    # listUti3 = utilityTarget[2]
    listUti4 = utilityTarget[3] * listVariation

    # Cf. inversion of footnote 16
    # TODO: should we use param["interest_rate"]?
    if options["correct_kappa"] == 1:
        dataRent = (
            data_sp["price"] ** (coeff_a)
            * (param["depreciation_rate"]
               + interest_rate)
            / (coeffKappa * coeff_b ** coeff_b * coeff_a ** coeff_a)
            )
    elif options["correct_kappa"] == 0:
        dataRent = (
            data_sp["price"] ** (coeff_a)
            * (param["depreciation_rate"]
               + interest_rate)
            / (coeffKappa * coeff_b ** coeff_b)
            )

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

    # Note that this may be long to run as it depends on the combination of all
    # inputs

    (parametersScan, scoreScan, parametersAmenitiesScan, modelAmenityScan,
     parametersHousing, _) = calscan.EstimateParametersByScanning(
         incomeNetOfCommuting, dataRent, data_sp["dwelling_size"],
         data_income_group, data_density, selected_density,
         housing_types_sp["x_sp"], housing_types_sp["y_sp"], selectedSP,
         amenities_sp, variables_regression, listRho, listBeta, listBasicQ,
         initUti2, listUti3, listUti4, options)

    # Amenity results differ a bit from the paper, but are not absurd
    # (even though hard to interpret)
    print(modelAmenityScan.summary())

    # Now we run the optimization algo with identified value of the parameters:
    # corresponds to interior-point algorithm

    initBeta = parametersScan[0]
    initBasicQ = parametersScan[1]
    initUti3 = parametersScan[2]
    initUti4 = parametersScan[3]

    # Note that this may be long to run
    # TODO: should score be positive?
    (parameters, scoreTot, parametersAmenities, modelAmenity,
     parametersHousing, selectedSPRent
     ) = calopt.EstimateParametersByOptimization(
         incomeNetOfCommuting, dataRent, data_sp["dwelling_size"],
         data_income_group, data_density, selected_density,
         housing_types_sp["x_sp"], housing_types_sp["y_sp"], selectedSP,
         amenities_sp, variables_regression, listRho, initBeta, initBasicQ,
         initUti2, initUti3, initUti4, options)

    print(modelAmenityScan.summary())

    # Exporting and saving outputs

    amenities_grid = calam.import_amenities(path_data, path_precalc_inp,
                                            'grid')
    predictors_grid = amenities_grid.loc[:, variables_regression]
    predictors_grid = np.vstack(
        [np.ones(predictors_grid.shape[0]),
         predictors_grid.T]
        ).T

    cal_amenities = np.exp(np.nansum(predictors_grid * parametersAmenities, 1))
    calw_amenities = cal_amenities / np.nanmean(cal_amenities)
    outexp.export_map(calw_amenities, grid, path_plots + 'amenity_map',
                      1.3, 0.8)

    # Note that amenity map from GLM estimates yields absurd results, hence it
    # is not coded here
    if options["glm"] == 1:
        modelAmenity.save(path_precalc_inp + 'modelAmenity')
    calibratedUtility_beta = parameters[0]
    calibratedUtility_q0 = parameters[1]

    np.save(path_precalc_inp + 'calibratedUtility_beta',
            calibratedUtility_beta)
    np.save(path_precalc_inp + 'calibratedUtility_q0', calibratedUtility_q0)
    np.save(path_precalc_inp + 'calibratedAmenities', cal_amenities)

    return calibratedUtility_beta, calibratedUtility_q0, cal_amenities
