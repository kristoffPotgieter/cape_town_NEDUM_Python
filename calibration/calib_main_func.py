# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:26:37 2022.

@author: monni
"""

# import pandas as pd
import numpy as np
import numpy.matlib
# import scipy.io
# from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os
import math
import scipy.io

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
        np.array([np.ones(len(data_sp["price"][selected_density])),
                  np.log(data_sp["price"][selected_density]),
                  np.log(data_sp["dwelling_size"][selected_density]),
                  np.log(param["max_land_use"]
                         * data_sp["unconstrained_area"][selected_density])])
        )
    # NB: Our data set for dwelling sizes only provides the average (not
    # median) dwelling size at the Sub-Place level, aggregating formal and
    # informal housing

    # model_construction = LinearRegression().fit(X, y)

    modelSpecification = sm.OLS(y, X, missing='drop')
    model_construction = modelSpecification.fit()
    print(model_construction.summary())
    parametersConstruction = model_construction.params

    # We export outputs of the model
    # coeff_b = model_construction.coef_[0]
    coeff_b = parametersConstruction["x1"]
    coeff_a = 1 - coeff_b
    # Comes from zero profit condition combined with footnote 16 from
    # optimization
    # TODO: correct typo in paper
    if options["correct_kappa"] == 1:
        # coeffKappa = ((1 / (coeff_b / coeff_a) ** coeff_b)
        #               * np.exp(model_construction.intercept_))
        coeffKappa = ((1 / (coeff_b / coeff_a) ** coeff_b)
                      * np.exp(parametersConstruction["const"]))
    elif options["correct_kappa"] == 0:
        # coeffKappa = ((1 / (coeff_b) ** coeff_b)
        #               * np.exp(model_construction.intercept_))
        coeffKappa = ((1 / (coeff_b) ** coeff_b)
                      * np.exp(parametersConstruction["const"]))

    try:
        os.mkdir(path_precalc_inp)
    except OSError as error:
        print(error)

    np.save(path_precalc_inp + 'calibratedHousing_b.npy', coeff_b)
    np.save(path_precalc_inp + 'calibratedHousing_kappa.npy', coeffKappa)

    # We add the option in case we want to reverse estimated elasticties to
    # stick closer to the literature
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

    (incomeCenters, distanceDistribution, scoreMatrix
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

    # Note that income is set to -inf for job centers and income groups in
    # which it could not be calibrated

    np.save(path_precalc_inp + 'incomeCentersKeep.npy', incomeCentersKeep)
    np.save(path_precalc_inp + 'lambdaKeep.npy', lambdaKeep)

    # Validate with initial input from Matlab
    # incomeCentersKeep_mat = scipy.io.loadmat(
    #     path_precalc_inp + 'incomeCentersKeep.mat')['incomeCentersKeep']

    # Note that it is unclear whether "average" income from data includes
    # unemployment or not: a priori, it does for short spells (less than one
    # year) and should therefore be slightly bigger than calibrated income:
    # this is what we observe in practice
    incomeCentersKeep[incomeCentersKeep < 0] = math.nan
    cal_avg_income = np.nanmean(incomeCentersKeep, 0)
    # incomeCentersKeep_mat[incomeCentersKeep_mat < 0] = math.nan
    # cal_avg_income_mat = np.nanmean(incomeCentersKeep_mat, 0)

    return (incomeCentersKeep, lambdaKeep, cal_avg_income, scoreKeep,
            bhattacharyyaDistances)


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
    selectedSP = (
        (data_number_formal > 0.90 * housing_types_sp.total_dwellings_SP_2011)
        & (data_income_group > 0)
        )

    # Coefficients of the model for simulations: acceptable range
    # listBeta = np.arange(0.43, 0.451, 0.01)
    # listBeta = np.arange(0.44, 0.441, 0.01)
    # listBeta = np.arange(0.1, 0.51, 0.1)
    listBeta = np.arange(0.27, 0.321, 0.01)

    # listBasicQ = np.arange(2, 14.1, 1)
    listBasicQ = np.arange(12.7, 13.21, 0.1)

    # Coefficient for spatial autocorrelation
    listRho = 0

    # Utilities for simulations: we start with levels close from what we expect
    # in equilibrium
    # TODO: to be updated with values close to what we obtain in equilibrium
    # (to speed up convergence)
    utilityTarget = np.array([1500, 5000, 17000, 80000])
    # utilityTarget = np.array([300, 1000, 3000, 10000])
    # utilityTarget = np.array([3000, 10000, 30000, 100000])

    # We scan varying values of utility targets
    listVariation = np.arange(0.5, 2.01, 0.25)
    # listVariation1 = np.arange(1, 1.01, 0.1)
    # listVariation2 = np.arange(0.5, 0.51, 0.1)

    # Note that the poorest income group is not taken into account as it is
    # excluded from the analysis.
    # Then, we do not vary the first income group as the magnitude is not that
    # large...
    initUti2 = utilityTarget[1]
    # However, we do so for the two richest groups.
    # NB: having opposite variations is key to maintain stability of beta and
    # q0 estimates
    # TODO: is it robust?
    listUti3 = utilityTarget[2] * listVariation
    listUti4 = utilityTarget[3] * listVariation

    # Cf. inversion of footnote 16
    if options["correct_kappa"] == 1 & options["deprec_land"] == 1:
        dataRent = (
            data_sp["price"] ** (coeff_a)
            * (param["depreciation_rate"]
               + interest_rate)
            / (coeffKappa * coeff_b ** coeff_b * coeff_a ** coeff_a)
            )
    elif options["correct_kappa"] == 1 & options["deprec_land"] == 0:
        dataRent = (
            (interest_rate * data_sp["price"]) ** coeff_a
            * (param["depreciation_rate"]
               + interest_rate) ** coeff_b
            / (coeffKappa * coeff_b ** coeff_b * coeff_a ** coeff_a)
            )
    elif options["correct_kappa"] == 0 & options["deprec_land"] == 1:
        dataRent = (
            data_sp["price"] ** (coeff_a)
            * (param["depreciation_rate"]
               + interest_rate)
            / (coeffKappa * coeff_b ** coeff_b)
            )
    elif options["correct_kappa"] == 0 & options["deprec_land"] == 0:
        dataRent = (
            (data_sp["price"] * interest_rate) ** coeff_a
            * (param["depreciation_rate"]
               + interest_rate) ** coeff_b
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
    # NB: RBFInterpolator does not work well but interp2d does not have
    # a regular behaviour...
    (parametersScan, scoreScan, parametersAmenitiesScan, modelAmenityScan,
     parametersHousing, _) = calscan.EstimateParametersByScanning(
         incomeNetOfCommuting, dataRent, data_sp["dwelling_size"],
         data_income_group, data_density, selected_density,
         housing_types_sp["x_sp"], housing_types_sp["y_sp"], selectedSP,
         amenities_sp, variables_regression, listRho, listBeta, listBasicQ,
         initUti2, listUti3, listUti4, options)

    # Coefficients appear to be stable as long as we allow utilities to vary
    # widly... Where should we set the limit?

    print(modelAmenityScan.summary())

    # Now we run the optimization algo with identified value of the parameters:
    # corresponds to interior-point algorithm

    # Note that this may be long to run
    # TODO: Should we allow more iterations?
    if options["param_optim"] == 1:
        options["griddata"] = 1
        options["log_form"] = 1
        # Gets pretty slow above 100
        options["interpol_neighbors"] = 100
        # Cannot rely on solver? Stuck on the wrong side of q0?

        # initBeta = parametersScan[0]
        # initBasicQ = parametersScan[1]
        # This allows to be on the good side of the solver?
        # TODO: how to justify?
        # initBasicQ = max(parametersScan[1], 5.1)
        # initUti3 = parametersScan[2]
        # initUti4 = parametersScan[3]

        initBeta = scipy.io.loadmat(
            path_precalc_inp + 'calibratedUtility_beta.mat'
            )["calibratedUtility_beta"].squeeze()
        initBasicQ = scipy.io.loadmat(
            path_precalc_inp + 'calibratedUtility_q0.mat'
            )["calibratedUtility_q0"].squeeze()
        utils = scipy.io.loadmat(
            path_precalc_inp + 'calibratedUtilities.mat'
            )["utilitiesCorrected"].squeeze()
        initUti3 = utils[0]
        initUti4 = utils[1]

        (parameters, scoreTot, parametersAmenities, modelAmenity,
         parametersHousing, selectedSPRent
         ) = calopt.EstimateParametersByOptimization(
             incomeNetOfCommuting, dataRent, data_sp["dwelling_size"],
             data_income_group, data_density, selected_density,
             housing_types_sp["x_sp"], housing_types_sp["y_sp"], selectedSP,
             amenities_sp, variables_regression, listRho, initBeta, initBasicQ,
             initUti2, initUti3, initUti4, options)

        print(modelAmenity.summary())

    # Exporting and saving outputs

    amenities_grid = calam.import_amenities(path_data, path_precalc_inp,
                                            'grid')
    predictors_grid = amenities_grid.loc[:, variables_regression]
    predictors_grid = np.vstack(
        [np.ones(predictors_grid.shape[0]),
         predictors_grid.T]
        ).T

    if options["param_optim"] == 1:
        cal_amenities = np.exp(
            np.nansum(predictors_grid * parametersAmenities, 1))
    elif options["param_optim"] == 0:
        cal_amenities = np.exp(
            np.nansum(predictors_grid * parametersAmenitiesScan, 1))
    calw_amenities = cal_amenities / np.nanmean(cal_amenities)

    try:
        os.mkdir(path_plots)
    except OSError as error:
        print(error)
    outexp.export_map(calw_amenities, grid, path_plots + 'amenity_map',
                      1.3, 0.8)

    # Note that amenity map from GLM estimates yields absurd results, hence it
    # is not coded here
    if options["glm"] == 1:
        modelAmenity.save(path_precalc_inp + 'modelAmenity')

    if options["param_optim"] == 1:
        calibratedUtility_beta = parameters[0]
        calibratedUtility_q0 = parameters[1]
    elif options["param_optim"] == 0:
        calibratedUtility_beta = parametersScan[0]
        calibratedUtility_q0 = parametersScan[1]

    np.save(path_precalc_inp + 'calibratedUtility_beta',
            calibratedUtility_beta)
    np.save(path_precalc_inp + 'calibratedUtility_q0', calibratedUtility_q0)
    np.save(path_precalc_inp + 'calibratedAmenities', cal_amenities)

    return (calibratedUtility_beta, calibratedUtility_q0, cal_amenities)
