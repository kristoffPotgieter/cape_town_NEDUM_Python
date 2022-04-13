# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:49:58 2020.

@author: Charlotte Liotta
"""

import numpy as np
import math
import scipy

import calibration.loglikelihood as callog


def EstimateParametersByOptimization(
        incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup,
        dataHouseholdDensity, selectedDensity, xData, yData, selectedSP,
        tableAmenities, variablesRegression, initRho, initBeta, initBasicQ,
        initUti2, initUti3, initUti4):
    """Automatically estimate parameters by maximizing log likelihood."""
    # Here we minimize the log-likelihood using fminsearch

    # Data as matrices, where should we regress (remove where we have no data)

    # Where is which class
    net_income = incomeNetOfCommuting[1:4, :]
    groupLivingSpMatrix = (net_income > 0)
    for i in range(0, 3):
        groupLivingSpMatrix[i, dataIncomeGroup != i] = np.zeros(1, 'bool')

    selectedTransportMatrix = (sum(groupLivingSpMatrix) == 1)
    net_income[net_income < 0] = np.nan

    selectedRents = (~np.isnan(dataRent)
                     & selectedTransportMatrix
                     & selectedSP)
    selectedDwellingSize = (~np.isnan(dataDwellingSize)
                            & ~np.isnan(dataRent)
                            & selectedTransportMatrix
                            & selectedSP)
    selectedDensity = selectedDwellingSize & selectedDensity

    # For the regression of amenities
    tableRegression = tableAmenities.loc[selectedRents, :]
    predictorsAmenitiesMatrix = tableRegression.loc[:, variablesRegression]
    # TODO: should be added?
    predictorsAmenitiesMatrix = np.vstack(
        [np.ones(predictorsAmenitiesMatrix.shape[0]),
          predictorsAmenitiesMatrix.T]
        ).T
    # modelAmenity = 0

    # %% Useful functions (precalculations for rents and dwelling sizes,
    # likelihood function)

    # Function for dwelling sizes
    # We estimate calcule_hous directly from data from rents (no extrapolation)
    CalculateDwellingSize = (
        lambda beta, basic_q, incomeTemp, rentTemp:
            beta * incomeTemp / rentTemp + (1 - beta) * basic_q
            )

    # Log likelihood for a lognormal law
    ComputeLogLikelihood = (
        lambda sigma, error:
            np.nansum(- np.log(2 * math.pi * sigma ** 2) / 2
                      - 1 / (2 * sigma ** 2) * (error) ** 2)
            )

    # %% Optimization algorithm

    # Initial value of parameters
    # TODO: where does it come from? Why not use init values?
    initialVector = np.array(
        [0.25332341, 3.97137219, 18683.85807256, 86857.19233169])

    Uo2 = 1000

    # Function that will be minimized
    optionRegression = 0

    minusLogLikelihoodModel = (
        lambda X0:
            - callog.LogLikelihoodModel(
                X0, Uo2, net_income, groupLivingSpMatrix, dataDwellingSize,
                selectedDwellingSize, dataRent, selectedRents, selectedDensity,
                predictorsAmenitiesMatrix, tableRegression,
                variablesRegression, CalculateDwellingSize,
                ComputeLogLikelihood, optionRegression)[0]
            )

    # Optimization w/ lower and upper bounds
    # lowerBounds = np.array([0.1, 3, 0, 0])
    # upperBounds = np.array([1, 18, 10 ** 6, 10 ** 7])
    bnds = ((0.1, 1), (3, 18), (0, 18 ** 6), (0, 10 ** 7))

    # (parameters, scoreTot, exitFlag) = scipy.optimize.minimize(
    #     minusLogLikelihoodModel, initialVector)

    res = scipy.optimize.minimize(
        minusLogLikelihoodModel, initialVector, bounds=bnds,
        options={'maxiter': 10, 'disp': True})

    parameters = res.x
    scoreTot = res.fun
    # exitFlag = res.success

    # Estimate the function to get the parameters for amenities
    optionRegression = 1
    (*_, parametersAmenities, modelAmenity, parametersHousing
     ) = callog.LogLikelihoodModel(
         parameters, initUti2, net_income, groupLivingSpMatrix,
         dataDwellingSize, selectedDwellingSize, dataRent,
         selectedRents, selectedDensity,
         predictorsAmenitiesMatrix, tableRegression, variablesRegression,
         CalculateDwellingSize, ComputeLogLikelihood, optionRegression)

    print('*** Estimation of beta and q0 done ***')

    return (parameters, scoreTot, parametersAmenities, modelAmenity,
            parametersHousing, selectedRents)
