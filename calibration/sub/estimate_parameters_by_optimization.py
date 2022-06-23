# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:49:58 2020.

@author: Charlotte Liotta
"""

import numpy as np
import math
import scipy
from numba import jit

import calibration.sub.loglikelihood as callog


# @jit
def EstimateParametersByOptimization(
        incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup,
        dataHouseholdDensity, selectedDensity, xData, yData, selectedSP,
        tableAmenities, variablesRegression, initRho, initBeta, initBasicQ,
        initUti2, initUti3, initUti4, options):
    """Automatically estimate parameters by maximizing log likelihood."""
    # We start as in EstimateParametersByScanning
    net_income = incomeNetOfCommuting[1:4, :]
    groupLivingSpMatrix = (net_income > 0)
    for i in range(0, 3):
        groupLivingSpMatrix[i, dataIncomeGroup != i+1] = np.zeros(1, 'bool')

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
    predictorsAmenitiesMatrix = np.vstack(
        [np.ones(predictorsAmenitiesMatrix.shape[0]),
         predictorsAmenitiesMatrix.T]
        ).T

    # %% Useful functions (precalculations for rents and dwelling sizes,
    # likelihood function)

    # Function for dwelling sizes
    # We estimate calcule_hous directly from data from rents (no extrapolation)
    np.seterr(divide='ignore', invalid='ignore')
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
    initialVector = np.array(
        [initBeta, initBasicQ, initUti3, initUti4])

    # Determines function that will be minimized
    optionRegression = 0

    minusLogLikelihoodModel = (
        lambda X0:
            - callog.LogLikelihoodModel(
                X0, initUti2, net_income, groupLivingSpMatrix,
                dataDwellingSize, selectedDwellingSize, dataRent,
                selectedRents, selectedDensity,
                predictorsAmenitiesMatrix, tableRegression,
                variablesRegression, CalculateDwellingSize,
                ComputeLogLikelihood, optionRegression, options)[0]
            )

    # Now, we optimize using interior-point minimization algorithms

    # We first define wide bounds for our parameters
    bnds = ((0.1, 1), (1, 20), (0, 10 ** 6), (0, 10 ** 7))

    # Nfeval = 1

    def callbackF(Xi):
        # global Nfeval
        # print(
        #     '{0:4d} {1:3.6f}'.format(Nfeval, minusLogLikelihoodModel(Xi))
        #     )
        # Nfeval += 1
        print(
            '{0:3.6f}'.format(minusLogLikelihoodModel(Xi))
            )

    # print('{0:4s} {1:9s}'.format('Iter', 'f(X)'))

    # Then we run the algorithm
    # TODO: play with algorigthm used?
    res = scipy.optimize.minimize(
        minusLogLikelihoodModel, initialVector, bounds=bnds,
        options={'maxiter': 10, 'disp': True},
        callback=callbackF)

    print(res)
    parameters = res.x
    scoreTot = res.fun
    # exitFlag = res.success

    # Estimate the function to get the parameters for amenities
    if options["glm"] == 1:
        optionRegression = 1
        (*_, parametersAmenities, modelAmenity, parametersHousing
         ) = callog.LogLikelihoodModel(
             parameters, initUti2, net_income, groupLivingSpMatrix,
             dataDwellingSize, selectedDwellingSize, dataRent,
             selectedRents, selectedDensity,
             predictorsAmenitiesMatrix, tableRegression, variablesRegression,
             CalculateDwellingSize, ComputeLogLikelihood, optionRegression,
             options)
    elif options["glm"] == 0:
        optionRegression = 0
        (*_, parametersAmenities, modelAmenity, parametersHousing
         ) = callog.LogLikelihoodModel(
             parameters, initUti2, net_income, groupLivingSpMatrix,
             dataDwellingSize, selectedDwellingSize, dataRent,
             selectedRents, selectedDensity,
             predictorsAmenitiesMatrix, tableRegression, variablesRegression,
             CalculateDwellingSize, ComputeLogLikelihood, optionRegression,
             options)

    print('*** Estimation of beta and q0 done ***')

    return (parameters, scoreTot, parametersAmenities, modelAmenity,
            parametersHousing, selectedRents)
