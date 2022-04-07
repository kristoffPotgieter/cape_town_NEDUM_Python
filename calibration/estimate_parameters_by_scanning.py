# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:50:37 2020.

@author: Charlotte Liotta
"""

import numpy as np
import math

import calibration.loglikelihood as callog


def EstimateParametersByScanning(incomeNetOfCommuting, dataRent,
                                 dataDwellingSize, dataIncomeGroup,
                                 dataHouseholdDensity, selectedDensity,
                                 xData, yData, selectedSP, tableAmenities,
                                 variablesRegression, initRho, listBeta,
                                 listBasicQ, initUti2, listUti3, listUti4):
    """Estimate parameters by maximizing log likelihood."""
    # Here we scan a set of values for each parameter and determine the value
    # of the log-likelihood (to see how the model behaves).
    # NB: In EstimateParameters By Optimization, we use the minimization
    # algorithm from Matlab to converge towards the solution

    # Data as matrices, where should we regress (remove where we have no data)
    # Where is which class

    # We remove poorest income group
    net_income = incomeNetOfCommuting[1:4, :]
    groupLivingSpMatrix = (net_income > 0)
    for i in range(0, 3):
        groupLivingSpMatrix[i, dataIncomeGroup != i] = np.zeros(1, 'bool')

    selectedTransportMatrix = (np.nansum(groupLivingSpMatrix, 0) == 1)
    net_income[net_income < 0] = np.nan

    selectedRents = ~np.isnan(dataRent) & selectedTransportMatrix & selectedSP
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
    modelAmenity = 0

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

    # Function that will be minimized
    optionRegression = 0

    # Initial value of parameters
    combinationInputs = np.array(
        np.meshgrid(listBeta, listBasicQ, listUti3, listUti4)).T.reshape(-1, 4)

    # Scanning of the list
    scoreAmenities = - 10000 * np.ones(combinationInputs.shape[0])
    scoreDwellingSize = - 10000 * np.ones(combinationInputs.shape[0])
    scoreIncomeSorting = - 10000 * np.ones(combinationInputs.shape[0])
    scoreHousing = - 10000 * np.ones(combinationInputs.shape[0])
    scoreTotal = - 10000 * np.ones(combinationInputs.shape[0])

    print('\nDone: ')

    Uo2 = 1000

    for index in range(0, combinationInputs.shape[0]):
        print(index)
        (scoreTotal[index], scoreAmenities[index], scoreDwellingSize[index],
         scoreIncomeSorting[index], scoreHousing[index], parametersAmenities,
         modelAmenities, parametersHousing) = callog.LogLikelihoodModel(
             combinationInputs[index, :], Uo2, net_income, groupLivingSpMatrix,
             dataDwellingSize, selectedDwellingSize, dataRent, selectedRents,
             selectedDensity, predictorsAmenitiesMatrix, tableRegression,
             variablesRegression, CalculateDwellingSize, ComputeLogLikelihood,
             optionRegression)

    print('\nScanning complete')
    print('\n')

    scoreVect = (scoreAmenities + scoreDwellingSize + scoreIncomeSorting
                 + scoreHousing)
    scoreTot = np.amax(scoreVect)
    which = np.argmax(scoreVect)
    parameters = combinationInputs[which, :]

    # Estimate the function to get the parameters for amenities
    optionRegression = 1
    (_, parametersAmenities, modelAmenity, parametersHousing
     ) = callog.LogLikelihoodModel(
         parameters, initUti2, incomeNetOfCommuting, groupLivingSpMatrix,
         dataDwellingSize, selectedDwellingSize, xData, yData, dataRent,
         selectedRents, dataHouseholdDensity, selectedDensity,
         predictorsAmenitiesMatrix, tableRegression, variablesRegression,
         CalculateDwellingSize, ComputeLogLikelihood, optionRegression)

    return (parameters, scoreTot, parametersAmenities, modelAmenity,
            parametersHousing, selectedRents)
