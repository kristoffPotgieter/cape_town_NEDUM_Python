# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:50:37 2020.

@author: Charlotte Liotta
"""

import numpy as np
import math

import calibration.sub.loglikelihood as callog


def EstimateParametersByScanning(incomeNetOfCommuting, dataRent,
                                 dataDwellingSize, dataIncomeGroup,
                                 dataHouseholdDensity, selectedDensity,
                                 xData, yData, selectedSP, tableAmenities,
                                 variablesRegression, initRho, listBeta,
                                 listBasicQ, initUti2, listUti3, listUti4):
    """Estimate parameters by maximizing log likelihood."""
    # Here we scan a set of values for each parameter and determine the value
    # of the log-likelihood (to see how the model behaves).
    # NB: In estimate_parameters_by_optimization, we use the minimization
    # algorithm from Scipy to converge towards the solution

    # We remove poorest income group as it is crowded out of formal sector
    # TODO: is it in line with the paper?
    net_income = incomeNetOfCommuting[1:4, :]
    # We generate a matrix of dummies for dominant income group in each SP
    # (can be always false when dominant group is poorest)
    groupLivingSpMatrix = (net_income > 0)
    for i in range(0, 3):
        groupLivingSpMatrix[i, dataIncomeGroup != i] = np.zeros(1, 'bool')

    # We generate an array of dummies for dominant being not poorest
    selectedTransportMatrix = (np.nansum(groupLivingSpMatrix, 0) == 1)
    net_income[net_income < 0] = np.nan

    # We define a set of selection arrays
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

    # %% Useful functions (precalculations for rents and dwelling sizes,
    # likelihood function)

    # Function for dwelling sizes
    # See equation 9
    # TODO: correct typo in equation C3
    CalculateDwellingSize = (
        lambda beta, basic_q, incomeTemp, rentTemp:
            beta * incomeTemp / rentTemp + (1 - beta) * basic_q
            )

    # Log likelihood for a lognormal law of mean 0
    # TODO: correct typo in paper
    ComputeLogLikelihood = (
        lambda sigma, error:
            np.nansum(- np.log(2 * math.pi * sigma ** 2) / 2
                      - 1 / (2 * sigma ** 2) * (error) ** 2)
            )

    # %% Optimization algorithm

    # Function that will be minimized
    optionRegression = 0

    # Initial value of parameters (all possible combinations)
    # Note that we do not consider rho here
    combinationInputs = np.array(
        np.meshgrid(listBeta, listBasicQ, listUti3, listUti4)).T.reshape(-1, 4)

    # Scanning of the list

    scoreAmenities = - 10000 * np.ones(combinationInputs.shape[0])
    scoreDwellingSize = - 10000 * np.ones(combinationInputs.shape[0])
    scoreIncomeSorting = - 10000 * np.ones(combinationInputs.shape[0])
    scoreHousing = - 10000 * np.ones(combinationInputs.shape[0])
    scoreTotal = - 10000 * np.ones(combinationInputs.shape[0])

    print('\nDone: ')

    # TODO: how strong are underlying Gumbel assumptions?
    for index in range(0, combinationInputs.shape[0]):
        print(index)
        (scoreTotal[index], scoreAmenities[index], scoreDwellingSize[index],
         scoreIncomeSorting[index], scoreHousing[index], parametersAmenities,
         modelAmenities, parametersHousing) = callog.LogLikelihoodModel(
             combinationInputs[index, :], initUti2, net_income,
             groupLivingSpMatrix, dataDwellingSize, selectedDwellingSize,
             dataRent, selectedRents, selectedDensity,
             predictorsAmenitiesMatrix, tableRegression, variablesRegression,
             CalculateDwellingSize, ComputeLogLikelihood, optionRegression)

    print('\nScanning complete')
    print('\n')

    # We just pick the parameters associated to the maximum score
    scoreVect = (scoreAmenities + scoreDwellingSize + scoreIncomeSorting
                 + scoreHousing)
    scoreTot = np.amax(scoreVect)
    which = np.argmax(scoreVect)
    parameters = combinationInputs[which, :]

    # We just re-estimate the function to get the (right) parameters for
    # amenities: we did not do it before as optionRegression = 0 is better for
    # the rest. Note that paramaters remain the same, but model is a GLM and
    # errors are Pearson residuals
    optionRegression = 1
    (*_, parametersAmenities, modelAmenities, parametersHousing
     ) = callog.LogLikelihoodModel(
         parameters, initUti2, net_income, groupLivingSpMatrix,
         dataDwellingSize, selectedDwellingSize, dataRent,
         selectedRents, selectedDensity,
         predictorsAmenitiesMatrix, tableRegression, variablesRegression,
         CalculateDwellingSize, ComputeLogLikelihood, optionRegression)

    return (parameters, scoreTot, parametersAmenities, modelAmenities,
            parametersHousing, selectedRents)
