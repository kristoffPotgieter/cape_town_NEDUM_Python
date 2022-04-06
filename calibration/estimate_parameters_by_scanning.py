# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:50:37 2020

@author: Charlotte Liotta
"""

import pandas as pd
import numpy as np
import math

from calibration.estimate_parameters_by_scanning import *
from calibration.loglikelihood import *


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
    net_income = incomeNetOfCommuting[1:4,:]
    groupLivingSpMatrix = (net_income > 0)
    for i in range(0, 3):
        groupLivingSpMatrix[i, data_income_group != i] = np.zeros(1, 'bool')

    selectedTransportMatrix = (np.nansum(groupLivingSpMatrix, 0) == 1)
    net_income[net_income < 0] = np.nan

    selectedRents = ~np.isnan(dataRent) & selectedTransportMatrix & selectedSP
    selectedDwellingSize = ~np.isnan(data_sp["dwelling_size"]) & ~np.isnan(dataRent) & selectedTransportMatrix & selectedSP
    selectedDensity = selectedDwellingSize & selected_density

    #For the regression of amenities
    tableRegression = amenities_sp.loc[selectedRents, :]
    predictorsAmenitiesMatrix = tableRegression.loc[:, variables_regression]
    predictorsAmenitiesMatrix = np.vstack([np.ones(predictorsAmenitiesMatrix.shape[0]), predictorsAmenitiesMatrix.T]).T
    modelAmenity = 0

    # %% Useful functions (precalculations for rents and dwelling sizes, likelihood function) 

    #Function for dwelling sizes
    #We estimate calcule_hous directly from data from rents (no extrapolation)
    CalculateDwellingSize = lambda beta, basic_q, incomeTemp, rentTemp : beta * incomeTemp / rentTemp + (1 - beta) * basic_q

    #Log likelihood for a lognormal law
    ComputeLogLikelihood = lambda sigma, error : np.nansum(- np.log(2 * math.pi * sigma ** 2) / 2 -  1 / (2 * sigma ** 2) * (error) ** 2)
    

    # %% Optimization algorithm

    #Function that will be minimized 
    optionRegression = 0

    #Initial value of parameters
    #combinationInputs = combvec(listBeta, listBasicQ, listUti3, listUti4) #So far, no spatial autocorrelation
    combinationInputs = np.array(np.meshgrid(listBeta, listBasicQ, listUti3, listUti4)).T.reshape(-1,4)
    
    #Scanning of the list
    scoreAmenities = - 10000 * np.ones(combinationInputs.shape[0])
    scoreDwellingSize = - 10000 * np.ones(combinationInputs.shape[0])
    scoreIncomeSorting = - 10000 * np.ones(combinationInputs.shape[0])
    scoreHousing = - 10000 * np.ones(combinationInputs.shape[0])
    scoreTotal = - 10000 * np.ones(combinationInputs.shape[0])
    iterPrint = np.floor(np.ones(combinationInputs.shape[0]) / 20)
    print('\nDone: ')
    for index in range(0, combinationInputs.shape[0]):
        print(index)
        #scoreTotal[index], scoreAmenities[index], scoreDwellingSize[index], scoreIncomeSorting[index], scoreHousing[index], parametersAmenities, modelAmenities, parametersHousing = LogLikelihoodModel(combinationInputs[:, index], Uo2, net_income, groupLivingSpMatrix, data_sp, selectedDwellingSize, dataRent, selectedRents, selectedDensity, predictorsAmenitiesMatrix, tableRegression, variables_regression, CalculateDwellingSize, ComputeLogLikelihood, optionRegression)
        scoreTotal[index], scoreAmenities[index], scoreDwellingSize[index], scoreIncomeSorting[index], scoreHousing[index], parametersAmenities, modelAmenities, parametersHousing = LogLikelihoodModel(combinationInputs[index, :], Uo2, net_income, groupLivingSpMatrix, data_sp, selectedDwellingSize, dataRent, selectedRents, selectedDensity, predictorsAmenitiesMatrix, tableRegression, variables_regression, CalculateDwellingSize, ComputeLogLikelihood, optionRegression)
        #if (floor(index / iterPrint) == index/iterPrint):
         #   fprintf('%0.f%%  ', round(index / size(combinationInputs,2) .* 100));

    print('\nScanning complete')
    print('\n')

    scoreVect = scoreAmenities + scoreDwellingSize + scoreIncomeSorting + scoreHousing
    scoreTot = np.amax(scoreVect)
    which = np.argmax(scoreVect)
    parameters = combinationInputs[which, :]

    #Estimate the function to get the parameters for amenities
    optionRegression = 1
    [_, parametersAmenities, modelAmenity, parametersHousing] = LogLikelihoodModel(parameters, initUti2, incomeNetOfCommuting, groupLivingSpMatrix, dataDwellingSize, selectedDwellingSize, xData, yData, dataRent, selectedRents, dataHouseholdDensity, selectedDensity, predictorsAmenitiesMatrix, tableRegression, variablesRegression, CalculateDwellingSize, ComputeLogLikelihood, optionRegression);
    
    return parameters, scoreTot, parametersAmenities, modelAmenity, parametersHousing, selectedRents


def confidence_interval(indices_max, quoi_indices, compute_score):

    d_beta = 1.05

    print('\n')
    beta_interval = np.zeros(size(indices_max))
    for index in range (0, size(indices_max,1)):
        indices_ici = indices_max
        score_tmp = compute_score(indices_ici)
    
        indices_ici = indices_max
        score_tmp2 = compute_score(indices_ici)
    
        indices_ici = indices_max
        indices_ici[index] = indices_ici[index] - indices_ici[index] * (d_beta - 1)
        score_tmp3 = compute_score(indices_ici)
    
        indices_ici = indices_max
        dd_l_beta = -(score_tmp2 + score_tmp3 - 2 * score_tmp) / (indices_ici[index] * (d_beta - 1)) ** 2
        beta_interval[index] = 1.96 / (np.sqrt(np.abs(dd_l_beta)))
        # fprintf('%s\t\t%g (%g ; %g)\n', quoi_indices{index}, indices_ici[index], indices_ici[index] - beta_interval[index], indices_ici[index] + beta_interval[index])

    return np.array([indices_ici - beta_interval, indices_ici + beta_interval])


