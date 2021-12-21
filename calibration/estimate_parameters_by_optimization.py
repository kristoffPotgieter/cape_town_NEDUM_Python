# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:49:58 2020

@author: Charlotte Liotta
"""

import pandas as pd
import numpy as np
import math

from calibration.estimate_parameters_by_optimization import *
from calibration.loglikelihood import *

def EstimateParametersByOptimization(incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup, dataHouseholdDensity, selectedDensity, xData, yData, selectedSP, tableAmenities, variablesRegression, initRho, initBeta, initBasicQ, initUti2, initUti3, initUti4):
    
    """ Automated estimation of the parameters of NEDUM by maximizing log likelihood
        
    Here we minimize the log-likelihood using fminsearch
        """

    #Data as matrices, where should we regress (remove where we have no data)

    #Where is which class
    net_income = income_net_of_commuting_costs[1:4,:] #We remove income group 1
    groupLivingSpMatrix = (net_income > 0)
    for i in range(0, 3):
        groupLivingSpMatrix[i, data_income_group != i] = np.zeros(1, 'bool')
    
    selectedTransportMatrix = (sum(groupLivingSpMatrix) == 1)
    net_income[net_income < 0] = np.nan

    selectedRents = ~np.isnan(dataRent) & selectedTransportMatrix & selectedSP
    selectedDwellingSize = ~np.isnan(data_sp["dwelling_size"]) & ~np.isnan(dataRent) & selectedTransportMatrix & selectedSP
    selectedDensity = selectedDwellingSize & selected_density

    #For the regression of amenities
    tableRegression = amenities_sp.loc[selectedRents, :]
    predictorsAmenitiesMatrix = tableRegression.loc[:, variables_regression]
    #predictorsAmenitiesMatrix = [np.ones(size(predictorsAmenitiesMatrix,1),1), predictorsAmenitiesMatrix]
    modelAmenity = 0

    # %% Useful functions (precalculations for rents and dwelling sizes, likelihood function) 

    #Function for dwelling sizes
    #We estimate calcule_hous directly from data from rents (no extrapolation)
    CalculateDwellingSize = lambda beta, basic_q, incomeTemp, rentTemp : beta * incomeTemp / rentTemp + (1 - beta) * basic_q

    #Log likelihood for a lognormal law
    ComputeLogLikelihood = lambda sigma, error : np.nansum(- np.log(2 * math.pi * sigma ** 2) / 2 -  1 / (2 * sigma ** 2) * (error) ** 2)
    
    # %% Optimization algorithm

    #Initial value of parameters
    initialVector = np.array([0.25332341, 3.97137219, 18683.85807256, 86857.19233169])

    #Function that will be minimized 
    optionRegression = 0
    minusLogLikelihoodModel = lambda X0 : -(LogLikelihoodModel(X0, Uo2, net_income, groupLivingSpMatrix, data_sp, selectedDwellingSize, dataRent, selectedRents, selectedDensity, predictorsAmenitiesMatrix, tableRegression, variables_regression, CalculateDwellingSize, ComputeLogLikelihood, optionRegression))

    #Optimization w/ lower and upper bounds
    lowerBounds = np.array([0.1, 3, 0, 0])
    upperBounds = np.array([1, 18, 10 ** 6, 10 ** 7])
    #optionsOptim = optimset('Display', 'iter');
    [parameters, scoreTot, exitFlag] = scipy.optimize.minimize(minusLogLikelihoodModel, initialVector, lowerBounds, upperBounds)
    bnds = ((0.1, 1), (3, 18), (0, 18 ** 6), (0, 10** 7))
    res = minimize(minusLogLikelihoodModel, initialVector, bounds=bnds, options={'maxiter': 10, 'disp': True})
    #Estimate the function to get the parameters for amenities
    optionRegression = 1
    [~, ~, ~, ~, ~, parametersAmenities, modelAmenity, parametersHousing] = LogLikelihoodModel(parameters, initUti2, incomeNetOfCommuting, groupLivingSpMatrix, dataDwellingSize, selectedDwellingSize, xData, yData, dataRent, selectedRents, dataHouseholdDensity, selectedDensity, predictorsAmenitiesMatrix, tableRegression, variablesRegression, CalculateDwellingSize, ComputeLogLikelihood, optionRegression)

    disp('*** Estimation of beta and q0 done ***')

    return parameters, scoreTot, parametersAmenities, modelAmenity, parametersHousing, selectedRents

def confidence_interval(indices_max, quoi_indices, compute_score):

    d_beta=1.05;

    fprintf('\n');
    beta_interval = zeros(size(indices_max));
    for index = 1:size(indices_max,1),
        indices_ici = indices_max;
        score_tmp = compute_score(indices_ici);
        
        indices_ici = indices_max;
        score_tmp2 = compute_score(indices_ici);
        
        indices_ici = indices_max;
        indices_ici(index) = indices_ici(index) - indices_ici(index)*(d_beta-1);
        score_tmp3 = compute_score(indices_ici);
        
        indices_ici = indices_max;
        dd_l_beta = -(score_tmp2 + score_tmp3 - 2*score_tmp) / (indices_ici(index)*(d_beta-1))^2;
        beta_interval(index) = 1.96 / (sqrt( abs(dd_l_beta)));
        fprintf('%s\t\t%g (%g ; %g)\n',quoi_indices{index},indices_ici(index),indices_ici(index)-beta_interval(index),indices_ici(index)+beta_interval(index))

    return [indices_ici-beta_interval,indices_ici+beta_interval]



