# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:22:55 2020

@author: Charlotte Liotta
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import optimize
import math
import copy
import scipy.io
import pickle
import os
import math
from sklearn.linear_model import LinearRegression

from calibration.compute_income import *

def EstimateIncome(param, timeOutput, distanceOutput, monetaryCost, costTime, job_centers, average_income, income_distribution, list_lambda):
    #Solve for income per employment centers for different values of lambda

    print('Estimation of local incomes, and lambda parameter')

    annualToHourly = 1 / (8*20*12)
    bracketsTime = np.array([0, 15, 30, 60, 90, np.nanmax(np.nanmax(np.nanmax(timeOutput)))])
    bracketsDistance = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 200])

    timeCost = copy.deepcopy(costTime)
    timeCost[np.isnan(timeCost)] = 10 ** 2
    monetary_cost = monetaryCost * annualToHourly
    monetary_cost[np.isnan(monetary_cost)] = 10 ** 3 * annualToHourly
    transportTimes = timeOutput / 2
    transportDistances = distanceOutput[:, :, 0]

    modalSharesTot = np.zeros((5, len(list_lambda)))
    incomeCentersSave = np.zeros((len(job_centers[:,0]), 4, len(list_lambda)))
    timeDistribution = np.zeros((len(bracketsTime) - 1, len(list_lambda)))
    distanceDistribution = np.zeros((len(bracketsDistance) - 1, len(list_lambda)))

    for i in range(0, len(list_lambda)):

        param_lambda = list_lambda[i]
        
        print('Estimating for lambda = ', param_lambda)
        
        incomeCentersAll = -math.inf * np.ones((len(job_centers[:,0]), 4))
        modalSharesGroup = np.zeros((5, 4))
        timeDistributionGroup = np.zeros((len(bracketsTime) - 1, 4))
        distanceDistributionGroup = np.zeros((len(bracketsDistance) - 1, 4))

        for j in range(0, 4):
        
            #Household size varies with transport costs
            householdSize = param["household_size"][j]
            
            averageIncomeGroup = average_income[j] * annualToHourly
        
            print('incomes for group ', j)
        
            whichJobsCenters = job_centers[:, j] > 600
            popCenters = job_centers[whichJobsCenters, j]
            #popResidence = copy.deepcopy(households_per_income_class[j]) * sum(job_centers[whichJobsCenters, j]) / np.nansum(households_per_income_class[j])
            #popResidence = income_distribution[cape_town_limits, j] * sum(job_centers[whichJobsCenters, j]) / sum(income_distribution[cape_town_limits, j])
            popResidence = income_distribution[:, j] * sum(job_centers[whichJobsCenters, j]) / sum(income_distribution[:, j])
           
            funSolve = lambda incomeCentersTemp: fun0(incomeCentersTemp, averageIncomeGroup, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, param_lambda)
            #funSolve = lambda incomeCentersTemp: fun0(incomeCentersTemp, averageIncomeGroup, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:], param_lambda)

            maxIter = 200
            tolerance = 0.001
            if j == 0:
                factorConvergenge = 0.008
            elif j == 1:
                factorConvergenge = 0.005
            else:
                factorConvergenge = 0.0005
        
            iter = 0
            error = np.zeros((len(popCenters), maxIter))
            scoreIter = np.zeros(maxIter)
            errorMax = 1
        
            #Initializing the solver
            incomeCenters = np.zeros((sum(whichJobsCenters), maxIter))
            incomeCenters[:, 0] = averageIncomeGroup * (popCenters / np.nanmean(popCenters)) ** (0.1)
            error[:, 0] = funSolve(incomeCenters[:, 0])

        
            while ((iter <= maxIter - 1) & (errorMax > tolerance)):
            
                
                incomeCenters[:,iter] = incomeCenters[:, max(iter-1, 0)] + factorConvergenge * averageIncomeGroup * error[:, max(iter - 1,0)] / popCenters
                
                error[:,iter] = funSolve(incomeCenters[:,iter])
                errorMax = np.nanmax(np.abs(error[:, iter] / popCenters))
                scoreIter[iter] = np.nanmean(np.abs(error[:, iter] / popCenters))
                print(np.nanmean(np.abs(error[:, iter])))
                iter = iter + 1
            
            if (iter > maxIter):
                scoreBest = np.amin(scoreIter)
                bestSolution = np.argmin(scoreIter)
                incomeCenters[:, iter-1] = incomeCenters[:, bestSolution]
                print(' - max iteration reached - mean error', scoreBest)
            else:
                print(' - computed - max error', errorMax)
        
        
            incomeCentersRescaled = incomeCenters[:, iter-1] * averageIncomeGroup / ((np.nansum(incomeCenters[:, iter-1] * popCenters) / np.nansum(popCenters)))
            #modalSharesGroup[:,j] = modalShares(incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, param_lambda)
            incomeCentersAll[whichJobsCenters,j] = incomeCentersRescaled
        
            #timeDistributionGroup[:,j] = computeDistributionCommutingTimes(incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, transportTimes[whichJobsCenters,:], bracketsTime, param_lambda)
            distanceDistributionGroup[:,j] = computeDistributionCommutingDistances(incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, transportDistances[whichJobsCenters,:], bracketsDistance, param_lambda)

        #modalSharesTot[:,i] = np.nansum(modalSharesGroup, 1) / np.nansum(np.nansum(modalSharesGroup))
        incomeCentersSave[:,:,i] = incomeCentersAll / annualToHourly
        #timeDistribution[:,i] = np.nansum(timeDistributionGroup, 1) / np.nansum(np.nansum(timeDistributionGroup))
        distanceDistribution[:,i] = np.nansum(distanceDistributionGroup, 1) / np.nansum(np.nansum(distanceDistributionGroup))

    return incomeCentersSave, distanceDistribution

def fun0(incomeCentersTemp, averageIncomeGroup, popCenters, popResidence, monetaryCost, timeCost, param_lambda):
    """ Computes error in employment allocation """

    #Redress the average income
    incomeCentersFull = incomeCentersTemp * averageIncomeGroup / ((np.nansum(incomeCentersTemp * popCenters) / np.nansum(popCenters)))

    #Transport costs and employment allocation
    transportCostModes = monetaryCost + timeCost * incomeCentersFull[:, None, None] #Eq 1

    #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    #Transport costs
    transportCost = - 1 / param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax) #Eq 2. Somme sur m de 1 à 5.

    #minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCentersFull[:, None] - transportCost))) - 500

    #Differences in the number of jobs
    #score = popCenters - np.nansum(np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome)) * popResidence, 1)
    proba = np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome), 0) #somme sur c
    #proba = np.exp((incomeCentersFull[:, None] - transportCost) - minIncome) / np.nansum(np.exp((incomeCentersFull[:, None] - transportCost) - minIncome), 0) #v2 sans le lambda   
    score = popCenters - np.nansum(popResidence[None, :] * proba, 1)
    return score


def modalShares(incomeCenters, popCenters, popResidence, monetaryCost, timeCost, param_lambda):
    """ Computes total modal shares """

    #Transport cost by modes
    transportCostModes = monetaryCost + timeCost * incomeCenters[:, None, None]

    #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    #Compute modal shares
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(np.exp(- param_lambda  * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]

    #Multiply by OD flows
    transportCost = - 1 / param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    #minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCenters[:, None] - transportCost))) - 500

    #Total modal shares
    modalSharesTot = np.nansum(np.nansum(modalSharesTemp * (np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)))[:, :, None] * popResidence, 1), 0)
    #modalSharesTot = np.tranpose(modalSharesTot, (2,0,1))

    return modalSharesTot


def computeDistributionCommutingTimes(incomeCenters, popCenters, popResidence, monetaryCost, timeCost, transportTime, bracketsTime, param_lambda):
    #incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, transportDistances[whichJobsCenters,:], bracketsDistance, param_lambda
    
    #Transport cost by modes
    transportCostModes = monetaryCost + timeCost * incomeCenters[:, None, None]

    #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 600

    #Compute modal shares
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]

    #Multiply by OD flows
    transportCost = - 1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    #minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCenters[:, None] - transportCost))) - 600

    #Total distribution of times
    nbCommuters = np.zeros(len(bracketsTime) - 1)
    for k in range(0, len(bracketsTime)-1):
        which = (transportTime > bracketsTime[k]) & (transportTime <= bracketsTime[k + 1]) & (~np.isnan(transportTime))
        nbCommuters[k] = np.nansum(np.nansum(np.nansum(which * modalSharesTemp * np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)[:, :, None] / np.nansum(np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)) * popResidence, 1)))

    return nbCommuters


def computeDistributionCommutingDistances(incomeCenters, popCenters, popResidence, monetaryCost, timeCost, transportDistance, bracketsDistance, param_lambda):
#incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, transportDistances[whichJobsCenters,:], bracketsDistance, param_lambda
    
    #Transport cost by modes
    transportCostModes = monetaryCost + timeCost * incomeCenters[:, None, None]

    #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    #Compute modal shares
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]

    #Multiply by OD flows
    transportCost = - 1/param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    #minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCenters[:, None] - transportCost))) - 500

    #Total distribution of times
    nbCommuters = np.zeros(len(bracketsDistance) - 1)
    for k in range(0, len(bracketsDistance)-1):
        which = (transportDistance > bracketsDistance[k]) & (transportDistance <= bracketsDistance[k + 1]) & (~np.isnan(transportDistance))
        nbCommuters[k] = np.nansum(np.nansum(np.nansum(which[:, :, None] * modalSharesTemp * np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)[:, :, None] / np.nansum(np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome), 0)[None, :, None] * popResidence[None, :, None], 1)))

    return nbCommuters
