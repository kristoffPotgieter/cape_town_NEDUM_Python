# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:22:55 2020.

@author: Charlotte Liotta
"""

import numpy as np
import pandas as pd
import math
import copy
import scipy.io


def import_transport_costs(income_2011, param, grid, yearTraffic,
                           path_precalc_inp, spline_inflation, spline_fuel):
    """Compute travel times and costs."""
    # TODO: Call directly into import_transport_data?
    # STEP 1: IMPORT TRAVEL TIMES AND COSTS

    # Import travel times and distances
    # TODO: check calibration
    transport_times = scipy.io.loadmat(path_precalc_inp
                                       + 'Transport_times_GRID.mat')

    # TODO: Check tables from Basile to link with data

    # Price per km: see appendix B2 and Roux(2013), table 4.15
    priceTrainPerKMMonth = (
        0.164 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
                            )
    priceTrainFixedMonth = (
        4.48 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceTaxiPerKMMonth = (
        0.785 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceTaxiFixedMonth = (
        4.32 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceBusPerKMMonth = (
        0.522 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceBusFixedMonth = (
        6.24 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )

    inflation = spline_inflation(yearTraffic)
    inflation = spline_inflation(0)
    infla_2012 = spline_inflation(2012 - param["baseline_year"])
    priceTrainPerKMMonth = priceTrainPerKMMonth * inflation / infla_2012
    priceTrainFixedMonth = priceTrainFixedMonth * inflation / infla_2012
    priceTaxiPerKMMonth = priceTaxiPerKMMonth * inflation / infla_2012
    priceTaxiFixedMonth = priceTaxiFixedMonth * inflation / infla_2012
    priceBusPerKMMonth = priceBusPerKMMonth * inflation / infla_2012
    priceBusFixedMonth = priceBusFixedMonth * inflation / infla_2012
    priceFuelPerKMMonth = spline_fuel(yearTraffic)
    # if yearTraffic > 8:
    #     priceFuelPerKMMonth = priceFuelPerKMMonth * 1.2
    #     # priceBusPerKMMonth = priceBusPerKMMonth * 1.2
    #     # priceTaxiPerKMMonth = priceTaxiPerKMMonth * 1.2

    # Fixed costs
    #  See appendix B2
    priceFixedVehiculeMonth = 400
    priceFixedVehiculeMonth = priceFixedVehiculeMonth * inflation / infla_2012

    # STEP 2: TRAVEL TIMES AND COSTS AS MATRIX (no endogenous congestion)

    # Parameters: see appendix B2
    numberDaysPerYear = 235
    numberHourWorkedPerDay = 8
    # annualToHourly = 1 / (8*20*12)

    # Time taken by each mode in both direction (in min)
    # Includes walking time to station and other features from EMME/2 model
    timeOutput = np.empty(
        (transport_times["durationTrain"].shape[0],
         transport_times["durationTrain"].shape[1], 5)
        )
    timeOutput[:] = np.nan
    # To get walking times, we take 2 times the distances by car (to get trips
    # in both directions) multiplied by 1.2 (sinusoity coefficient), divided
    # by the walking speed (in km/h), which we multiply by 60 to get minutes
    timeOutput[:, :, 0] = (transport_times["distanceCar"]
                           / param["walking_speed"] * 60 * 1.2 * 2)
    timeOutput[:, :, 0][np.isnan(transport_times["durationCar"])] = np.nan
    timeOutput[:, :, 1] = copy.deepcopy(transport_times["durationTrain"])
    timeOutput[:, :, 2] = copy.deepcopy(transport_times["durationCar"])
    timeOutput[:, :, 3] = copy.deepcopy(transport_times["durationMinibus"])
    timeOutput[:, :, 4] = copy.deepcopy(transport_times["durationBus"])

    # Length (in km) using each mode (in direct line)
    multiplierPrice = np.empty((timeOutput.shape))
    multiplierPrice[:] = np.nan
    multiplierPrice[:, :, 0] = np.zeros((timeOutput[:, :, 0].shape))
    multiplierPrice[:, :, 1] = transport_times["distanceCar"]
    multiplierPrice[:, :, 2] = transport_times["distanceCar"]
    multiplierPrice[:, :, 3] = transport_times["distanceCar"]
    multiplierPrice[:, :, 4] = transport_times["distanceCar"]

    # Multiplying by 235 (nb of working days per year)
    pricePerKM = np.empty(5)
    pricePerKM[:] = np.nan
    pricePerKM[0] = np.zeros(1)
    pricePerKM[1] = priceTrainPerKMMonth*numberDaysPerYear
    pricePerKM[2] = priceFuelPerKMMonth*numberDaysPerYear
    pricePerKM[3] = priceTaxiPerKMMonth*numberDaysPerYear
    pricePerKM[4] = priceBusPerKMMonth*numberDaysPerYear

    # Distances (not useful to calculate price but useful output)
    distanceOutput = np.empty((timeOutput.shape))
    distanceOutput[:] = np.nan
    distanceOutput[:, :, 0] = transport_times["distanceCar"]
    distanceOutput[:, :, 1] = transport_times["distanceCar"]
    distanceOutput[:, :, 2] = transport_times["distanceCar"]
    distanceOutput[:, :, 3] = transport_times["distanceCar"]
    distanceOutput[:, :, 4] = transport_times["distanceCar"]

    # Monetary price per year (for each employment center)
    monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
    # trans_monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
    for index2 in range(0, 5):
        monetaryCost[:, :, index2] = (pricePerKM[index2]
                                      * multiplierPrice[:, :, index2])

    #  Train (monthly fare)
    monetaryCost[:, :, 1] = monetaryCost[:, :, 1] + priceTrainFixedMonth * 12
    #  Private car
    monetaryCost[:, :, 2] = (monetaryCost[:, :, 2] + priceFixedVehiculeMonth
                             * 12)
    #  Minibus/taxi
    monetaryCost[:, :, 3] = monetaryCost[:, :, 3] + priceTaxiFixedMonth * 12
    #  Bus
    monetaryCost[:, :, 4] = monetaryCost[:, :, 4] + priceBusFixedMonth * 12
    # trans_monetaryCost = copy.deepcopy(monetaryCost)

    # STEP 3: COMPUTE PROBA TO WORK IN C, EXPECTED INCOME, AND EXPECTED NB OF
    # RESIDENTS OF INCOME GROUP I WORKING IN C

    # In transport hours per working hour
    costTime = ((timeOutput * param["time_cost"])
                / (60 * numberHourWorkedPerDay))
    # We assume that people not taking some transport mode
    # have a extra high cost of doing so
    costTime[np.isnan(costTime)] = 10 ** 2

    return timeOutput, distanceOutput, monetaryCost, costTime


def import_employment_data(households_per_income_class, param, path_data):
    """Import number of jobs per selected employment center."""
    # Number of jobs per Transport Zone (TZ)
    TAZ = pd.read_csv(path_data + 'TAZ_amp_2013_proj_centro2.csv')

    # Number of employees in each TZ for the 12 income classes
    # NB: we break income classes given by CoCT to stick better to equivalent
    # from census data, as we will reweight towards the end
    jobsCenters12Class = np.array(
        [np.zeros(len(TAZ.Ink1)), TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink1/3,
         TAZ.Ink2/2, TAZ.Ink2/2, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink3/3,
         TAZ.Ink4/3, TAZ.Ink4/3, TAZ.Ink4/3]
        )

    # We get geographic coordinates
    codeCentersInitial = TAZ.TZ2013
    xCoord = TAZ.X / 1000
    yCoord = TAZ.Y / 1000

    # We arbitrarily set the threshold at 2,500 jobs
    # TODO: set as param
    selectedCenters = sum(jobsCenters12Class, 0) > 2500

    # Corrections where we don't have reliable transport data
    selectedCenters[xCoord > -10] = np.zeros(1, 'bool')
    selectedCenters[yCoord > -3719] = np.zeros(1, 'bool')
    selectedCenters[(xCoord > -20) & (yCoord > -3765)] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1010] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1012] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1394] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1499] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 4703] = np.zeros(1, 'bool')

    # xCenter = xCoord[selectedCenters]
    # yCenter = yCoord[selectedCenters]

    # Number of workers per group for the selected centers
    jobsCentersNgroup = np.zeros((len(xCoord), param["nb_of_income_classes"]))
    for j in range(0, param["nb_of_income_classes"]):
        jobsCentersNgroup[:, j] = np.sum(
            jobsCenters12Class[param["income_distribution"] == j + 1, :], 0)
    jobsCentersNgroup = jobsCentersNgroup[selectedCenters, :]
    # Rescale (wrt census data) to keep the correct global income distribution
    jobsCentersNGroupRescaled = (
        jobsCentersNgroup * households_per_income_class[None, :]
        / np.nansum(jobsCentersNgroup, 0)
        )

    return jobsCentersNGroupRescaled


def EstimateIncome(param, timeOutput, distanceOutput, monetaryCost, costTime,
                   job_centers, average_income, income_distribution,
                   list_lambda):
    """Solve for income per employment center for some values of lambda."""
    # Setting time and space

    #  We assume 8 working hours per day and 20 working days per month
    annualToHourly = 1 / (8*20*12)
    # bracketsTime = np.array(
    #     [0, 15, 30, 60, 90, np.nanmax(timeOutput)])
    #  TODO: meaning?
    bracketsDistance = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 200])

    timeCost = copy.deepcopy(costTime)
    monetary_cost = monetaryCost * annualToHourly
    monetary_cost[np.isnan(monetary_cost)] = 10 ** 3 * annualToHourly
    # transportTimes = timeOutput / 2
    transportDistances = distanceOutput[:, :, 0]

    # modalSharesTot = np.zeros((5, len(list_lambda)))
    incomeCentersSave = np.zeros((len(job_centers[:, 0]), 4, len(list_lambda)))
    # timeDistribution = np.zeros((len(bracketsTime) - 1, len(list_lambda)))
    distanceDistribution = np.zeros(
        (len(bracketsDistance) - 1, len(list_lambda)))

    # We begin simulations for different values of lambda

    for i in range(0, len(list_lambda)):

        param_lambda = list_lambda[i]

        print('Estimating for lambda = ', param_lambda)

        # We initialize output vectors
        incomeCentersAll = -math.inf * np.ones((len(job_centers[:, 0]), 4))
        # modalSharesGroup = np.zeros((5, 4))
        # timeDistributionGroup = np.zeros((len(bracketsTime) - 1, 4))
        distanceDistributionGroup = np.zeros((len(bracketsDistance) - 1, 4))

        # We run separate simulations for each income group

        for j in range(0, 4):

            # Household size varies with income group / transport costs
            householdSize = param["household_size"][j]
            # So does average income
            averageIncomeGroup = average_income[j] * annualToHourly

            print('incomes for group ', j)

            # We consider job centers where selected income group represents
            # more than 1/4 of the job threshold: it allows to avoid marginal
            # crossings between income classes and job centers, hence reduces
            # the number of equations to solve and makes optimization faster
            # (+ no corner cases)

            # NB: pb with Python numeric solver (gradient descent) when
            # function is not always differentiable (which is the case here as,
            # above/below some utility threshold, we have tipping effects),
            # hence we code our own solver to remain in the interior

            whichJobsCenters = job_centers[:, j] > 600
            popCenters = job_centers[whichJobsCenters, j]
            # We reweight population in each income group per SP to make it
            # comparable with population in job centers
            popResidence = (
                income_distribution[:, j]
                * sum(job_centers[whichJobsCenters, j])
                / sum(income_distribution[:, j])
                )

            # Numeric parameters come from trial and error and do not change
            # results a priori
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

            # Initializing the solver
            incomeCenters = np.zeros((sum(whichJobsCenters), maxIter))
            incomeCenters[:, 0] = (
                averageIncomeGroup
                * (popCenters / np.nanmean(popCenters)) ** (0.1)
                )
            # TODO: elicit later
            error[:, 0] = funSolve(incomeCenters[:, 0])

            while ((iter <= maxIter - 1) & (errorMax > tolerance)):

                incomeCenters[:, iter] = (
                    incomeCenters[:, max(iter-1, 0)]
                    + factorConvergenge
                    * averageIncomeGroup
                    * error[:, max(iter - 1, 0)]
                    / popCenters
                    )

                error[:, iter] = funSolve(incomeCenters[:, iter])
                errorMax = np.nanmax(np.abs(error[:, iter] / popCenters))
                scoreIter[iter] = np.nanmean(
                    np.abs(error[:, iter] / popCenters))
                print(np.nanmean(np.abs(error[:, iter])))
                iter = iter + 1

            if (iter > maxIter):
                scoreBest = np.amin(scoreIter)
                bestSolution = np.argmin(scoreIter)
                incomeCenters[:, iter-1] = incomeCenters[:, bestSolution]
                print(' - max iteration reached - mean error', scoreBest)
            else:
                print(' - computed - max error', errorMax)

            incomeCentersRescaled = (
                incomeCenters[:, iter-1] * averageIncomeGroup
                / ((np.nansum(incomeCenters[:, iter-1] * popCenters)
                    / np.nansum(popCenters)))
                )
            # modalSharesGroup[:,j] = modalShares(
            #     incomeCentersRescaled, popCenters, popResidence,
            #     monetary_cost[whichJobsCenters,:,:] * householdSize,
            #     timeCost[whichJobsCenters,:,:] * householdSize, param_lambda)
            incomeCentersAll[whichJobsCenters, j] = incomeCentersRescaled

            # timeDistributionGroup[:,j] = computeDistributionCommutingTimes(
            #     incomeCentersRescaled, popCenters, popResidence,
            #     monetary_cost[whichJobsCenters, :, :] * householdSize,
            #     timeCost[whichJobsCenters, :, :] * householdSize,
            #     transportTimes[whichJobsCenters,:], bracketsTime,
            #     param_lambda)
            (distanceDistributionGroup[:, j]
             ) = computeDistributionCommutingDistances(
                 incomeCentersRescaled, popCenters, popResidence,
                 monetary_cost[whichJobsCenters, :, :] * householdSize,
                 timeCost[whichJobsCenters, :, :] * householdSize,
                 transportDistances[whichJobsCenters, :], bracketsDistance,
                 param_lambda)

        # modalSharesTot[:, i] = (np.nansum(modalSharesGroup, 1)
        #                         / np.nansum(np.nansum(modalSharesGroup))
        incomeCentersSave[:, :, i] = incomeCentersAll / annualToHourly
        # timeDistribution[:,i] = (np.nansum(timeDistributionGroup, 1)
        #                          / np.nansum(np.nansum(timeDistributionGroup)
        #                          ))
        distanceDistribution[:, i] = (
            np.nansum(distanceDistributionGroup, 1)
            / np.nansum(np.nansum(distanceDistributionGroup))
            )

    return incomeCentersSave, distanceDistribution


def funSolve(incomeCentersTemp):
    """d."""
    return fun0(
        incomeCentersTemp, averageIncomeGroup, popCenters, popResidence,
        monetary_cost[whichJobsCenters, :, :] * householdSize,
        timeCost[whichJobsCenters, :, :] * householdSize, param_lambda
        )


def fun0(incomeCentersTemp, averageIncomeGroup, popCenters, popResidence,
         monetaryCost, timeCost, param_lambda):
    """Compute error in employment allocation."""
    # We redress the average income in each group per job center to match
    # income data
    # TODO: correspond to wages? Or timeCost already includes unemployment?
    incomeCentersFull = (
        incomeCentersTemp * averageIncomeGroup
        / ((np.nansum(incomeCentersTemp * popCenters) / np.nansum(popCenters)))
        )

    # Transport costs and employment allocation
    transportCostModes = (
        monetaryCost + timeCost * incomeCentersFull[:, None, None])

    # To prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    # Transport costs
    # Eq 2. Somme sur m de 1 à 5.
    transportCost = - 1 / param_lambda * (np.log(np.nansum(
        np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    # minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(
        np.nanmax(param_lambda * (incomeCentersFull[:, None] - transportCost))) - 500

    # Differences in the number of jobs
    #score = popCenters - np.nansum(np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome)) * popResidence, 1)
    proba = np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome) / np.nansum(
        np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome), 0)  # somme sur c
    # proba = np.exp((incomeCentersFull[:, None] - transportCost) - minIncome) / np.nansum(np.exp((incomeCentersFull[:, None] - transportCost) - minIncome), 0) #v2 sans le lambda
    score = popCenters - np.nansum(popResidence[None, :] * proba, 1)
    return score


def modalShares(incomeCenters, popCenters, popResidence, monetaryCost, timeCost, param_lambda):
    """ Computes total modal shares """

    # Transport cost by modes
    transportCostModes = monetaryCost + timeCost * incomeCenters[:, None, None]

    # Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    # Compute modal shares
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(
        np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]

    # Multiply by OD flows
    transportCost = - 1 / param_lambda * (np.log(np.nansum(
        np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    # minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(
        np.nanmax(param_lambda * (incomeCenters[:, None] - transportCost))) - 500

    # Total modal shares
    modalSharesTot = np.nansum(np.nansum(modalSharesTemp * (np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome) / np.nansum(
        np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)))[:, :, None] * popResidence, 1), 0)
    #modalSharesTot = np.tranpose(modalSharesTot, (2,0,1))

    return modalSharesTot


def computeDistributionCommutingTimes(incomeCenters, popCenters, popResidence, monetaryCost, timeCost, transportTime, bracketsTime, param_lambda):
    #incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, transportDistances[whichJobsCenters,:], bracketsDistance, param_lambda

    # Transport cost by modes
    transportCostModes = monetaryCost + timeCost * incomeCenters[:, None, None]

    # Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 600

    # Compute modal shares
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(
        np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]

    # Multiply by OD flows
    transportCost = - 1 / param_lambda * (np.log(np.nansum(
        np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    # minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(
        np.nanmax(param_lambda * (incomeCenters[:, None] - transportCost))) - 600

    # Total distribution of times
    nbCommuters = np.zeros(len(bracketsTime) - 1)
    for k in range(0, len(bracketsTime)-1):
        which = (transportTime > bracketsTime[k]) & (
            transportTime <= bracketsTime[k + 1]) & (~np.isnan(transportTime))
        nbCommuters[k] = np.nansum(np.nansum(np.nansum(which * modalSharesTemp * np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)[
                                   :, :, None] / np.nansum(np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)) * popResidence, 1)))

    return nbCommuters


def computeDistributionCommutingDistances(incomeCenters, popCenters, popResidence, monetaryCost, timeCost, transportDistance, bracketsDistance, param_lambda):
    #incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, transportDistances[whichJobsCenters,:], bracketsDistance, param_lambda

    # Transport cost by modes
    transportCostModes = monetaryCost + timeCost * incomeCenters[:, None, None]

    # Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    # Compute modal shares
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(
        np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]

    # Multiply by OD flows
    transportCost = - 1/param_lambda * (np.log(np.nansum(
        np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    # minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(
        np.nanmax(param_lambda * (incomeCenters[:, None] - transportCost))) - 500

    # Total distribution of times
    nbCommuters = np.zeros(len(bracketsDistance) - 1)
    for k in range(0, len(bracketsDistance)-1):
        which = (transportDistance > bracketsDistance[k]) & (
            transportDistance <= bracketsDistance[k + 1]) & (~np.isnan(transportDistance))
        nbCommuters[k] = np.nansum(np.nansum(np.nansum(which[:, :, None] * modalSharesTemp * np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)[
                                   :, :, None] / np.nansum(np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome), 0)[None, :, None] * popResidence[None, :, None], 1)))

    return nbCommuters
