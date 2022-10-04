# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import copy
import math


def import_transport_costs(grid, param, yearTraffic,
                           households_per_income_class,
                           spline_inflation, spline_fuel,
                           spline_population_income_distribution,
                           spline_income_distribution, path_precalc_inp,
                           path_precalc_transp, dim, options):
    """Compute job center distribution, commuting and net income."""
    # STEP 1: IMPORT TRAVEL TIMES AND COSTS

    # Import travel times and distances
    transport_times = scipy.io.loadmat(path_precalc_inp
                                       + 'Transport_times_' + dim)

    # Price per km: see appendix B2 and Basile's regressions based upon
    # Roux(2013), table 4.15
    # Note that we take into account the difference in inflation between the
    # year the data was collected and the year of our analysis
    # NB: 40 in variable cost correspond to 20 working days per month, times
    # 2 for the round trip
    priceTrainPerKMMonth = (
        0.164 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
                            )
    priceTrainFixedMonth = (
        4.48 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    # Note that bus and taxi had been interverted in Basile's code
    priceBusPerKMMonth = (
        0.785 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceBusFixedMonth = (
        4.32 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceTaxiPerKMMonth = (
        0.522 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceTaxiFixedMonth = (
        6.24 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )

    # Again, we correct for inflation wrt reference year
    inflation = spline_inflation(yearTraffic)
    if options["correct_infla_base"] == 0:
        infla_base = spline_inflation(2012 - param["baseline_year"])
    elif options["correct_infla_base"] == 1:
        infla_base = spline_inflation(2011 - param["baseline_year"])
    priceTrainPerKMMonth = priceTrainPerKMMonth * inflation / infla_base
    priceTrainFixedMonth = priceTrainFixedMonth * inflation / infla_base
    priceTaxiPerKMMonth = priceTaxiPerKMMonth * inflation / infla_base
    priceTaxiFixedMonth = priceTaxiFixedMonth * inflation / infla_base
    priceBusPerKMMonth = priceBusPerKMMonth * inflation / infla_base
    priceBusFixedMonth = priceBusFixedMonth * inflation / infla_base
    priceFuelPerKMMonth = spline_fuel(yearTraffic)

    # Fixed costs
    #  See appendix B2
    priceFixedVehiculeMonth = 400
    priceFixedVehiculeMonth = priceFixedVehiculeMonth * inflation / infla_base

    # STEP 2: TRAVEL TIMES AND COSTS AS MATRIX (no endogenous congestion)

    # Parameters: see appendix B2
    numberDaysPerYear = 235
    numberHourWorkedPerDay = 8
    #  We assume 8 working hours per day and 20 working days per month
    # annualToHourly = 1 / (8*20*12)

    # Time taken by each mode for the round trip (in min)
    # Includes walking time to station and other features from EMME/2 model
    timeOutput = np.empty(
        (transport_times["durationTrain"].shape[0],
         transport_times["durationTrain"].shape[1], 5)
        )
    timeOutput[:] = np.nan
    # To get walking times, we take 2 times the distances by car (to get trips
    # in both directions) multiplied by 1.2 (sinusoity coefficient), divided
    # by the walking speed (in km/h), which we multiply by 60 to get minutes
    # NB: see Viguié et al. (2014), table B.1 for sinusoity estimate
    # NB: duration is already for round trips in principle, as we would get low
    # average speeds of 13 km/h otherwise
    timeOutput[:, :, 0] = (transport_times["distanceCar"]
                           / param["walking_speed"] * 60 * 1.2 * 2)
    timeOutput[:, :, 0][np.isnan(transport_times["durationCar"])] = np.nan
    timeOutput[:, :, 1] = copy.deepcopy(transport_times["durationTrain"])
    timeOutput[:, :, 2] = copy.deepcopy(transport_times["durationCar"])
    timeOutput[:, :, 3] = copy.deepcopy(transport_times["durationMinibus"])
    timeOutput[:, :, 4] = copy.deepcopy(transport_times["durationBus"])

    # Length (in km) using each mode
    if options["correct_round_trip"] == 1:
        multiplierPrice = np.empty((timeOutput.shape))
        multiplierPrice[:] = np.nan
        multiplierPrice[:, :, 0] = np.zeros((timeOutput[:, :, 0].shape))
        multiplierPrice[:, :, 1] = transport_times["distanceCar"] * 2
        multiplierPrice[:, :, 2] = transport_times["distanceCar"] * 2
        multiplierPrice[:, :, 3] = transport_times["distanceCar"] * 2
        multiplierPrice[:, :, 4] = transport_times["distanceCar"] * 2
    elif options["correct_round_trip"] == 0:
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

    # Simple distances (not useful to calculate price but interesting output)
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

    # We assume that people not taking some transport mode
    # have a extra high cost of doing so"
    # monetaryCost[np.isnan(monetaryCost)] = 10 ** 3
    monetaryCost[np.isnan(monetaryCost)] = 10 ** 5

    # trans_monetaryCost = copy.deepcopy(monetaryCost)

    # STEP 3: COMPUTE PROBA TO WORK IN C, EXPECTED INCOME, AND EXPECTED NB OF
    # RESIDENTS OF INCOME GROUP I WORKING IN C

    # In transport hours per working hours in a day
    costTime = (timeOutput * param["time_cost"]
                / (60 * numberHourWorkedPerDay))
    # We assume that people not taking some transport mode
    # have a extra high cost of doing so
    costTime[np.isnan(costTime)] = 10 ** 2

    return timeOutput, distanceOutput, monetaryCost, costTime


def EstimateIncome(param, timeOutput, distanceOutput, monetaryCost, costTime,
                   job_centers, average_income, income_distribution,
                   list_lambda, options):
    """Solve for income per employment center for some values of lambda."""
    # Setting time and space
    annualToHourly = 1 / (8*20*12)
    #  Corresponds to the brackets for which we have aggregate statistics on
    #  the nb of commuters to fit our calibration
    bracketsDistance = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 200])
    monetary_cost = monetaryCost * annualToHourly

    #  We initialize income and distance output vectors
    incomeCentersSave = np.zeros((len(job_centers[:, 0]), 4, len(list_lambda)))
    distanceDistribution = np.zeros(
        (len(bracketsDistance) - 1, len(list_lambda)))

    scoreMatrix = np.zeros((len(list_lambda), param["nb_of_income_classes"]))

    # We begin simulations for different values of lambda

    for i in range(0, len(list_lambda)):

        param_lambda = list_lambda[i]

        print('Estimating for lambda = ', param_lambda)

        # We initialize output vectors for each lambda
        incomeCentersAll = -math.inf * np.ones((len(job_centers[:, 0]), 4))
        distanceDistributionGroup = np.zeros((len(bracketsDistance) - 1, 4))

        # We run separate simulations for each income group

        for j in range(0, param["nb_of_income_classes"]):

            # Household size varies with income group / transport costs
            householdSize = param["household_size"][j]
            # So does average income (which needs to be adapted to hourly
            # from income data)
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

            whichCenters = job_centers[:, j] > 600
            # whichCenters = (
            #     job_centers[:, j] > param["job_center_threshold"] / 4)
            popCenters = job_centers[whichCenters, j]

            # We reweight population in each income group per SP to make it
            # comparable with population in SELECTED job centers
            # Note that unemployed population is not included!
            # Also note that SP data includes more areas than included in grid
            popResidence = (
                income_distribution[:, j]
                * sum(job_centers[whichCenters, j])
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
            elif j == 2:
                factorConvergenge = 0.003
            elif j == 3:
                factorConvergenge = 0.001

            iter = 0
            error = np.zeros((len(popCenters), maxIter))
            scoreIter = np.zeros(maxIter)
            errorMax = 1

            # Initializing the solver
            incomeCenters = np.zeros((sum(whichCenters), maxIter))
            incomeCenters[:, 0] = (
                averageIncomeGroup
                * (popCenters / np.nanmean(popCenters))
                ** (0.1)
            )
            # Initial error corresponds to the difference between observed
            # and simulated population working in each job center
            error[:, 0], _ = funSolve(incomeCenters[:, 0], averageIncomeGroup,
                                      popCenters, popResidence,
                                      monetary_cost, costTime, param_lambda,
                                      householdSize, whichCenters,
                                      bracketsDistance, distanceOutput,
                                      options)

            # Then we iterate by adding to each job center the average value of
            # its income weighted by the importance of the error relative to
            # its observed population: if we underestimate the population, we
            # increase the income (cf. equation 3)
            while ((iter <= maxIter - 1) & (errorMax > tolerance)):

                incomeCenters[:, iter] = (
                    incomeCenters[:, max(iter-1, 0)]
                    + factorConvergenge
                    * averageIncomeGroup
                    * error[:, max(iter - 1, 0)]
                    / popCenters
                )

                # We also update the error term before and store some values
                # before iterating over
                error[:, iter], _ = funSolve(
                    incomeCenters[:, iter], averageIncomeGroup, popCenters,
                    popResidence, monetary_cost, costTime, param_lambda,
                    householdSize, whichCenters, bracketsDistance,
                    distanceOutput, options)

                errorMax = np.nanmax(
                    np.abs(error[:, iter] / popCenters))
                scoreIter[iter] = np.nanmean(
                    np.abs(error[:, iter] / popCenters))
                # print(np.nanmean(np.abs(error[:, iter])))

                iter = iter + 1

            # At the end of the process, we keep the minimum score, and define
            # the corresponding best solution for some lambda and income group
            if (iter > maxIter - 1):
                scoreBest = np.amin(scoreIter)
                scoreMatrix[i, j] = scoreBest
                bestSolution = np.argmin(scoreIter)
                incomeCenters[:, iter-1] = incomeCenters[:, bestSolution]
                print(' - max iteration reached - mean error', scoreBest)

            # If we manage to have a maximum error that falls under the
            # tolerance threshold, we leave the loop and consider the solution
            # corresponding to the latest iteration
            else:
                scoreBest = scoreIter[iter-1]
                scoreMatrix[i, j] = scoreBest
                print(' - computed - max error', errorMax)

            # We also get (for the given income group) the number of commuters
            # for all job centers in given distance brackets
            _, distanceDistributionGroup[:, j] = funSolve(
                incomeCenters[:, iter-1], averageIncomeGroup, popCenters,
                popResidence, monetary_cost, costTime, param_lambda,
                householdSize, whichCenters, bracketsDistance, distanceOutput,
                options)

            # We rescale the parameter to stick to overall income data scale:
            # remember that we only computed it for a subset of the population
            incomeCentersRescaled = (
                incomeCenters[:, iter-1] * averageIncomeGroup
                / ((np.nansum(incomeCenters[:, iter-1] * popCenters)
                    / np.nansum(popCenters)))
            )

            # Then we update the output vector for the given income group
            # (with lambda still fixed)
            incomeCentersAll[whichCenters, j] = incomeCentersRescaled

        # We can now loop over different values of lambda and store values back
        # in yearly format for incomes
        incomeCentersSave[:, :, i] = incomeCentersAll / annualToHourly

        # Likewise, for each value of lambda, we store the % of total commuters
        # for each distance bracket
        distanceDistribution[:, i] = (
            np.nansum(distanceDistributionGroup, 1)
            / np.nansum(distanceDistributionGroup)
        )

    return incomeCentersSave, distanceDistribution, scoreMatrix


def compute_ODflows(householdSize, monetaryCost, costTime, incomeCentersFull,
                    whichCenters, param_lambda):
    """Apply commuting formulas from working paper."""
    # Corresponds to t_mj without error term (explained cost)
    #  Note that incomeCentersFull correspond to y_ic, hence householdSize is
    #  already taken into account as a multiplier of w_ic, therefore there
    #  is no need to multiply the second term by householdSize (and both
    #  monetary and time costs indeed correspond to round trips).
    #  However, the monetary cost should still be taken into account for the
    #  two members of the household, hence the first factor.

    transportCostModes = (
        (householdSize * monetaryCost[whichCenters, :, :]
         + (costTime[whichCenters, :, :]
            * incomeCentersFull[:, None, None]))
    )

    # This prevents exponentials/logarithms from diverging towards infinity
    # This is neutral on the result and is set by trial and error
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    # Transport costs (min_m(t_mj))
    # NB: here, we consider the Gumbel quantile function
    transportCost = (
        - 1 / param_lambda
        * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes
                                   + valueMax[:, :, None]), 2)) - valueMax)
    )

    # Also for exponential/logarithm convergence
    minIncome = (np.nanmax(
        param_lambda * (incomeCentersFull[:, None] - transportCost), 0)
        - 700
        )

    # OD flows: corresponds to pi_c|ix (here, not the full matrix)
    # NB: here, we consider maximum Gumbel (typo in original paper)
    ODflows = (
        np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost)
               - minIncome)
        / np.nansum(np.exp(param_lambda * (incomeCentersFull[:, None]
                                           - transportCost) - minIncome),
                    0)[None, :]
    )

    return transportCostModes, transportCost, ODflows, valueMax, minIncome


def funSolve(incomeCentersTemp, averageIncomeGroup, popCenters,
             popResidence, monetaryCost, costTime, param_lambda, householdSize,
             whichCenters, bracketsDistance, distanceOutput, options):
    """Compute error in employment allocation."""
    # We redress the average income in each group per job center to match
    # income data, as this must be matched with popResidence
    incomeCentersFull = (
        incomeCentersTemp * averageIncomeGroup
        / ((np.nansum(incomeCentersTemp * popCenters)
            / np.nansum(popCenters)))
    )

    transportCostModes, transportCost, ODflows, *_ = compute_ODflows(
        householdSize, monetaryCost, costTime, incomeCentersFull,
        whichCenters, param_lambda)

    # We compare the true population distribution with its theoretical
    # equivalent computed from simulated net income distribution: the closer
    # the score is to zero, the better (see equation 3 for formula)

    # Note that ODflows is of dimension 119x24014
    # NB: correction is not needed a priori, as income distribution data from
    # SP does not include people out of employment
    if options["correct_eq3"] == 1:
        score = (popCenters
                 - (householdSize / 2
                    * np.nansum(popResidence[None, :] * ODflows, 1))
                 )
    elif options["correct_eq3"] == 0:
        score = (popCenters
                 - np.nansum(popResidence[None, :] * ODflows, 1))

    # We also return the total number of commuters (in a given income group)
    # for job centers in predefined distance brackets
    nbCommuters = np.zeros(len(bracketsDistance) - 1)
    for k in range(0, len(bracketsDistance)-1):
        which = ((distanceOutput[whichCenters, :] > bracketsDistance[k])
                 & (distanceOutput[whichCenters, :] <= bracketsDistance[k + 1])
                 & (~np.isnan(distanceOutput[whichCenters, :])))
        if options["correct_eq3"] == 1:
            nbCommuters[k] = (
                np.nansum(which * ODflows * popResidence[None, :])
                * (householdSize / 2)
                )
        elif options["correct_eq3"] == 0:
            nbCommuters[k] = (
                np.nansum(which * ODflows * popResidence[None, :]))

    return score, nbCommuters
