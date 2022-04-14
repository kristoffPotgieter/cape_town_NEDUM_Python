# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:46:13 2020.

@author: Charlotte Liotta
"""

import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp2d


def LogLikelihoodModel(X0, Uo2, net_income, groupLivingSpMatrix,
                       dataDwellingSize,
                       selectedDwellingSize, dataRent, selectedRents,
                       selectedDensity, predictorsAmenitiesMatrix,
                       tableRegression, variables_regression,
                       CalculateDwellingSize, ComputeLogLikelihood,
                       optionRegression):
    """Estimate the total likelihood of the model given the parameters."""
    beta = X0[0]
    basicQ = X0[1]
    Uo = np.array([Uo2, X0[2], X0[3]])

    # %% Errors on the amenity

    # Calculate amenities as a residual: corresponds to ln(A_s), appendix C4
    residualAmenities = (
        np.log(Uo[:, None])
        - np.log(
            (1 - beta) ** (1 - beta) * beta ** beta
            * (net_income[:, selectedRents]
               - basicQ * dataRent[None, selectedRents])
            / (dataRent[None, selectedRents] ** beta))
        )
    # We select amenities for dominant income groups and flatten the array
    residualAmenities = np.nansum(
        residualAmenities * groupLivingSpMatrix[:, selectedRents], 0)
    residualAmenities[np.abs(residualAmenities.imag) > 0] = np.nan
    residualAmenities[residualAmenities == 0] = np.nan

    # Residual for the regression of amenities follow a log-normal law
    if (optionRegression == 0):
        # Here regression as a matrix division (much faster)
        # TODO: note that predictors are dummies, and not log-values as in
        # paper
        A = predictorsAmenitiesMatrix[~np.isnan(residualAmenities), :]
        y = (residualAmenities[~np.isnan(residualAmenities)]).real
        parametersAmenities, residuals, rank, s = np.linalg.lstsq(A, y)
        errorAmenities = y - np.nansum(A * parametersAmenities, 1)
        modelAmenities = 0

    elif (optionRegression == 1):
        # Compute regression with fitglm (longer)
        # Can only work if length(lists) = 1
        residu = residualAmenities.real
        A = predictorsAmenitiesMatrix[~np.isnan(residualAmenities), :]
        y = (residualAmenities[~np.isnan(residualAmenities)]).real
        parametersAmenities, residuals, rank, s = np.linalg.lstsq(A, y)
        modelSpecification = sm.GLM(
            residu, tableRegression.loc[:, variables_regression])
        modelAmenities = modelSpecification.fit()
        print(modelAmenities.summary())
        errorAmenities = modelAmenities.resid_pearson

    scoreAmenities = ComputeLogLikelihood(
        np.sqrt(np.nansum(errorAmenities ** 2)
                / np.nansum(~np.isnan(errorAmenities))),
        errorAmenities)

    # %% Error on allocation of income groups

    # approx = (((1 - beta) ** (1 - beta)) * (beta ** beta)
    #           * (net_income[:, selectedRents])
    #           / (Uo[:, None] / residualAmenities[None, :])) ** (1/beta)

    # Here, we want the likelihood that simulated rent is equal to max bid rent
    # to reproduce observed income sorting

    # Method from Basile
    #  We get a function that predicts rent based on any given income and u/A,
    #  interpolated from parameter initial values
    griddedRents = InterpolateRents(beta, basicQ, net_income)
    bidRents = np.empty((3, sum(selectedRents)))
    #  For each income group and each selected SP,
    #  we obtain the bid rent as a function of initial income and u/ln(A)
    #  TODO: shouldn't we give u/A as an argument instead?
    #  TODO: should we care about the fit with observed rents?
    for i in range(0, 3):
        for j in range(0, sum(selectedRents)):
            bidRents[i, j] = griddedRents(
                net_income[:, selectedRents][i, j],
                (Uo[:, None] / np.exp(residualAmenities[None, :]))[i, j]
                )

    # Estimation of the parameters by maximization of the log-likelihood
    # (in overarching function)
    selectedBidRents = (np.nansum(bidRents, 0) > 0)
    incomeGroupSelectedRents = groupLivingSpMatrix[:, selectedRents]
    # Corresponds to formula in appendix C4
    likelihoodIncomeSorting = (
        lambda scaleParam:
            - np.nansum(np.nansum(
                bidRents[:, selectedBidRents] / scaleParam
                * incomeGroupSelectedRents[:, selectedBidRents], 0))
            - np.nansum(np.log(np.nansum(
                np.exp(bidRents[:, selectedBidRents] / scaleParam),
                0)))
            )
    # TODO: does the choice of scale parameter matter?
    scoreIncomeSorting = - likelihoodIncomeSorting(10000)

    # %% Errors on the dwelling sizes
    # Simulated rent, real sorting
    simulatedRents = np.nansum(
        bidRents[:, selectedDwellingSize[selectedRents]]
        * groupLivingSpMatrix[:, selectedDwellingSize],
        0)
    dwellingSize = CalculateDwellingSize(
        beta,
        basicQ,
        np.nansum(net_income[:, selectedDwellingSize]
                  * groupLivingSpMatrix[:, selectedDwellingSize], 0),
        simulatedRents)

    # Define errors
    # Here we call on real data as it is part of the error term definition
    errorDwellingSize = (
        np.log(dwellingSize)
        - np.log(dataDwellingSize[selectedDwellingSize])
        )
    scoreDwellingSize = ComputeLogLikelihood(
        np.sqrt(np.nansum(errorDwellingSize ** 2)
                / np.nansum(~np.isnan(errorDwellingSize))),
        errorDwellingSize)

    # %% Total

    scoreTotal = scoreAmenities + scoreDwellingSize + scoreIncomeSorting

    # TODO: why?
    scoreHousing = 0
    parametersHousing = 0

    return (scoreTotal, scoreAmenities, scoreDwellingSize, scoreIncomeSorting,
            scoreHousing, parametersAmenities, modelAmenities,
            parametersHousing)


def utilityFromRents(Ro, income, basic_q, beta):
    """d."""
    # Equal to u/A
    utility = (((1 - beta) ** (1 - beta))
               * (beta ** beta)
               * (income - (basic_q * Ro))
               / (Ro ** beta))
    utility[(income - (basic_q * Ro)) < 0] = 0
    utility[income == 0] = 0
    return utility


def InterpolateRents(beta, basicQ, net_income):
    """Precalculate for rents, as a function."""
    # The output of the function is a griddedInterpolant object, that gives the
    # log of rents as a function of the log utility and the log income.

    # Decomposition for the interpolation (the more points, the slower)
    decompositionRent = np.concatenate(
        ([np.array([10 ** (-9), 10 ** (-4), 10 ** (-3), 10 ** (-2)]),
          np.arange(0.02, 0.80, 0.01),
          np.arange(0.8, 1, 0.02)])
        )
    decompositionIncome = np.concatenate(
        (np.array([10 ** (-9)]),
         10 ** np.arange(-4, -2, 0.5),
         np.array([0.03]),
         np.arange(0.06, 1.4, 0.02),
         np.arange(1.5, 2.5, 0.1),
         np.arange(4, 10, 2),
         np.array([20, 10 ** 9]))
        )

    # Min and Max values for the decomposition

    #  Yields u/A for all values of decomposition
    utilityMatrix = utilityFromRents(
        np.matlib.repmat(decompositionRent, len(decompositionIncome), 1),
        np.transpose(np.matlib.repmat(
            decompositionIncome, len(decompositionRent), 1)),
        basicQ,
        beta
        )
    #  We interpolate rent as a function of income and u/A (calculated upon
    #  initial values of parameters...)
    #  TODO: why choose this specific functional form?
    #  Is it because we already use the equations we have for LL on amenities
    #  and dwelling size? Note that we observe rent but not actual max bid rent
    #  for all income groups
    solusRentTemp = (
        lambda x, y:
            interp2d(
                np.transpose(np.matlib.repmat(
                    decompositionIncome, len(decompositionRent), 1)),
                utilityMatrix,
                np.matlib.repmat(
                    decompositionRent, len(decompositionIncome), 1)
                )(x, y)
            )

    return solusRentTemp
