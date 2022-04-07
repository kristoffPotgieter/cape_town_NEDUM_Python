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

    # Calculate amenities as a residual
    residualAmenities = (
        np.log(Uo[:, None])
        - np.log(
            (1 - beta) ** (1 - beta) * beta ** beta
            * (net_income[:, selectedRents]
               - basicQ * dataRent[None, selectedRents])
            / (dataRent[None, selectedRents] ** beta))
        )
    residualAmenities = np.nansum(
        residualAmenities * groupLivingSpMatrix[:, selectedRents], 0)
    residualAmenities[np.abs(residualAmenities.imag) > 0] = np.nan
    residualAmenities[residualAmenities == 0] = np.nan

    # Residual for the regression of amenities follow a log-normal law
    if (optionRegression == 0):
        # Here regression as a matrix division (much faster)
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

    # Method from Basile
    griddedRents = InterpolateRents(beta, basicQ, net_income)
    bidRents = np.empty((3, sum(selectedRents)))
    for i in range(0, 3):
        for j in range(0, sum(selectedRents)):
            bidRents[i, j] = griddedRents(
                net_income[:, selectedRents][i, j],
                (Uo[:, None] / residualAmenities[None, :])[i, j])

    # Estimation of the scale parameter by maximization of the log-likelihood
    selectedBidRents = (np.nansum(bidRents, 0) > 0)
    incomeGroupSelectedRents = groupLivingSpMatrix[:, selectedRents]
    likelihoodIncomeSorting = (
        lambda scaleParam:
            - (np.nansum(np.nansum(
                bidRents[:, selectedBidRents] / scaleParam
                * incomeGroupSelectedRents[:, selectedBidRents], 0))
                - np.nansum(np.log(np.nansum(
                    np.exp(bidRents[:, selectedBidRents] / scaleParam),
                    0))))
            )
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

    scoreHousing = 0
    parametersHousing = 0

    return (scoreTotal, scoreAmenities, scoreDwellingSize, scoreIncomeSorting,
            scoreHousing, parametersAmenities, modelAmenities,
            parametersHousing)


def utilityFromRents(Ro, income, basic_q, beta):
    """d."""
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

    utilityMatrix = utilityFromRents(
        np.matlib.repmat(decompositionRent, len(decompositionIncome), 1),
        np.transpose(np.matlib.repmat(
            decompositionIncome, len(decompositionRent), 1)),
        basicQ,
        beta
        )
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
