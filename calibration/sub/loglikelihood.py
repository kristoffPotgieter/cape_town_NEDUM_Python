# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:46:13 2020.

@author: Charlotte Liotta
"""

import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp2d
from scipy import interpolate
import scipy
# from numba import jit


# @jit
def LogLikelihoodModel(X0, Uo2, net_income, groupLivingSpMatrix,
                       dataDwellingSize,
                       selectedDwellingSize, dataRent, selectedRents,
                       selectedDensity, predictorsAmenitiesMatrix,
                       tableRegression, variables_regression,
                       CalculateDwellingSize, ComputeLogLikelihood,
                       optionRegression, options):
    """Estimate the total likelihood of the model given the parameters."""
    beta = X0[0]
    basicQ = X0[1]
    Uo = np.array([Uo2, X0[2], X0[3]])

    # %% Errors on the amenity

    # Calculate amenities as a residual: corresponds to ln(A_s), appendix C4
    # residualAmenities = (
    #     np.log(Uo[:, None])
    #     - np.log(
    #         (1 - beta) ** (1 - beta) * beta ** beta
    #         * (net_income[:, selectedRents]
    #            - basicQ * dataRent[None, selectedRents])
    #         / (dataRent[None, selectedRents] ** beta))
    #     )
    residualAmenities = (
        np.log(np.array(Uo)[:, None])
        - np.log(
            (1 - beta) ** (1 - beta) * beta ** beta
            * (net_income[:, selectedRents]
               - basicQ * np.array(dataRent)[None, selectedRents])
            / (np.array(dataRent)[None, selectedRents] ** beta))
        )
    # We select amenities for dominant income groups and flatten the array
    residualAmenities = np.nansum(
        residualAmenities * groupLivingSpMatrix[:, selectedRents], 0)
    residualAmenities[np.abs(residualAmenities.imag) > 0] = np.nan
    residualAmenities[residualAmenities == 0] = np.nan

    # Residual for the regression of amenities follow a log-normal law
    if (optionRegression == 0):
        # Here regression as a matrix division (much faster)
        # Note that predictors are dummies, and not log-values as in paper
        A = predictorsAmenitiesMatrix[~np.isnan(residualAmenities), :]
        y = (residualAmenities[~np.isnan(residualAmenities)]).real
        # parametersAmenities, residuals, rank, s = np.linalg.lstsq(A, y,
        #                                                           rcond=None)
        # res = scipy.optimize.lsq_linear(A, y)
        # parametersAmenities = res.x
        # residuals = res.fun
        modelSpecification = sm.OLS(y, A)
        modelAmenities = modelSpecification.fit()
        parametersAmenities = modelAmenities.params
        errorAmenities = modelAmenities.resid_pearson
        # errorAmenities = y - np.nansum(A * parametersAmenities, 1)
        # modelAmenities = 0

    elif (optionRegression == 1):
        # Compute regression with fitglm (longer)
        # Can only work if length(lists) = 1
        residu = residualAmenities.real
        A = predictorsAmenitiesMatrix[~np.isnan(residualAmenities), :]
        y = (residualAmenities[~np.isnan(residualAmenities)]).real
        parametersAmenities, residuals, rank, s = np.linalg.lstsq(A, y,
                                                                  rcond=None)
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

    # Here, we want the likelihood that simulated rent is equal to max bid rent
    # to reproduce observed income sorting

    #  We get a function that predicts ln(rent) based on any given ln(income)
    #  and ln(u/A), interpolated from parameter initial values
    griddedRents = InterpolateRents(beta, basicQ, net_income, options)
    bidRents = np.empty((3, sum(selectedRents)))
    for i in range(0, 3):
        for j in range(0, sum(selectedRents)):
            if options["griddata"] == 0 & options["log_form"] == 1:
                bidRents[i, j] = np.exp(griddedRents(
                    (np.log(np.array(Uo)[:, None])
                     - np.array(residualAmenities)[None, :])[i, j],
                    np.log(net_income[:, selectedRents][i, j])
                    ))
            # elif options["griddata"] == 0 & options["log_form"] == 0:
            #     bidRents[i, j] = griddedRents(
            #         (np.array(Uo)[:, None]
            #          / np.exp(np.array(residualAmenities)[None, :])[i, j]),
            #         net_income[:, selectedRents][i, j]
            #         )
            # elif options["griddata"] == 1 & options["log_form"] == 1:
            #     coord = np.stack([
            #         (np.log(np.array(Uo)[:, None])
            #          - np.array(residualAmenities)[None, :])[i, j],
            #         np.log(net_income[:, selectedRents][i, j])
            #         ], -1)
            #     coord = np.expand_dims(coord, axis=0)
            #     bidRents[i, j] = np.exp(griddedRents(coord))

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

    # We then optimize over the scale parameter to be used in score
    bnds = {(0, 10**10)}
    initScale = 10**9
    res = scipy.optimize.minimize(
        likelihoodIncomeSorting, initScale, bounds=bnds,
        options={'maxiter': 10, 'disp': True})
    # scaleParameter = res.x
    errorIncomeSorting = res.fun
    exitFlag = res.success
    # print(exitFlag)
    if exitFlag is True:
        scoreIncomeSorting = - errorIncomeSorting
    elif exitFlag is False:
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

    # We may also include a measure of the fit for household density,
    # which has not been retained in this version
    # NB: this is because it is already taken into account in other likelihoods
    # and we want to avoid overfit
    scoreHousing = 0
    parametersHousing = 0

    return (scoreTotal, scoreAmenities, scoreDwellingSize, scoreIncomeSorting,
            scoreHousing, parametersAmenities, modelAmenities,
            parametersHousing)


def utilityFromRents(Ro, income, basic_q, beta):
    """Return utility / amenity index ratio."""
    # Equal to u/A (equation C2)
    utility = (((1 - beta) ** (1 - beta))
               * (beta ** beta)
               * (income - (basic_q * Ro))
               / (Ro ** beta))
    utility[(income - (basic_q * Ro)) < 0] = 0
    utility[income == 0] = 0
    return utility


# @jit
def InterpolateRents(beta, basicQ, net_income, options):
    """Interpolate log(rents) as a function of log(beta) and log(q0)."""
    # Decomposition for the interpolation (the more points, the slower)
    # TODO: should be changed?
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

    if options["griddata"] == 0:

        # We scale the income vector accordingly
        choiceIncome = 100000 * decompositionIncome
        incomeMatrix = np.matlib.repmat(
            choiceIncome, len(decompositionRent), 1)
        # We do the same for rent vector by considering that the rent is max
        # when utility equals zero
        if options["test_maxrent"] == 0:
            choiceRent = choiceIncome / basicQ
        elif options["test_maxrent"] == 1:
            choiceRent = choiceIncome
        rentMatrix = (np.array(choiceRent)[:, None]
                      * np.array(decompositionRent)[None, :])
        #  Yields u/A for all values of decomposition
        utilityMatrix = utilityFromRents(
            rentMatrix, np.transpose(incomeMatrix), basicQ, beta)

        #  We interpolate ln(rent) as a function of ln(income) and ln(u/A)
        #  (calculated upon initial values of parameters)
        #  Note that we observe rent but not actual max bid rent
        #  for all income groups

        # We first do it in simple form
        solusRentTemp = (
            lambda x, y:
                interp2d(
                    np.transpose(incomeMatrix),
                    utilityMatrix,
                    rentMatrix  # ** beta
                    )(x, y)
                )

        if options["log_form"] == 1:
            # Then we go to log-form (quickens computations and helps convergence)
            utilityVectLog = np.arange(-1, np.log(np.nanmax(10 * net_income)), 0.1)
            incomeLog = np.arange(
                -1, np.log(np.nanmax(np.nanmax(10 * net_income))), 0.2)
            rentLog = (
                np.log(solusRentTemp(np.exp(incomeLog), np.exp(utilityVectLog)))
                )  # * 1/beta

            griddedRents = interp2d(
                utilityVectLog, incomeLog, np.transpose(rentLog))

            return griddedRents

        elif options["log_form"] == 0:
            return solusRentTemp

    if options["griddata"] == 1:

        # We scale the income vector accordingly
        choiceIncome = 100000 * decompositionIncome
        incomeVector = np.repeat(choiceIncome, len(decompositionRent))
        # We do the same for rent vector by considering that the rent is max
        # when utility equals zero
        if options["test_maxrent"] == 0:
            choiceRent = choiceIncome / basicQ
        elif options["test_maxrent"] == 1:
            choiceRent = choiceIncome
        rentList = [rent * decompositionRent for rent in choiceRent]
        rentVector = np.concatenate(rentList)

        logincomeVector = np.log(incomeVector)
        logrentVector = np.log(rentVector)

        y, z = logincomeVector, logrentVector
        ey, ez = np.exp(y), np.exp(z)
        x = np.log(utilityFromRents(ez, ey, basicQ, beta))
        ex = np.exp(x)

        # npts = 400
        # index_select = np.random.randint(
        #     0, len(logincomeVector), size=npts)
        # py, pz = logincomeVector[index_select], logrentVector[index_select]
        # epy, epz = np.exp(py), np.exp(pz)
        # px = np.log(utilityFromRents(epz, epy, basicQ, beta))
        # pX, pY = np.meshgrid(px, py)
        # X, Y = np.meshgrid(x, y)

        if options["log_form"] == 1:
            points = np.stack([x, y], -1)
            # TODO: play with kernel?
            griddedRents = interpolate.RBFInterpolator(
                points, z, neighbors=options["interpol_neighbors"]
                )
            # griddedRents = interpolate.griddata((x, y), z, (pX, pY))

        elif options["log_form"] == 0:
            points = np.stack([ex, ey], -1)
            griddedRents = interpolate.RBFInterpolator(
                points, ez, neighbors=options["interpol_neighbors"]
                )

        return griddedRents
