# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:13:50 2020.

@author: Charlotte Liotta
"""

import numpy as np
import copy
# import scipy
# from scipy.optimize import minimize
from scipy.interpolate import interp1d


def compute_dwelling_size_formal(utility, amenities, param,
                                 income_net_of_commuting_costs,
                                 fraction_capital_destroyed):
    """Return optimal dwelling size per income group for formal housing."""
    income_temp = copy.deepcopy(income_net_of_commuting_costs)
    income_temp[income_temp < 0] = np.nan

    # According to WP, corresponds to [(Q*-q_0)/(Q*-alpha x q_0)^(alpha)] x B
    # (draft, p.11), see theoretical expression in implicit_qfunc()
    left_side = (
        (np.array(utility)[:, None] / np.array(amenities)[None, :])
        * ((1 + (param["fraction_z_dwellings"]
                 * np.array(fraction_capital_destroyed.contents_formal)[
                     None, :])) ** (param["alpha"]))
        / ((param["alpha"] * income_temp) ** param["alpha"])
        )

    # approx = left_side ** (1/param["beta"])

    # We get a regression spline expressing q as a function of
    # implicit_qfunc(q) for some arbitrarily chosen q
    # TODO: where does it come from?
    x = np.concatenate((
        [10 ** (-8), 10 ** (-7), 10 ** (-6), 10 ** (-5), 10 ** (-4),
         10 ** (-3), 10 ** (-2), 10 ** (-1)],
        np.arange(0.11, 0.15, 0.01),
        np.arange(0.15, 1.15, 0.05),
        np.arange(1.2, 3.1, 0.1),
        np.arange(3.5, 13.1, 0.25),
        np.arange(15, 60, 0.5),
        np.arange(60, 100, 2.5),
        np.arange(110, 210, 10),
        [250, 300, 500, 1000, 2000, 200000, 1000000, 10 ** 12]))

    # TODO: Check whether extrapolation yields erroneous results
    # f = interp1d(implicit_qfunc(x, param["q0"], param["alpha"]), x,
    #              fill_value="extrapolate")

    f = interp1d(implicit_qfunc(x, param["q0"], param["alpha"]), x)

    # We define dwelling size as q corresponding to true values of
    # implicit_qfunc(q), for each selected pixel and each income group
    dwelling_size = f(left_side)

    # We cap dwelling size to 10**12 (why?)
    dwelling_size[dwelling_size > np.nanmax(x)] = np.nanmax(x)

    return dwelling_size


def implicit_qfunc(q, q_0, alpha):
    """Implicitely define optimal dwelling size."""
    # Note that with above x definition, q-alpha*q_0 can be negative

    # Note that numpy returns null when trying to get the fractional power of a
    # negative number (which is fine, because we are not interested in such
    # values), hence we ignore the error
    np.seterr(divide='ignore', invalid='ignore')
    result = (
        (q - q_0)
        / ((q - (alpha * q_0)) ** alpha)
        )

    return result


def compute_housing_supply_formal(
        R, options, housing_limit, param, agricultural_rent, interest_rate,
        fraction_capital_destroyed, minimum_housing_supply, construction_param,
        housing_in, dwelling_size
        ):
    """Calculate the formal housing supply as a function of rents."""
    if options["adjust_housing_supply"] == 1:
        # We consider two different damage functions above and below some
        # exogenous dwelling size threshold
        capital_destroyed = np.ones(
            len(fraction_capital_destroyed.structure_formal_2))
        (capital_destroyed[dwelling_size > param["threshold"]]
         ) = fraction_capital_destroyed.structure_formal_2[
             dwelling_size > param["threshold"]
             ]
        (capital_destroyed[dwelling_size <= param["threshold"]]
         ) = fraction_capital_destroyed.structure_formal_1[
             dwelling_size <= param["threshold"]
             ]

        # See research note, p.10
        # NB: we convert to supply per km²
        housing_supply = (
            1000000
            * (construction_param ** (1/param["coeff_a"]))
            * ((param["coeff_b"]
                / (interest_rate + param["depreciation_rate"]
                   + capital_destroyed))
               ** (param["coeff_b"]/param["coeff_a"]))
            * ((R) ** (param["coeff_b"]/param["coeff_a"]))
            )

        # Outside the agricultural rent, no housing (accounting for a tax)
        housing_supply[R < agricultural_rent] = 0

        housing_supply[np.isnan(housing_supply)] = 0
        # housing_supply[housing_supply.imag != 0] = 0
        housing_supply[housing_supply < 0] = 0
        housing_supply = np.minimum(housing_supply, housing_limit)

        # To add the construction on Mitchells_Plain
        housing_supply = np.maximum(
            housing_supply, minimum_housing_supply * 1000000)

    else:
        housing_supply = housing_in

    return housing_supply


def compute_housing_supply_backyard(R, param, income_net_of_commuting_costs,
                                    fraction_capital_destroyed, grid,
                                    income_class_by_housing_type):
    """Compute backyard housing supply as a function of rents."""
    # Same as before
    capital_destroyed = np.ones(
        len(fraction_capital_destroyed.structure_formal_2))
    # TODO: shouldn't we consider size of RDP instead?
    # Check potential dimensionality issues
    # capital_destroyed[dwelling_size > param["threshold"]
    #                   ] = fraction_capital_destroyed.structure_subsidized_2[
    #                       dwelling_size > param["threshold"]]
    # capital_destroyed[dwelling_size <= param["threshold"]
    #                   ] = fraction_capital_destroyed.structure_subsidized_1[
    #                       dwelling_size <= param["threshold"]]
    dwelling_size = param["RDP_size"] * np.ones((len(grid.dist)))
    # dwelling_size[income_class_by_housing_type.subsidized == 0, :] = np.nan
    capital_destroyed[dwelling_size > param["threshold"]
                      ] = fraction_capital_destroyed.structure_subsidized_2[
                          dwelling_size > param["threshold"]]
    capital_destroyed[dwelling_size <= param["threshold"]
                      ] = fraction_capital_destroyed.structure_subsidized_1[
                          dwelling_size <= param["threshold"]]
    # NB: in practice, the distinction is not used

    # See research note, p.11
    # TODO: Check that divide by zero come from groups 3 and 4
    # np.seterr(divide='ignore', invalid='ignore')
    housing_supply = (
        (param["alpha"] *
         (param["RDP_size"] + param["backyard_size"] - param["q0"])
         / (param["backyard_size"]))
        - (param["beta"]
           * (income_net_of_commuting_costs[0, :]
              - (capital_destroyed * param["subsidized_structure_value"]))
           / (param["backyard_size"] * R))
    )

    # NB: we convert units to km²
    housing_supply[R == 0] = 0
    housing_supply = np.minimum(housing_supply, 1)
    housing_supply = np.maximum(housing_supply, 0)
    housing_supply = 1000000 * housing_supply

    return housing_supply
