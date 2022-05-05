# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:13:50 2020

@author: Charlotte Liotta
"""

import numpy as np
import copy
import scipy
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def compute_housing_supply_formal(R, options, housing_limit, param, agricultural_rent, interest_rate, fraction_capital_destroyed, minimum_housing_supply, construction_param, housing_in, dwelling_size):
    ''' Calculates the housing construction as a function of rents '''

    if options["adjust_housing_supply"] == 1:
    
        capital_destroyed = np.ones(len(fraction_capital_destroyed.structure_formal_2))
        capital_destroyed[dwelling_size > param["threshold"]] = fraction_capital_destroyed.structure_formal_2[dwelling_size > param["threshold"]]
        capital_destroyed[dwelling_size <= param["threshold"]] = fraction_capital_destroyed.structure_formal_1[dwelling_size <= param["threshold"]]
        housing_supply = 1000000 * (construction_param ** (1/param["coeff_a"])) * ((param["coeff_b"] / (interest_rate + param["depreciation_rate"] + capital_destroyed)) ** (param["coeff_b"]/param["coeff_a"])) * ((R) ** (param["coeff_b"]/param["coeff_a"]))
    
    
        #Outside the agricultural rent, no housing (accounting for a tax)
        housing_supply[R < agricultural_rent] = 0
    
        housing_supply[np.isnan(housing_supply)] = 0
        #housing_supply[housing_supply.imag != 0] = 0
        housing_supply[housing_supply < 0] = 0
        housing_supply = np.minimum(housing_supply, housing_limit)
        
        #To add the construction on Mitchells_Plain
        housing_supply = np.maximum(housing_supply, minimum_housing_supply * 1000000)
    
    else:
    
        housing_supply = housing_in
    
    return housing_supply

def compute_housing_supply_backyard(R, param, income_net_of_commuting_costs, fraction_capital_destroyed, dwelling_size):
    """ Calculates the backyard available for construction as a function of rents """

    capital_destroyed = np.ones(len(fraction_capital_destroyed.structure_formal_2))
    capital_destroyed[dwelling_size > param["threshold"]] = fraction_capital_destroyed.structure_subsidized_2[dwelling_size > param["threshold"]]
    capital_destroyed[dwelling_size <= param["threshold"]] = fraction_capital_destroyed.structure_subsidized_1[dwelling_size <= param["threshold"]]
        
    housing_supply = (param["alpha"] * (param["RDP_size"] + param["backyard_size"] - param["q0"]) / (param["backyard_size"])) - (param["beta"] * (income_net_of_commuting_costs[0,:] - (capital_destroyed * param["subsidized_structure_value"])) / (param["backyard_size"] * R))
    
    housing_supply[R == 0] = 0
    housing_supply = np.minimum(housing_supply, 1)
    housing_supply = np.maximum(housing_supply, 0)
    housing_supply = 1000000 * housing_supply

    return housing_supply

def compute_dwelling_size_formal(utility, amenities, param, income_net_of_commuting_costs, fraction_capital_destroyed):
    
    income_temp = copy.deepcopy(income_net_of_commuting_costs)
    income_temp[income_temp < 0] = np.nan
    
    left_side = (utility[:, None] / amenities[None, :]) * ((1 + (param["fraction_z_dwellings"] * fraction_capital_destroyed.contents_formal[None, :])) ** (param["alpha"])) / ((param["alpha"] * income_temp) ** param["alpha"])
    
    approx = left_side ** (1/param["beta"])
    
    fun = lambda q: (q - param["q0"])/((q - (param["alpha"] * param["q0"])) ** param["alpha"])
    x = np.concatenate(([10 ** (-8), 10 ** (-7), 10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1)], np.arange(0.11, 0.15, 0.01), np.arange(0.15, 1.15, 0.05), np.arange(1.2, 3.1, 0.1), np.arange(3.5, 13.1, 0.25), np.arange(15, 60, 0.5), np.arange(60, 100, 2.5), np.arange(110, 210, 10), [250, 300, 500, 1000, 2000, 200000, 1000000, 10 ** 12]))
    f = interp1d(fun(x), x)
    dwelling_size = f(left_side)
    
    dwelling_size[dwelling_size > np.nanmax(x)] = np.nanmax(x)
    
    #hous = solus(income_temp, utility/amenities)
    #dwelling_size(Uo'*ones(1,size(income,2))./amenities > param.max_U) = param.max_q;
    
    #Solution 2

    #left_side = (utility[:, None] / amenities[None, :]) * ((1 + (param["fraction_z_dwellings"] * fraction_capital_destroyed.contents[None, :])) ** (param["alpha"])) / ((param["alpha"] * income_net_of_commuting_costs) ** param["alpha"])
    #approx = left_side ** (1/param["beta"])
    
    #dwelling_size = np.empty((4, len(amenities)))
    
    #for i in range(0, 4):
    #    print(i)
    #    for j in range(0, len(amenities)):
    #        print(j)
    #        fun = lambda q: np.abs(left_side[i, j] - ((q - param["q0"])/((q - (param["alpha"] * param["q0"])) ** param["alpha"])))
    #        res = minimize(fun, approx[i, j])
    #        dwelling_size[i, j] = res.x

    return dwelling_size