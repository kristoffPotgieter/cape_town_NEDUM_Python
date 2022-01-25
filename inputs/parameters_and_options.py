# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:50:59 2020.

@author: Charlotte Liotta
"""

import numpy as np
import scipy.io
import copy


def import_options():
    """Import default options."""
    # Not useful for now (for coding green belt)
    options = {"urban_edge": 0}
    # Used in solver_equil (dummy for housing supply adaptability)
    options["adjust_housing_supply"] = 1
    # Dummy for agents taking floods into account in their choices
    options["agents_anticipate_floods"] = 1
    # Dummy for using flood data from WBUS2 on top of FATHOM
    options["WBUS2"] = 0
    # Dummy for considering pluvial floods on top of fluvial floods
    options["pluvial"] = 1
    # Dummy for new informal housing construction possibility
    options["informal_land_constrained"] = 0

    return options


def import_param(precalculated_inputs):
    """Import default parameters."""
    # Define baseline year
    param = {"baseline_year": 2011}

    # Utility function parameters, as calibrated in Pfeiffer et al. (table C7)
    #  Surplus housing elasticity
    param["beta"] = scipy.io.loadmat(
        precalculated_inputs + 'calibratedUtility_beta.mat'
        )["calibratedUtility_beta"].squeeze()
    #  Basic need in housing
    param["q0"] = scipy.io.loadmat(
        precalculated_inputs + 'calibratedUtility_q0.mat'
        )["calibratedUtility_q0"].squeeze()
    #  Composite good elasticity
    param["alpha"] = 1 - param["beta"]

    # Housing production function parameters, as calibrated in Pfeiffer et al.
    # (table C7)
    #  Capital elasticity
    param["coeff_b"] = scipy.io.loadmat(
        precalculated_inputs + 'calibratedHousing_b.mat')["coeff_b"].squeeze()
    # Land elasticity
    param["coeff_a"] = 1 - param["coeff_b"]
    #  Scale parameter
    param["coeff_A"] = scipy.io.loadmat(
        precalculated_inputs + 'calibratedHousing_kappa.mat'
        )["coeffKappa"].squeeze()

    # Gravity parameter of the minimum Gumbel distribution (see Pfeiffer et
    # al.), as calibrated in appendix C3
    param["lambda"] = scipy.io.loadmat(precalculated_inputs + 'lambda.mat'
                                       )["lambdaKeep"].squeeze()

    # Discount factors
    param["depreciation_rate"] = 0.025
    param["interest_rate"] = 0.025

    # Housing parameters
    #  Size of an informal dwelling unit (m^2)
    param["shack_size"] = 14
    #  Size of a social housing dwelling unit (m^2), see table C6
    param["RDP_size"] = 40
    #  Size of backyard dwelling unit (m^2), see table C6
    param["backyard_size"] = 70
    #  Nb of social housing units built per year
    param["future_rate_public_housing"] = 1000
    #  Cost of inputs for building an informal dwelling unit (in rands)
    param["informal_structure_value"] = 4000
    #  Fraction of the composite good that is kept inside the house and that
    #  can possibly be destroyed by floods (food, furniture, etc.)
    param["fraction_z_dwellings"] = 0.49
    #  Value of a social housing dwelling unit (in rands)
    param["subsidized_structure_value"] = 150000

    # Max % of land that can be built for housing (to take roads into account),
    # by housing type
    param["max_land_use"] = 0.7
    param["max_land_use_backyard"] = 0.45
    param["max_land_use_settlement"] = 0.4

    # Constraints on housing supply (in meters?), set high height limit to make
    # as if no constraints
    param["historic_radius"] = 100
    param["limit_height_center"] = 10
    param["limit_height_out"] = 10

    # Agricultural land rents (in rands)
    param["agricultural_rent_2011"] = 807.2
    param["agricultural_rent_2001"] = 70.7

    # Year urban edge constraint kicks in
    # (only useful if options["urban_edge"] == 1)
    param["year_urban_edge"] = 2015

    # Labor parameters
    #  Nb of income classes set to 4 as in Pfeiffer et al. (see table A1)
    param["nb_of_income_classes"] = 4
    #  Equivalence between income classes in the data (12) and in the model (4)
    param["income_distribution"] = np.array(
        [0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4])
    #  Nb of jobs above which we keep employment center in the analysis
    param["threshold_jobs"] = 20000
    #  Avg nb of employed workers per household of each income class
    #  (see table B1)
    param["household_size"] = [1.14, 1.94, 1.92, 1.94]

    # Transportation cost parameters
    param["waiting_time_metro"] = 10  # in minutes
    param["walking_speed"] = 4  # in km/h
    param["time_cost"] = 1  # in which unit?

    # Used in eqcmp.compute_equilibrium: iteration stops when the error in the
    # computed nb of households per income bracket falls below some precision
    # level, or when the nb of iterations reaches some threshold (to limit
    # processing time)
    param["max_iter"] = 5000
    param["precision"] = 0.01

    # Dynamic factors
    #  Time (in years?) for building a new housing unit
    param["time_invest_housing"] = 3
    #  Time (in years?) for the full depreciation of a housing unit
    param["time_depreciation_buildings"] = 100
    #  Set the nb of simulations per year
    param["iter_calc_lite"] = 1

    # Size (in m^2) above which we need to switch flood damage functions for
    # formal housing
    param["threshold"] = 130

    # Make copies of parameters that may change (why not create new variables?)
    param["informal_structure_value_ref"] = copy.deepcopy(
        param["informal_structure_value"])
    param["subsidized_structure_value_ref"] = copy.deepcopy(
        param["subsidized_structure_value"])

    return param


def import_construction_parameters(param, grid, housing_types, dwelling_size_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land):
    
    param["housing_in"] = np.empty(len(grid_formal_density_HFA))
    param["housing_in"][:] = np.nan
    param["housing_in"][coeff_land[0,:] != 0] = grid_formal_density_HFA[coeff_land[0,:] != 0] / coeff_land[0,:][coeff_land[0,:] != 0] * 1.1
    param["housing_in"][(coeff_land[0,:] == 0) | np.isnan(grid_formal_density_HFA)] = 0
    
    param["housing_in"][param["housing_in"] > 2 * (10**6)] = 2 * (10**6)
    param["housing_in"][param["housing_in"] < 0] = 0
    
    #In Mitchells Plain, housing supply is given exogenously (planning), and household of group 2 live there (Coloured neighborhood). 
    param["minimum_housing_supply"] = np.zeros(len(grid.dist))
    param["minimum_housing_supply"][mitchells_plain_grid_2011] = mitchells_plain_grid_2011[mitchells_plain_grid_2011] / coeff_land[0, mitchells_plain_grid_2011]
    param["minimum_housing_supply"][(coeff_land[0,:] < 0.1) | (np.isnan(param["minimum_housing_supply"]))] = 0
    param["multi_proba_group"] = np.empty((param["nb_of_income_classes"], len(grid.dist)))
    param["multi_proba_group"][:] = np.nan
    
    #Define minimum lot-size 
    param["mini_lot_size"] = np.nanmin(dwelling_size_sp[housing_types.total_dwellings_SP_2011 != 0][(housing_types.informal_SP_2011[housing_types.total_dwellings_SP_2011 != 0] + housing_types.backyard_SP_2011[housing_types.total_dwellings_SP_2011 != 0]) / housing_types.total_dwellings_SP_2011[housing_types.total_dwellings_SP_2011 != 0] < 0.1])

    return param