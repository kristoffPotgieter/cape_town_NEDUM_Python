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
    # Useful for coding green belt
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


def import_param(path_precalc_inp, path_outputs):
    """Import default parameters."""
    # Define baseline year
    param = {"baseline_year": 2011}

    # Utility function parameters, as calibrated in Pfeiffer et al. (table C7)
    #  Surplus housing elasticity
    param["beta"] = scipy.io.loadmat(
        path_precalc_inp + 'calibratedUtility_beta.mat'
        )["calibratedUtility_beta"].squeeze()
    #  Basic need in housing
    #  TODO: not the same a in paper
    param["q0"] = scipy.io.loadmat(
        path_precalc_inp + 'calibratedUtility_q0.mat'
        )["calibratedUtility_q0"].squeeze()
    #  Composite good elasticity
    param["alpha"] = 1 - param["beta"]

    # Housing production function parameters, as calibrated in Pfeiffer et al.
    # (table C7)
    #  Capital elasticity
    # param["coeff_b"] = scipy.io.loadmat(
    #     path_precalc_inp + 'calibratedHousing_b.mat')["coeff_b"].squeeze()
    param["coeff_b"] = np.load(path_precalc_inp + 'calibratedHousing_b.npy')
    # Land elasticity
    param["coeff_a"] = 1 - param["coeff_b"]
    #  Scale parameter
    # param["coeff_A"] = scipy.io.loadmat(
    #     path_precalc_inp + 'calibratedHousing_kappa.mat'
    #     )["coeffKappa"].squeeze()
    param["coeff_A"] = np.load(
        path_precalc_inp + 'calibratedHousing_kappa.npy')

    # Gravity parameter of the minimum Gumbel distribution (see Pfeiffer et
    # al.), as calibrated in appendix C3
    # param["lambda"] = scipy.io.loadmat(path_precalc_inp + 'lambda.mat'
    #                                    )["lambdaKeep"].squeeze()
    param["lambda"] = np.load(path_precalc_inp + 'lambdaKeep.npy')

    # Discount factors
    #  From Viguié et al. (2014)
    param["depreciation_rate"] = 0.025
    #  From World Development Indicator database (World Bank, 2016)
    #  TODO: check consistency with interest_rate from macro_data
    param["interest_rate"] = 0.025

    # Housing parameters
    #  Size of an informal dwelling unit (m^2), (not 20 as in the paper,
    #  cf. Claus)
    param["shack_size"] = 14
    #  Size of a social housing dwelling unit (m^2), see table C6
    param["RDP_size"] = 40
    #  Size of backyard dwelling unit (m^2), see table C6 (not rented fraction)
    #  NB: in theory, a backyard can therefore host up to 5 households
    param["backyard_size"] = 70
    #  Nb of social housing units built per year (cf. Housing Pipeline)
    #  Doesn't correspond to scenario from WP (cf. Claus for post-2020)
    param["future_rate_public_housing"] = 1000
    #  Cost of inputs for building an informal dwelling unit (in rands)
    #  Not zero as in the paper to account for potential destructions from
    #  floods
    param["informal_structure_value"] = 4000
    #  Fraction of the composite good that is kept inside the house and that
    #  can possibly be destroyed by floods (food, furniture, etc.)
    #  Correspond to average share of durable and semi-durable goods in total
    #  HH budget (excluding rent) : see Aux data/HH Income per DU - CL
    param["fraction_z_dwellings"] = 0.53
    #  Value of a social housing dwelling unit (in rands)
    #  For floods destruction
    #  TODO: How to determine it?
    param["subsidized_structure_value"] = 150000

    # Max % of land that can be built for housing (to take roads into account),
    # by housing type: comes from analogy with Viguié et al., 2014 (table B1)
    param["max_land_use"] = 0.7
    param["max_land_use_backyard"] = 0.45
    param["max_land_use_settlement"] = 0.4

    # Constraints on housing supply (in meters)
    #  A priori not binding
    param["historic_radius"] = 100
    param["limit_height_center"] = 10
    param["limit_height_out"] = 10

    # Agricultural land rents (in rands)
    #  Corresponds to the ninth decile in the sales data sets, when
    #  selecting only agricultural properties in rural areas
    param["agricultural_rent_2011"] = 807.2
    #  Estimated the same way
    param["agricultural_rent_2001"] = 70.7

    # Year urban edge constraint kicks in
    # (only useful if options["urban_edge"] == 1)
    param["year_urban_edge"] = 2015

    # Labor parameters
    #  Nb of income classes set to 4 as in Pfeiffer et al. (see table A1)
    param["nb_of_income_classes"] = 4
    #  Equivalence between income classes in the data (12) and in the model
    #  (4). Note that we exclude people earning no income from the analysis
    param["income_distribution"] = np.array(
        [0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4])
    #  Nb of jobs above which we keep employment center in the analysis
    param["threshold_jobs"] = 20000
    #  Avg nb of employed workers per household of each income class
    #  (see appendix B1: 2*ksi)
    param["household_size"] = [1.14, 1.94, 1.92, 1.94]
    #  Makes sense as we have data on household (not individual) income
    param["household_size"] = [size / 2 for size in param["household_size"]]

    # Transportation cost parameters
    param["waiting_time_metro"] = 10  # in minutes
    param["walking_speed"] = 4  # in km/h
    param["time_cost"] = 1  # equivalence in monetary terms

    # Used in eqcmp.compute_equilibrium: iteration stops when the error in the
    # computed nb of households per income bracket falls below some precision
    # level, or when the nb of iterations reaches some threshold (to limit
    # processing time)
    param["max_iter"] = 5000
    param["precision"] = 0.02

    # Dynamic parameters (from Viguié et al., 2014)
    #  Lag in housing building
    param["time_invest_housing"] = 3
    #  Time (in years) for the full depreciation of a housing unit
    param["time_depreciation_buildings"] = 100
    #  Set the nb of simulations per year
    param["iter_calc_lite"] = 1

    # Size (in m^2) above which we need to switch flood damage functions for
    # formal housing
    # TODO: Where from?
    param["threshold"] = 130

    # Make copies of parameters that may change (to be used in simulations)
    param["informal_structure_value_ref"] = copy.deepcopy(
        param["informal_structure_value"])
    param["subsidized_structure_value_ref"] = copy.deepcopy(
        param["subsidized_structure_value"])

    # Disamenity parameters for informal settlements and backyard shacks,
    # coming from location-based calibration, as opposed to general calibration
    # used in Pfeiffer et al. (appendix C5)
    disamenity_param = scipy.io.loadmat(
        path_precalc_inp + 'calibratedParamAmenities.mat'
        )["calibratedParamAmenities"].squeeze()
    param["pockets"] = np.matlib.repmat(
        disamenity_param[1], 24014, 1).squeeze()
    param["backyard_pockets"] = np.matlib.repmat(
        disamenity_param[0], 24014, 1).squeeze()
    # param["pockets"] = np.load(
    #     path_precalc_inp + 'param_pockets.npy')
    # param["backyard_pockets"] = np.load(
    #     path_precalc_inp + 'param_backyards.npy')

    return param


def import_construction_parameters(param, grid, housing_types_sp,
                                   dwelling_size_sp, mitchells_plain_grid_2011,
                                   grid_formal_density_HFA, coeff_land,
                                   interest_rate):
    """Update parameters with values for construction."""
    # We define housing supply per unit of land for simulations where
    # developers do not adjust
    param["housing_in"] = np.empty(len(grid_formal_density_HFA))
    param["housing_in"][:] = np.nan
    # Fill vector with population density in formal housing divided by the
    # share of built formal area (times 1.1) for areas with some formal housing
    # This gives the density for formal area instead of just the pixel area:
    # we therefore consider that developers always provide one surface unit of
    # housing per household per unit of land. In practice, this is not used.
    # NB: HFA = habitable floor area
    cond = coeff_land[0, :] != 0
    param["housing_in"][cond] = (
        grid_formal_density_HFA[cond] / coeff_land[0, :][cond] * 1.1)
    param["housing_in"][~np.isfinite(param["housing_in"])] = 0
    # Deal with formally non-built or non-inhabited areas
    # param["housing_in"][
    #     (coeff_land[0, :] == 0) | np.isnan(param["housing_in"])
    #     ] = 0
    # Put a cap and a floor on values
    param["housing_in"][param["housing_in"] > 2 * (10**6)] = 2 * (10**6)
    param["housing_in"][param["housing_in"] < 0] = 0

    # In Mitchells Plain, housing supply is given exogenously (planning),
    # and only households of group 2 live there (coloured neighborhood).
    # We do the same as before with a starting supply of 1 in Mitchells Plain:
    # the idea is to have a min housing supply in this zone whose density might
    # be underestimated by the model
    # TODO: check error in pre-treatment when importing density from the area

    param["minimum_housing_supply"] = np.zeros(len(grid.dist))
    param["minimum_housing_supply"][mitchells_plain_grid_2011] = (
        (grid_formal_density_HFA[mitchells_plain_grid_2011]
         / coeff_land[0, :][mitchells_plain_grid_2011]))
    # cond = mitchells_plain_grid_2011 * coeff_land[0, :]
    # (param["minimum_housing_supply"][cond != 0]
    #  ) = (mitchells_plain_grid_2011[cond != 0] / coeff_land[0, :][cond != 0])
    param["minimum_housing_supply"][
        (coeff_land[0, :] < 0.1) | (np.isnan(param["minimum_housing_supply"]))
        ] = 0
    minimum_housing_supply = param["minimum_housing_supply"]

    # We take minimum dwelling size of built areas where the share of informal
    # and backyard is smaller than 10% of the overall number of dwellings
    # See WP, p.18 (formal neighborhoods)
    param["mini_lot_size"] = np.nanmin(
        dwelling_size_sp[housing_types_sp.total_dwellings_SP_2011 != 0][
            (housing_types_sp.informal_SP_2011[
                housing_types_sp.total_dwellings_SP_2011 != 0]
                + housing_types_sp.backyard_SP_2011[
                    housing_types_sp.total_dwellings_SP_2011 != 0])
            / housing_types_sp.total_dwellings_SP_2011[
                housing_types_sp.total_dwellings_SP_2011 != 0]
            < 0.1
            ]
        )

    # Comes from zero profit condition: allows to convert land prices into
    # housing prices (cf. also inversion from footnote 16)
    # TODO: use interest_rate or param["interest_rate"]?
    agricultural_rent = (
        param["agricultural_rent_2011"] ** (param["coeff_a"])
        * (param["depreciation_rate"] + param["interest_rate"])
        / (param["coeff_A"] * param["coeff_b"] ** param["coeff_b"]
           * param["coeff_a"] ** param["coeff_a"])
        )

    return param, minimum_housing_supply, agricultural_rent
