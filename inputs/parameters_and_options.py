# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:50:59 2020.

@author: Charlotte Liotta
"""

import numpy as np
# import pandas as pd
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
    # Dummy for loading pre-calibrated (from Basile) parameters, as opposed
    # to newly calibrated paramaters
    options["load_precal_param"] = 1
    # Dummy for fitting informal housing disamenity parameter to grid pixels
    options["location_based_calib"] = 0

    return options


def import_param(path_precalc_inp, path_outputs, path_folder, options):
    """Import default parameters."""
    # Define baseline year
    param = {"baseline_year": 2011}

    # Utility function parameters, as calibrated in Pfeiffer et al. (table C7)
    #  Surplus housing elasticity
    if options["load_precal_param"] == 1:
        param["beta"] = scipy.io.loadmat(
            path_precalc_inp + 'calibratedUtility_beta.mat'
            )["calibratedUtility_beta"].squeeze()
    elif options["load_precal_param"] == 0:
        param["beta"] = np.load(
            path_precalc_inp + 'calibratedUtility_beta.npy')
    #  Basic need in housing
    if options["load_precal_param"] == 1:
        param["q0"] = scipy.io.loadmat(
            path_precalc_inp + 'calibratedUtility_q0.mat'
            )["calibratedUtility_q0"].squeeze()
    elif options["load_precal_param"] == 0:
        param["q0"] = np.load(path_precalc_inp + 'calibratedUtility_q0.npy')
    #  Composite good elasticity
    param["alpha"] = 1 - param["beta"]

    # Housing production function parameters, as calibrated in Pfeiffer et al.
    # (table C7)
    #  Capital elasticity
    #  TODO: is it too small compared to existing literature?
    if options["load_precal_param"] == 1:
        param["coeff_b"] = scipy.io.loadmat(
            path_precalc_inp + 'calibratedHousing_b.mat')["coeff_b"].squeeze()
    elif options["load_precal_param"] == 0:
        param["coeff_b"] = np.load(
            path_precalc_inp + 'calibratedHousing_b.npy')
    # Land elasticity
    param["coeff_a"] = 1 - param["coeff_b"]
    #  Scale parameter
    if options["load_precal_param"] == 1:
        param["coeff_A"] = scipy.io.loadmat(
            path_precalc_inp + 'calibratedHousing_kappa.mat'
            )["coeffKappa"].squeeze()
    elif options["load_precal_param"] == 0:
        param["coeff_A"] = np.load(
            path_precalc_inp + 'calibratedHousing_kappa.npy')

    # Gravity parameter of the minimum Gumbel distribution (see Pfeiffer et
    # al.), as calibrated in appendix C3
    # TODO: correct typo in paper
    if options["load_precal_param"] == 1:
        param["lambda"] = scipy.io.loadmat(path_precalc_inp + 'lambda.mat'
                                           )["lambdaKeep"].squeeze()
    elif options["load_precal_param"] == 0:
        param["lambda"] = np.load(path_precalc_inp + 'lambdaKeep.npy')

    # Threshold above which we retain TAZ as a job center (for calibration)
    param["job_center_threshold"] = 2500

    # Discount factors
    # TODO: correct typo in paper
    #  From Viguié et al. (2014)
    param["depreciation_rate"] = 0.025
    #  From World Development Indicator database (World Bank, 2016)
    #  TODO: Note that this will not be used in practice as we will prefer
    #  interpolation from historical interest rates (need to create option to
    #  change formulas otherwise)
    param["interest_rate"] = 0.025

    # Housing parameters
    #  Size of an informal dwelling unit (m^2)
    #  TODO: correct value in paper
    param["shack_size"] = 14
    #  Size of a social housing dwelling unit (m^2), see table C6
    #  TODO: does this need to be updated given research note?
    param["RDP_size"] = 40
    #  Size of backyard dwelling unit (m^2), see table C6 (not rented fraction)
    #  NB: in theory, a backyard can therefore host up to 5 households
    param["backyard_size"] = 70
    #  Nb of social housing units built per year (cf. Housing Pipeline)
    #  TODO :correct value in paper
    param["future_rate_public_housing"] = 1000
    #  Cost of inputs for building an informal dwelling unit (in rands)
    #  This is used to account for potential destructions from floods
    #  TODO: add some references from Claus (why not 3000 as in research note?)
    #  TODO: plug flow costs as a fraction of land used to recover values?
    # param["informal_structure_value"] = 4000
    param["informal_structure_value"] = 3000
    #  Fraction of the composite good that is kept inside the house and that
    #  can possibly be destroyed by floods (food, furniture, etc.)
    #  Correspond to average share of such goods in total HH budget
    #  (excluding rent): comes from Quantec data
    #  Import from original data
    # quantec_data = pd.read_excel(
    #     path_folder + 'Aux data/HH Income per DU - CL.xlsx',
    #     sheet_name='Quantec HH Budgets', header=6)
    # param["fraction_z_dwellings"] = quantec_data.iloc[77, 11]
    #  Equivalent raw number import
    #  TODO: discuss the need to adjust non-durable goods taken into account
    #  in the budget, as in research note for informal settlements
    param["fraction_z_dwellings"] = 0.53
    #  Value of a social housing dwelling unit (in rands): again needed for
    #  flood damage estimation
    #  TODO: include references from Claus
    #  Old estimate for reference
    # param["subsidized_structure_value"] = 150000
    #  New estimate
    param["subsidized_structure_value"] = 127000

    # Max % of land that can be built for housing (to take roads into account),
    # by housing type: comes from analogy with Viguié et al., 2014 (table B1)
    # More precisely, maximum fraction of ground surface devoted to housing
    # where building is possible is 62% in Paris
    # TODO: discuss alternative specifications
    param["max_land_use"] = 0.7
    param["max_land_use_backyard"] = 0.45
    param["max_land_use_settlement"] = 0.4

    # Constraints on housing supply (in meters), a priori not binding
    # TODO: include references from Claus
    param["historic_radius"] = 6
    # Note that we allow for higher construction in the core center, compared
    # to previous version
    param["limit_height_center"] = 80
    param["limit_height_out"] = 10

    # Agricultural land rents (in rands)
    #  Corresponds to the ninth decile in the sales data sets, when
    #  selecting only agricultural properties in rural areas
    #  NB: corresponds to the price of land, not of real estate
    #  TODO: recover original data
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
    #  Avg nb of employed workers per household of each income class
    #  (see appendix B1: 2*ksi): we need to take into account monetary cost
    #  for both members of the household (cf. import_transport_data)
    #  NB: looks like employment vs. unemployment rate (no participation rate)
    #  TODO: recover original data
    param["household_size"] = [1.14, 1.94, 1.92, 1.94]

    # Transportation cost parameters
    # TODO: where does waiting time kick in?
    # param["waiting_time_metro"] = 10  # in minutes
    param["walking_speed"] = 4  # in km/h
    param["time_cost"] = 1  # equivalence in monetary terms
    # NB: this parameter is estimated in QSE models such as Tsivanidis'

    # Used in eqcmp.compute_equilibrium: iteration stops when the error in the
    # computed nb of households per income bracket falls below some precision
    # level, or when the nb of iterations reaches some threshold (to limit
    # processing time)
    param["max_iter"] = 5000
    param["precision"] = 0.01

    # Dynamic parameters (from Viguié et al., 2014, table B1)
    #  Lag in housing building
    #  NB: this parameter varies through time in models such as Gomtsyan's
    param["time_invest_housing"] = 3
    #  Time (in years) for the full depreciation of a housing unit
    param["time_depreciation_buildings"] = 100
    #  Set the nb of simulations per year
    param["iter_calc_lite"] = 1

    # Size (in m^2) above which we need to switch flood damage functions for
    # formal housing: corresponds to existence of a 2nd floor
    # TODO: Where from?
    param["threshold"] = 130

    # Make copies of parameters that may change (to be used in simulations)
    param["informal_structure_value_ref"] = copy.deepcopy(
        param["informal_structure_value"])
    param["subsidized_structure_value_ref"] = copy.deepcopy(
        param["subsidized_structure_value"])

    # Disamenity parameters for informal settlements and backyard shacks

    if (options["location_based_calib"] == 0
       and options["load_precal_param"] == 1):
        disamenity_param = scipy.io.loadmat(
            path_precalc_inp + 'calibratedParamAmenities.mat'
            )["calibratedParamAmenities"].squeeze()
        param["pockets"] = np.matlib.repmat(
            disamenity_param[1], 24014, 1).squeeze()
        param["backyard_pockets"] = np.matlib.repmat(
            disamenity_param[0], 24014, 1).squeeze()

    elif (options["location_based_calib"] == 0
          and options["load_precal_param"] == 0):
        param_amenity_settlement = np.load(
            path_precalc_inp + 'param_amenity_settlement.npy')
        param["pockets"] = np.matlib.repmat(
            param_amenity_settlement, 24014, 1).squeeze()
        param_amenity_backyard = np.load(
            path_precalc_inp + 'param_amenity_backyard.npy')
        param["backyard_pockets"] = np.matlib.repmat(
            param_amenity_backyard, 24014, 1).squeeze()

    elif options["location_based_calib"] == 1:
        param["pockets"] = np.load(
            path_precalc_inp + 'param_pockets.npy')
        param["backyard_pockets"] = np.load(
            path_precalc_inp + 'param_backyards.npy')

    return param


def import_construction_parameters(param, grid, housing_types_sp,
                                   dwelling_size_sp, mitchells_plain_grid_2011,
                                   grid_formal_density_HFA, coeff_land,
                                   interest_rate, options):
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
    # TODO: why 1.1?
    param["housing_in"][cond] = (
        grid_formal_density_HFA[cond] / coeff_land[0, :][cond] * 1.1)
    param["housing_in"][~np.isfinite(param["housing_in"])] = 0
    # Deal with formally non-built or non-inhabited areas
    param["housing_in"][
        (coeff_land[0, :] == 0) | np.isnan(param["housing_in"])
        ] = 0
    # Put a cap and a floor on values
    param["housing_in"][param["housing_in"] > 2 * (10**6)] = 2 * (10**6)
    param["housing_in"][param["housing_in"] < 0] = 0

    # In Mitchells Plain, housing supply is given exogenously (planning),
    # and only households of group 2 live there (coloured neighborhood).
    # We do the same as before: the idea is to have a min housing supply in
    # this zone whose formal density might be underestimated by the model

    if options["correct_mitchells_plain"] == 0:
        param["minimum_housing_supply"] = np.zeros(len(grid.dist))
    elif options["correct_mitchells_plain"] == 1:
        # Original specification from Matlab
        param["minimum_housing_supply"][mitchells_plain_grid_2011] = (
            (grid_formal_density_HFA[mitchells_plain_grid_2011]
             / coeff_land[0, :][mitchells_plain_grid_2011]))
        # Alternative specification accounting for minimum dwelling size
        # param["minimum_housing_supply"][mitchells_plain_grid_2011] = (
        #     (grid_formal_density_HFA[mitchells_plain_grid_2011] * param["q0"]
        #      / coeff_land[0, :][mitchells_plain_grid_2011]))
        param["minimum_housing_supply"][
            (coeff_land[0, :] < 0.1)
            | (np.isnan(param["minimum_housing_supply"]))
            ] = 0

    # TODO: Discuss better correction than such ad hoc procedure (although
    # for Philippi and Khayelitsha)

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

    agricultural_rent = compute_agricultural_rent(
        param["agricultural_rent_2011"], param["coeff_A"], interest_rate,
        param, options
        )

    return param, minimum_housing_supply, agricultural_rent


def compute_agricultural_rent(rent, scale_fact, interest_rate, param, options):
    """Convert land price into real estate price for land."""
    if options["correct_agri_rent"] == 1:
        agricultural_rent = (
            rent ** (param["coeff_a"])
            * (param["depreciation_rate"] + interest_rate)
            / (scale_fact * param["coeff_b"] ** param["coeff_b"]
                * param["coeff_a"] ** param["coeff_a"])
            )
    elif options["correct_agri_rent"] == 0:
        agricultural_rent = (
            rent ** (param["coeff_a"])
            * (param["depreciation_rate"] + interest_rate)
            / (scale_fact * param["coeff_b"] ** param["coeff_b"])
            )

    return agricultural_rent
