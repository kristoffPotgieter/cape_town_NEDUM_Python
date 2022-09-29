# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import copy


def import_options():
    """
    Import default options.

    Import set of numerical values coding for options used in the model.
    We can group them as follows: structural assumptions regarding agents'
    behaviour, assumptions about different land uses, options about flood data
    used, about re-processing input data, about calibration process, about
    math correction relative to the original code, and about scenarios used
    for time-moving exogenous variables.

    Returns
    -------
    options : dict
        Dictionary of default options

    """
    # STRUCTURAL ASSUMPTIONS
    # Dummy for formal private developers adjusting housing supply to demand
    options = {"adjust_housing_supply": 1}
    # Dummy for agents taking floods into account in their choices
    options["agents_anticipate_floods"] = 1

    # LAND USE ASSUMPTIONS
    # Dummy for coding green belt
    options["urban_edge"] = 0
    # Dummy for forbidding new informal housing construction
    options["informal_land_constrained"] = 0

    # FLOOD DATA OPTIONS
    # Dummy for using flood data from WBUS2 on top of FATHOM + DELTARES data
    # (deprecated)
    options["WBUS2"] = 0
    # Dummy for considering pluvial floods on top of fluvial floods
    options["pluvial"] = 1
    # Dummy for reducing pluvial risk for (better protected) formal structures
    options["correct_pluvial"] = 1
    # Dummy for working with defended (vs. undefended) fluvial flood maps
    options["defended"] = 0
    # Dummy for taking coastal floods into account (on top of fluvial floods)
    options["coastal"] = 1
    # Digital elevation model to be used with coastal flood data
    # (set to "MERITDEM" or "NASADEM")
    options["dem"] = "MERITDEM"
    # Dummy for taking sea-level rise into account in coastal flood data
    options["slr"] = 1

    # REPROCESSING OPTIONS
    # Default is set at zero to save computing time
    # (data is simply loaded in the model)
    # Convert SAL (Small-Area-Level) data (on housing types) to grid level
    options["convert_sal_data"] = 0
    # Convert SP (Small Place) data on income groups to grid level
    options["convert_sp_data"] = 0
    # Compute income net of commuting costs for each pixel in each income
    # group based upon calibrated incomes for each job center in each income
    # group (for every period)
    options["compute_net_income"] = 0

    # CALIBRATION OPTIONS
    # Dummy for loading pre-calibrated parameters (from Pfeiffer et al.),
    # as opposed to newly calibrated paramaters
    options["load_precal_param"] = 0
    # Dummy for fitting informal housing disamenity parameter to grid level
    options["location_based_calib"] = 1
    # Dummy for defining dominant income group based on number of people
    # instead of median income at SP level
    options["correct_dominant_incgrp"] = 0
    # Dummy for substracting subsidized housing units to count of formal units
    # at SP level
    options["substract_RDP_from_formal"] = 0
    # Dummy for leaving Mitchells Plain out in calibration and to do ad hoc
    # correction of housing supply in the area
    options["correct_mitchells_plain"] = 0
    # Dummy for including more criteria in sample selection for calibration
    # (compared to Pfeiffer et al.)
    options["correct_selected_density"] = 1
    # Dummy for correcting the formula for the construction function scale
    # factor (compared to original code)
    options["correct_kappa"] = 1
    # Dummy for setting inflation base year at baseline year (2011)
    # (instead of 2012 in original code)
    options["correct_infla_base"] = 1
    # Dummy for taking round trips into account in estimated monetary
    # transport costs (compared to original code)
    options["correct_round_trip"] = 1
    # Dummy for taking unemployment into account in the formula for
    # the number of commuters (deprecated)
    options["correct_eq3"] = 0
    # Resolution of the scanning to be used in calibration
    # (for commuting gravity and utility function parameters)
    # NB: should be set to "rough", "normal", or "fine"
    options["scan_type"] = "fine"
    # Dummy for reversing calibrated capital and land elasticities in
    # construction cost function (deprecated)
    # NB: estimate is not in line with the literature and we might want to
    # check how this affects the model
    options["reverse_elasticities"] = 0
    # Dummy for using GLM (instead of OLS) for the estimation of exogenous
    # amenity parameters (deprecated)
    options["glm"] = 0
    # Dummy for using RBFInterpolator instead of interp2d for 2D interpolation
    # of rents based on incomes and utilities (deprecated)
    options["griddata"] = 0
    # Number of neighbours to be used if RBFInterpolator is chosen (deprecated)
    options["interpol_neighbors"] = 50
    # Dummy for improving rent interpolation in calibration by assuming away
    # basic need in housing from maximum rent estimation (deprecated)
    options["test_maxrent"] = 0
    # Dummy for using scipy solver to refine utility function parameter
    # estimates from scanning
    options["param_optim"] = 1
    # Dummy for taking log form into account in rent interpolation for utility
    # function parameter estimation (helps convergence)
    options["log_form"] = 1

    # CODE CORRECTION OPTIONS
    # Dummy for taking formal backyards into account in backyard land use
    # coefficients and structural damages from floods (deprecated)
    options["actual_backyards"] = 0
    # Dummy for allocating no-income population to each income group based on
    # their respective unemployment rates, instead of applying a unique rate
    # (deprecated)
    options["unempl_reweight"] = 0
    # Dummy for correcting the formula for agricultural rent
    # (compared to original version of the code)
    options["correct_agri_rent"] = 1
    # Dummy for taking into account capital depreciation as a factor of
    # land price in profit function (has an impact on agricultural rent)
    options["deprec_land"] = 0

    # SCENARIO OPTIONS
    #  Code corresponds to low/medium/high
    options["inc_ineq_scenario"] = 2
    #  Code corresponds to low/medium/high/high_corrected
    options["pop_growth_scenario"] = 4
    #  NB: we do not add and option for interest rate: expected future value
    #  can just be plugged direcly into the scenario table.
    #  Same goes for inflation.
    #  However, price of fuel should be defined independently to be of interest
    #  We define dummy scenarios for the time being...
    #  Code corresponds to low/medium/high
    options["fuel_price_scenario"] = 2

    return options


def import_param(path_precalc_inp, options):
    """
    Import default parameters.

    Import set of numerical parameters used in the model.
    Some parameters are the output of a calibration process: it is the case
    of construction function parameters, incomes and associated gravity
    parameter, utility function parameters, and disamenity index for informal
    housing. Some other parameters are just defined ad hoc, based on existing
    empirical evidence.

    Parameters
    ----------
    path_precalc_inp : str
        Path for precalcuted input data (calibrated parameters)
    options : dict
        Dictionary of default options

    Returns
    -------
    param : dict
        Dictionary of default parameters

    """
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
    #  NB: we take this value as given by Pfeiffer et al., since it is not
    #  possible to run a stable optimization along with utility levels and
    #  utility function parameters (see calibration.calib_main_func)
    param["q0"] = scipy.io.loadmat(
        path_precalc_inp + 'calibratedUtility_q0.mat'
        )["calibratedUtility_q0"].squeeze()
    #  Composite good elasticity
    param["alpha"] = 1 - param["beta"]

    # Housing production function parameters, as calibrated in Pfeiffer et al.
    # (table C7)
    #  Capital elasticity
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
    # al.), as calibrated in appendix C3 (typo in original paper)
    if options["load_precal_param"] == 1:
        param["lambda"] = scipy.io.loadmat(path_precalc_inp + 'lambda.mat'
                                           )["lambdaKeep"].squeeze()
    elif options["load_precal_param"] == 0:
        param["lambda"] = np.load(path_precalc_inp + 'lambdaKeep.npy')

    # Threshold above which we retain transport zone (TAZ) as a job center
    # (for calibration)
    param["job_center_threshold"] = 2500

    # Discount factors (typo in original paper)
    #  From Viguié et al. (2014)
    param["depreciation_rate"] = 0.025

    #  From World Development Indicator database (World Bank, 2016)
    #  NB: Note that this will not be used in practice as we will prefer
    #  interpolation from historical interest rates
    # param["interest_rate"] = 0.025

    # Housing parameters
    #  Size of an informal dwelling unit in m² (wrong value in Pfeiffer et al.)
    param["shack_size"] = 14
    #  Size of a social housing dwelling unit (m²),
    #  see table C6 (Pfeiffer et al.)
    param["RDP_size"] = 40
    #  Size of backyards in RDP (m²), see table C6 (Pfeiffer et al.)
    #  NB: in theory, a backyard can therefore host up to 5 households
    #  TODO: does this need to be updated given Claus' research note?
    param["backyard_size"] = 70
    #  Number of formal subsidized housing units built per year
    #  (cf. Housing Pipeline from CoCT used in Pfeiffer et al.)
    param["future_rate_public_housing"] = 1000
    #  Cost of inputs for building an informal dwelling unit (in rands)
    #  This is used to account for potential destructions from floods
    #  TODO: ask Claus for reference
    param["informal_structure_value"] = 3000

    #  Fraction of the composite good that is kept inside the house and that
    #  can possibly be destroyed by floods (food, furniture, etc.)
    #  Correspond to average share of such goods in total household budget
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

    #  Value of a formal subsidized housing dwelling unit (in rands):
    #  again needed for flood damage estimation
    param["subsidized_structure_value"] = 127000

    # Max % of land that can be built for housing (to take roads into account),
    # by housing type: comes from analogy with Viguié et al., 2014 (table B1).
    # More precisely, maximum fraction of ground surface devoted to housing
    # where building is possible is 62% in Paris
    # TODO: discuss alternative specifications
    param["max_land_use"] = 0.7
    param["max_land_use_backyard"] = 0.45
    param["max_land_use_settlement"] = 0.4

    # Constraints on housing supply (in meters): a priori not binding
    param["historic_radius"] = 6
    # Note that we allow for higher construction in the core center, compared
    # to previous version
    param["limit_height_center"] = 80
    param["limit_height_out"] = 10

    # Agricultural land prices (in rands)
    #  Corresponds to the ninth decile in the sales data sets, when
    #  selecting only agricultural properties in rural areas (Pfeiffer et al.)
    param["agricultural_price_2011"] = 807.2
    #  Estimated the same way
    param["agricultural_price_2001"] = 70.7

    # Year urban edge constraint kicks in
    param["year_urban_edge"] = 2015

    # Labor parameters
    #  Number of income classes set to 4 as in Pfeiffer et al. (see table A1)
    param["nb_of_income_classes"] = 4
    #  Equivalence between income classes in the data (12) and in the model
    #  (4), from poorest to richest. Note that we exclude people earning no
    #  income from the analysis
    param["income_distribution"] = np.array(
        [0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4])
    #  Average number of employed workers per household of each income class
    #  (see appendix B1 in Pfeiffer et al.: corresponds to 2 * \ksi):
    #  we need to take into account monetary cost for both members of the
    #  household (used in import_transport_data function below)
    #  NB: we consider population in the labour force all along
    param["household_size"] = [1.14, 1.94, 1.92, 1.94]

    # Transportation cost parameters
    param["walking_speed"] = 4  # in km/h
    param["time_cost"] = 1  # equivalence in monetary terms

    # Parameters used in equilibrium.compute_equilibrium: iteration stops when
    # the error in the computed number of households per income bracket falls
    # below some precision level, or when the number of iterations reaches some
    #  threshold (to limit processing time)
    param["max_iter"] = 2000
    param["precision"] = 0.02

    # We also allow for customisation of convergence factor for disamenity
    # parameter calibration (see calibration.calib_main_func)
    param["disamenity_cvfactor"] = 20000

    # Dynamic parameters (from Viguié et al., 2014, table B1)
    #  Lag in housing building
    #  NB: this parameter varies through time in models such as Gomtsyan's
    param["time_invest_housing"] = 3
    #  Time (in years) for the full depreciation of a housing unit (deprecated)
    #  In practice, we rather use the inverse value of capital depreciation.
    param["time_depreciation_buildings"] = 100
    #  Set the number of simulations per year (deprecated)
    param["iter_calc_lite"] = 1

    # Size (in m²) above which we need to switch flood damage functions for
    # formal housing: corresponds to existence of a 2nd floor
    # TODO: Where from?
    param["threshold"] = 130

    # Make copies of parameters that may change over time
    # (to be used in simulations)
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
    """
    Update default parameters with construction parameters.

    Import set of numerical construction-related parameters used in the model.
    They depend on pre-loaded data and are therefore imported as part of a
    separate function

    Parameters
    ----------
    param : dict
        Dictionary of default parameters
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    housing_types_sp : DataFrame
        Table yielding, for each Small Place (1,046), the number of
        informal backyards, informal settlements, and total number
        of dwelling units at baseline year (2011), as well as its
        x and y (centroid) coordinates
    dwelling_size_sp : Series
        Average dwelling size (in m²) in each Small Place (1,046)
        at baseline year (2011)
    mitchells_plain_grid_2011 : ndarray(uint8)
        Dummy coding for belonging to Mitchells Plain neighbourhood
        at the grid-cell (24,014) level
    grid_formal_density_HFA : ndarray(float64)
        Population density (per m²) in formal private housing at baseline year
        (2011) at the grid-cell (24,014) level
    coeff_land : ndarray(float64, ndim=2)
        Table yielding, for each grid cell (24,014), the percentage of land
        area available for construction in each housing type (4) respectively.
        In the order: formal private housing, informal backyards, informal
        settlements, formal subsidized housing.
    interest_rate : float64
        Interest rate for the overall economy, corresponding to an average
        over past years
    options : dict
        Dictionary of default options

    Returns
    -------
    param : dict
        Updated dictionary of default parameters
    minimum_housing_supply : ndarray(float64)
        Minimum housing supply (in m²) for each grid cell (24,014), allowing
        for an ad hoc correction of low values in Mitchells Plain
    agricultural_rent : int
        Annual housing rent below which it is not profitable for formal private
        developers to urbanize (agricultural) land: endogenously limits urban
        sprawl

    """
    # We define housing supply per unit of land for simulations where
    # formal private developers do not adjust to demand (deprecated)
    param["housing_in"] = np.empty(len(grid_formal_density_HFA))
    param["housing_in"][:] = np.nan
    # Fill vector with population density in formal housing divided by the
    # share of built formal area (times 1.1) for areas with some formal housing
    # This gives the density in formal area instead of just the pixel area:
    # we therefore consider that developers always provide one surface unit of
    # housing per household per unit of land. In practice, this is not used.
    # NB: HFA = habitable floor area
    cond = coeff_land[0, :] != 0
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
    # this zone whose formal density might be underestimated by the model.
    # Then, we may choose whether or not to apply this ad hoc correction.

    if options["correct_mitchells_plain"] == 0:
        param["minimum_housing_supply"] = np.zeros(len(grid.dist))
    elif options["correct_mitchells_plain"] == 1:
        # Original specification
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

    minimum_housing_supply = param["minimum_housing_supply"]

    # Let us define a minimum dwelling size for formal private housing.
    # We take minimum dwelling size of built areas where the share of informal
    # and backyard is smaller than 10% of the overall number of dwellings.
    # See Pfeiffer et al., section 4.2 (formal neighborhoods)
    # TODO: Might need to update variable names in raw data
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

    # We define agricultural (annual) rent to put a floor on formal private
    # housing market rents (and endogenously limit urban expansion).
    # Comes from zero profit condition for formal private developer: allows to
    # convert land prices into housing prices
    # (cf. Pfeiffer et al., footnote 16)

    agricultural_rent = compute_agricultural_rent(
        param["agricultural_price_2011"], param["coeff_A"], interest_rate,
        param, options
        )

    return param, minimum_housing_supply, agricultural_rent


def compute_agricultural_rent(rent, scale_fact, interest_rate, param, options):
    """
    Convert agricultural land price into theoretical annual housing rent.

    The conversion leverages the zero profit condition for formal private
    developers in equilibrium.

    Parameters
    ----------
    rent : float
        Parametric agricultural land price at baseline year (2011)
    scale_fact : float
        (Calibrated) scale factor for the construction function of
        formal private developers
    interest_rate : float
        Parametric interest rate at baseline year (computed as the
        average over previous years)
    param : dict
        Dictionary of default parameters
    options : dict
        Dictionary of default options

    Returns
    -------
    agricultural_rent : float64
        Theoretical agricultural (annual) rent (corresponds to opportunity
        cost of non-urbanized land)

    """
    if options["correct_agri_rent"] == 1 and options["deprec_land"] == 1:
        agricultural_rent = (
            rent ** (param["coeff_a"])
            * (param["depreciation_rate"] + interest_rate)
            / (scale_fact * param["coeff_b"] ** param["coeff_b"]
                * param["coeff_a"] ** param["coeff_a"])
            )
    elif options["correct_agri_rent"] == 0 and options["deprec_land"] == 1:
        agricultural_rent = (
            rent ** (param["coeff_a"])
            * (param["depreciation_rate"] + interest_rate)
            / (scale_fact * param["coeff_b"] ** param["coeff_b"])
            )
    elif options["correct_agri_rent"] == 1 and options["deprec_land"] == 0:
        agricultural_rent = (
            rent ** param["coeff_a"]
            * interest_rate ** param["coeff_a"]
            * (param["depreciation_rate"] + interest_rate) ** param["coeff_b"]
            / (scale_fact * param["coeff_b"] ** param["coeff_b"]
                * param["coeff_a"] ** param["coeff_a"])
            )
    elif options["correct_agri_rent"] == 0 and options["deprec_land"] == 0:
        agricultural_rent = (
            rent ** param["coeff_a"]
            * interest_rate ** param["coeff_a"]
            * (param["depreciation_rate"] + interest_rate) ** param["coeff_b"]
            / (scale_fact * param["coeff_b"] ** param["coeff_b"])
            )

    return agricultural_rent
