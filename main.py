# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:33:37 2020.

@author: Charlotte Liotta
"""

# %% Preamble

# IMPORT PACKAGES

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import datetime

import inputs.data as inpdt
import inputs.parameters_and_options as inpprm

import equilibrium.compute_equilibrium as eqcmp
import equilibrium.run_simulations as eqsim
import equilibrium.functions_dynamic as eqdyn

import calibration.calib_main_func as calmain

print("Import packages and define file paths")


# DEFINE FILE PATHS

path_code = '..'
path_folder = path_code + '/2. Data/'
path_precalc_inp = path_folder + '0. Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_folder + 'data_Cape_Town/Scenarios/'
path_outputs = path_code + '/4. Sorties/'
path_floods = path_folder + "FATHOM/"

# TODO: rethink folder architecture


# START TIMER FOR CODE OPTIMIZATION

start = time.process_time()


# %% Import parameters and options

print("Import default parameters and options, define custom ones")

# TODO: convert excel to csv

# IMPORT PARAMETERS AND OPTIONS

options = inpprm.import_options()
param = inpprm.import_param(
    path_precalc_inp, path_outputs, path_folder, options)

# Set custom options for this simulation
#  Dummy for taking floods into account in the utility function
options["agents_anticipate_floods"] = 1
#  Dummy for preventing new informal settlement development
options["informal_land_constrained"] = 0
#  TODO: add option to take into account less likely developments?


# More custom options regarding flood model
#  Dummy for taking pluvial floods into account (on top of fluvial floods)
options["pluvial"] = 1
#  Dummy for reducing pluvial risk for (better protected) formal structures
options["correct_pluvial"] = 1
#  Dummy for taking coastal floods into account (on top of fluvial floods)
options["coastal"] = 1
#  Digital elevation to be used with coastal flood data (MERITDEM or NASADEM)
#  NB: MERITDEM is also the DEM used for fluvial and pluvial flood data
options["dem"] = "MERITDEM"
#  Dummy for taking sea-level rise into account in coastal flood data
#  NB: Projections are up to 2050, based upon IPCC AR5 assessment for the
#  RCP 8.5 scenario
options["slr"] = 1

# Re-processing options: default is set at zero to save computing time (data
# is simply loaded in the model)
# NB: this is only needed to create the data for the first time, or when the
# source is changed, so that pre-processed data is updated
options["convert_sal_data"] = 0
options["compute_net_income"] = 0

# RE-RUN CALIBRATION: note that this takes time and is only useful for the
# first time or if data used for calibration changes
options["run_calib"] = 0

#  SET TIMELINE FOR SIMULATIONS
t = np.arange(0, 30)

# GIVE NAME TO SIMULATION TO EXPORT THE RESULTS
# (change according to custom parameters to be included)

name = ('floods' + str(options["agents_anticipate_floods"])
        + str(options["informal_land_constrained"]) + '_P'
        + str(options["pluvial"]) + str(options["correct_pluvial"])
        + '_C' + str(options["coastal"]) + str(options["slr"])
        + '_loc')
path_plots = path_outputs + name + '/plots/'


# %% Load data

print("Load and pre-process data to be used in model (may take some time"
      + " when agents anticipate floods and we re-process some data)")


# BASIC GEOGRAPHIC DATA

grid, center = inpdt.import_grid(path_data)
amenities = inpdt.import_amenities(path_precalc_inp, options)


# MACRO DATA

(interest_rate, population, housing_type_data, total_RDP
 ) = inpdt.import_macro_data(param, path_scenarios, path_folder)


# HOUSEHOLDS AND INCOME DATA

income_class_by_housing_type = inpdt.import_hypothesis_housing_type()

#  See appendix A1 for income group and housing type definitions
(mean_income, households_per_income_class, average_income, income_mult,
 income_2011, households_per_income_and_housing
 ) = inpdt.import_income_classes_data(param, path_data)

#  We create this parameter to maintain money illusion in simulations
#  (see eqsim.run_simulation)
#  NB: set as a variable, not a parameter?
param["income_year_reference"] = mean_income

#  Other data at SP level used for calibration and validation
(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(path_precalc_inp)

#  Import nb of households per pixel, by housing type (from SAL data).
#  Note that RDP is included in formal, and there are both formal and informal
#  backyards

if options["convert_sal_data"] == 1:
    print("Convert SAL data to grid dimensions - start")
    housing_types = inpdt.import_sal_data(grid, path_folder, path_data,
                                          housing_type_data)
    print("Convert SAL data to grid dimensions - end")

housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')
housing_types[np.isnan(housing_types)] = 0

# LAND USE PROJECTIONS

(spline_RDP, spline_estimate_RDP, spline_land_RDP,
 spline_land_backyard, spline_land_informal, spline_land_constraints,
 number_properties_RDP) = (
     inpdt.import_land_use(grid, options, param, data_rdp, housing_types,
                           housing_type_data, path_data, path_folder)
     )

#  We correct areas for each housing type at baseline year for the amount of
#  constructible land in each type
coeff_land = inpdt.import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal,
    spline_land_RDP, param, 0)

#  We update land use parameters at baseline (relies on loaded data)
housing_limit = inpdt.import_housing_limit(grid, param)

#  NB: plug outputs in a new variable (not param) and adapt linked functions?
(param, minimum_housing_supply, agricultural_rent
 ) = inpprm.import_construction_parameters(
    param, grid, housing_types_sp, data_sp["dwelling_size"],
    mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land,
    interest_rate, options
    )

# FLOOD DATA (takes some time when agents anticipate floods)
#  NB: create a new variable instead of storing in param
#  NB: WBUS2 corresponds to old data from CoCT (not useful anymore with FATHOM)
#  param = inpdt.infer_WBUS2_depth(housing_types, param, path_floods)
if options["agents_anticipate_floods"] == 1:
    print("Compute flood damages for each damage category - start")
    (fraction_capital_destroyed, structural_damages_small_houses,
     structural_damages_medium_houses, structural_damages_large_houses,
     content_damages, structural_damages_type1, structural_damages_type2,
     structural_damages_type3a, structural_damages_type3b,
     structural_damages_type4a, structural_damages_type4b
     ) = inpdt.import_full_floods_data(options, param, path_folder,
                                       housing_type_data)
    print("Compute flood damages for each damage category - end")

elif options["agents_anticipate_floods"] == 0:
    fraction_capital_destroyed = pd.DataFrame()
    fraction_capital_destroyed["structure_formal_2"] = np.zeros(24014)
    fraction_capital_destroyed["structure_formal_1"] = np.zeros(24014)
    fraction_capital_destroyed["structure_subsidized_2"] = np.zeros(24014)
    fraction_capital_destroyed["structure_subsidized_1"] = np.zeros(24014)
    fraction_capital_destroyed["contents_formal"] = np.zeros(24014)
    fraction_capital_destroyed["contents_informal"] = np.zeros(24014)
    fraction_capital_destroyed["contents_subsidized"] = np.zeros(24014)
    fraction_capital_destroyed["contents_backyard"] = np.zeros(24014)
    fraction_capital_destroyed["structure_backyards"] = np.zeros(24014)
    fraction_capital_destroyed["structure_formal_backyards"] = np.zeros(24014)
    fraction_capital_destroyed["structure_informal_backyards"
                               ] = np.zeros(24014)
    fraction_capital_destroyed["structure_informal_settlements"
                               ] = np.zeros(24014)


# SCENARIOS

(spline_agricultural_rent, spline_interest_rate,
 spline_population_income_distribution, spline_inflation,
 spline_income_distribution, spline_population,
 spline_income, spline_minimum_housing_supply, spline_fuel
 ) = eqdyn.import_scenarios(income_2011, param, grid, path_scenarios,
                            options)

#  Import income net of commuting costs, as calibrated in Pfeiffer et al.
#  (see part 3.1 or appendix C3)

if options["compute_net_income"] == 1:
    print("Compute local incomes net of commuting costs for every simulation"
          + " period - start")
    for t_temp in t:
        print(t_temp)
        (incomeNetOfCommuting, modalShares, ODflows, averageIncome
         ) = inpdt.import_transport_data(
             grid, param, t_temp, households_per_income_class, average_income,
             spline_inflation, spline_fuel,
             spline_population_income_distribution, spline_income_distribution,
             path_precalc_inp, path_precalc_transp, 'GRID', options)
        print("Compute local incomes net of commuting costs for every"
              + " simulation period - end")

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')


# %% Re-run calibration (takes time, only if needed)

# NB: use np.linspace instead of np.arange?

if options["run_calib"] == 1:

    print("Calibration process - start")

    # PREAMBLE

    if options["correct_dominant_incgrp"] == 0:
        # We associate income group to each census block according to median
        # income
        data_income_group = np.zeros(len(data_sp["income"]))
        for j in range(0, param["nb_of_income_classes"] - 1):
            data_income_group[data_sp["income"] >
                              threshold_income_distribution[j]] = j+1
    elif options["correct_dominant_incgrp"] == 1:
        # We use more numerous group instead
        data_income_group = np.zeros(len(income_distribution))
        for i in range(0, len(income_distribution)):
            data_income_group[i] = np.argmax(income_distribution[i])
    # Although the second option seems more logical, it may make sense to use
    # the first one given that we are going to regress on median SP prices

    # We get the number of formal housing units per SP
    # NB: it is not clear whether RDP are included in SP formal count, and
    # if they should be taken out based on imperfect cadastral estimations.
    # For our benchmark, we prefer to rely on sample selection.

    if options["substract_RDP_from_formal"] == 1:
        # We retrieve number of RDP units per SP from grid-level data
        grid_intersect = pd.read_csv(path_data + 'grid_SP_intersect.csv',
                                     sep=';')
        # When pixels are associated to several SPs, we allocate them to the
        # one with the biggest intersection area.
        # NB: it would be more rigorous to split the number of RDP across
        # SPs according to their respective intersection areas, but this is
        # unlikely to change much
        grid_intersect = grid_intersect.groupby('ID_grille').max('Area')
        data_rdp["ID_grille"] = data_rdp.index
        data_rdp["ID_grille"] = data_rdp["ID_grille"] + 1

        rdp_grid = pd.merge(data_rdp, grid_intersect, on="ID_grille",
                            how="outer")
        rdp_sp = rdp_grid.groupby('SP_CODE')['count'].sum()
        rdp_sp = rdp_sp.reset_index()
        rdp_sp = rdp_sp.rename(columns={'SP_CODE': 'sp_code'})
        # We just fill the list with unmatched SPs to get the full SP vector
        rdp_sp_fill = pd.merge(rdp_sp, data_sp['sp_code'], on="sp_code",
                               how="outer")
        rdp_sp_fill['count'] = rdp_sp_fill['count'].fillna(0)
        rdp_sp_fill = rdp_sp_fill.sort_values(by='sp_code')

        data_number_formal = (
            housing_types_sp.total_dwellings_SP_2011
            - housing_types_sp.backyard_SP_2011
            - housing_types_sp.informal_SP_2011
            - rdp_sp_fill['count'])

    elif options["substract_RDP_from_formal"] == 0:
        data_number_formal = (
            housing_types_sp.total_dwellings_SP_2011
            - housing_types_sp.backyard_SP_2011
            - housing_types_sp.informal_SP_2011
            )

    # Although it makes more sense to substract RDP from number of formal
    # private units, it may make sense to keep them if we are unable to select
    # SPs with few RDP units

    # Note that SP housing type data contain more households than SAL housing
    # type data, or aggregate income data (with fewer backyards and more of
    # everything else): in fact, it seems to include more SPs

    # We select the data points we are going to use (cf. appendix C2).
    # As Cobb-Douglas log-linear relation is only true for the formal sector,
    # we exclude SPs in the bottom quintile of property prices and for which
    # more than 5% of households are reported to live in "informal" housing.
    # We also exclude "rural" SPs (i.e., those that are large, with a small
    # share than can be urbanized).

    # NB: we also add other criteria compared to the working paper, namely we
    # exclude poorest income group (which is in effect crowded out from the
    # formal sector), as well as Mitchell's Plain (as its housing market is
    # very specific) and far-away land (for which we have few observations)

    if options["correct_selected_density"] == 0:
        selected_density = (
            (data_sp["price"] > np.nanquantile(data_sp["price"], 0.2))
            & (data_number_formal
               > 0.95 * housing_types_sp.total_dwellings_SP_2011)
            & (data_sp["unconstrained_area"]
                < np.nanquantile(data_sp["unconstrained_area"], 0.8))
            & (data_sp["unconstrained_area"] > 0.6 * 1000000 * data_sp["area"])
            )
    elif (options["correct_selected_density"] == 1
          and options["correct_mitchells_plain"] == 0):
        selected_density = (
            (data_sp["price"] > np.nanquantile(data_sp["price"], 0.2))
            & (data_number_formal
               > 0.95 * housing_types_sp.total_dwellings_SP_2011)
            & (data_sp["unconstrained_area"]
                < np.nanquantile(data_sp["unconstrained_area"], 0.8))
            & (data_sp["unconstrained_area"] > 0.6 * 1000000 * data_sp["area"])
            & (data_income_group > 0)
            & (data_sp["distance"] < 40)
            )
    elif (options["correct_selected_density"] == 1
          and options["correct_mitchells_plain"] == 1):
        selected_density = (
            (data_sp["price"] > np.nanquantile(data_sp["price"], 0.2))
            & (data_number_formal
               > 0.95 * housing_types_sp.total_dwellings_SP_2011)
            & (data_sp["unconstrained_area"]
                < np.nanquantile(data_sp["unconstrained_area"], 0.8))
            & (data_sp["unconstrained_area"] > 0.6 * 1000000 * data_sp["area"])
            & (data_income_group > 0)
            & (data_sp["mitchells_plain"] == 0)
            & (data_sp["distance"] < 40)
            )

    # NB: re-run the following regressions with robust standard errors

    # CONSTRUCTION FUNCTION PARAMETERS

    # We then estimate the coefficients of construction function
    coeff_b, coeff_a, coeffKappa = calmain.estim_construct_func_param(
        options, param, data_sp, threshold_income_distribution,
        income_distribution, data_rdp, housing_types_sp,
        data_number_formal, data_income_group, selected_density,
        path_data, path_precalc_inp, path_folder)

    # NB: relation between invested capital and building density to back up
    # values of empirical estimates?

    # We update parameter vector
    param["coeff_a"] = coeff_a
    param["coeff_b"] = coeff_b
    param["coeff_A"] = coeffKappa

    # INCOMES AND GRAVITY PARAMETER

    # We scan values for the gravity parameter to estimate incomes as a
    # function of it.
    # The value range is set by trial and error: the wider the range you want
    # to test, the longer.
    if options["scan_type"] == "rough":
        list_lambda = 10 ** np.arange(0.40, 0.51, 0.05)
    if options["scan_type"] == "normal":
        list_lambda = 10 ** np.arange(0.42, 0.441, 0.01)
    if options["scan_type"] == "fine":
        list_lambda = 10 ** np.arange(0.427, 0.4291, 0.001)

    # NB: this is too long and complex to run a solver directly
    # NB: We need to proceed in two steps as errors were drawn directly on
    # transport costs (and not on commuting pairs), hence no separate
    # identification of the gravity parameter and the incomes net of commuting
    # costs

    (incomeCentersKeep, lambdaKeep, cal_avg_income, scoreKeep,
     bhattacharyyaDistances) = (
        calmain.estim_incomes_and_gravity(
            param, grid, list_lambda, households_per_income_class,
            average_income, income_distribution, spline_inflation, spline_fuel,
            spline_population_income_distribution, spline_income_distribution,
            path_data, path_precalc_inp, path_precalc_transp, options)
        )

    # NB: compare estimates with existing literature

    # We validate calibrated incomes
    data_graph = pd.DataFrame(
        {'Calibration': cal_avg_income,
         'Data': average_income},
        index=["Poor", "Mid-poor", "Mid-rich", "Rich"])
    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    figure.tight_layout()
    data_graph.plot(kind="bar", ax=axis)
    plt.ylabel("Average income")
    plt.tick_params(labelbottom=True)
    plt.xticks(rotation='horizontal')
    # plt.savefig(path_plots + 'validation_cal_income.png')
    # plt.close()

    # We update parameter vector
    param["lambda"] = np.array(lambdaKeep)

    # UTILITY FUNCTION PARAMETERS

    # We compute local incomes net of commuting costs at the SP (not grid)
    # level that is used in calibration
    # Note that lambda and calibrated incomes have an impact here:
    # from now on, we will stop loading precalibrated parameters
    options["load_precal_param"] = 0
    (incomeNetOfCommuting, *_
     ) = inpdt.import_transport_data(
         grid, param, 0, households_per_income_class, average_income,
         spline_inflation, spline_fuel,
         spline_population_income_distribution, spline_income_distribution,
         path_precalc_inp, path_precalc_transp, 'SP', options)

    # Note that we dropped potential spatial autocorrelation for numerical
    # simplicity

    # Here, we also have an impact from construction parameters and sample
    # selection (+ number of formal units)

    # TODO: q0 needs to be fixed for the solver to be stable!
    (calibratedUtility_beta, calibratedUtility_q0, cal_amenities
     ) = calmain.estim_util_func_param(
         data_number_formal, data_income_group, housing_types_sp, data_sp,
         coeff_a, coeff_b, coeffKappa, interest_rate,
         incomeNetOfCommuting, selected_density, path_data, path_precalc_inp,
         options, param)

    # param["beta"] = calibratedUtility_beta
    # param["q0"] = calibratedUtility_q0


# %% Reload calibrated data

if options["run_calib"] == 1:

    # First, incomes net of commuting costs for all periods
    for t_temp in t:
        print(t_temp)
        (incomeNetOfCommuting, modalShares, ODflows, averageIncome
         ) = inpdt.import_transport_data(
             grid, param, t_temp, households_per_income_class, average_income,
             spline_inflation, spline_fuel,
             spline_population_income_distribution, spline_income_distribution,
             path_precalc_inp, path_precalc_transp, 'GRID', options)

    income_net_of_commuting_costs = np.load(
        path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')

    # Then, amenity data
    amenities = inpdt.import_amenities(path_precalc_inp, options)


# %% End calibration by fitting disamenity parameter for backyard
# and informal housing to the model

# Unemployment reweighting does not work fine: results should be shown for
# employed population

if options["run_calib"] == 1:

    # General calibration (see Pfeiffer et al., appendix C5)

    list_amenity_backyard = np.arange(0.64, 0.681, 0.01)
    list_amenity_settlement = np.arange(0.60, 0.641, 0.01)
    housing_type_total = pd.DataFrame(np.array(np.meshgrid(
        list_amenity_backyard, list_amenity_settlement)).T.reshape(-1, 2))
    housing_type_total.columns = ["param_backyard", "param_settlement"]
    housing_type_total["formal"] = np.zeros(
        len(housing_type_total.param_backyard))
    housing_type_total["backyard"] = np.zeros(
        len(housing_type_total.param_backyard))
    housing_type_total["informal"] = np.zeros(
        len(housing_type_total.param_backyard))
    housing_type_total["subsidized"] = np.zeros(
        len(housing_type_total.param_backyard))

    debut_calib_time = time.process_time()
    number_total_iterations = (
        len(list_amenity_backyard) * len(list_amenity_settlement))
    print(f"** Calibration: {number_total_iterations} iterations **")

    for i in range(0, len(list_amenity_backyard)):
        for j in range(0, len(list_amenity_settlement)):
            param["amenity_backyard"] = list_amenity_backyard[i]
            param["amenity_settlement"] = list_amenity_settlement[j]
            param["pockets"] = np.ones(24014) * param["amenity_settlement"]
            param["backyard_pockets"] = (np.ones(24014)
                                         * param["amenity_backyard"])
            (initial_state_utility,
             initial_state_error,
             initial_state_simulated_jobs,
             initial_state_households_housing_types,
             initial_state_household_centers,
             initial_state_households,
             initial_state_dwelling_size,
             initial_state_housing_supply,
             initial_state_rent,
             initial_state_rent_matrix,
             initial_state_capital_land,
             initial_state_average_income,
             initial_state_limit_city) = eqcmp.compute_equilibrium(
                 fraction_capital_destroyed,
                 amenities,
                 param,
                 housing_limit,
                 population,
                 households_per_income_class,
                 total_RDP,
                 coeff_land,
                 income_net_of_commuting_costs,
                 grid,
                 options,
                 agricultural_rent,
                 interest_rate,
                 number_properties_RDP,
                 average_income,
                 mean_income,
                 income_class_by_housing_type,
                 minimum_housing_supply,
                 param["coeff_A"],
                 income_2011)

            # We fill output matrix with the total number of HHs per housing
            # type for given values of backyard and informal amenity parameters
            housing_type_total.iloc[
                (housing_type_total.param_backyard
                 == param["amenity_backyard"])
                & (housing_type_total.param_settlement
                   == param["amenity_settlement"]),
                2:6] = np.nansum(initial_state_households_housing_types, 1)
            time_elapsed = time.process_time() - debut_calib_time
            iteration_number = i * len(list_amenity_settlement) + j + 1

            print(f"iteration {iteration_number}/{number_total_iterations}.",
                  str(datetime.timedelta(seconds=round(time_elapsed))),
                  f"elapsed ({round(time_elapsed/iteration_number)}s per iter",
                  "There remains:",
                  str(datetime.timedelta(seconds=round(
                      time_elapsed
                      / iteration_number
                      * (number_total_iterations-iteration_number)))))

    # We choose the set of parameters that minimize the sum of abs differences
    # between simulated and observed total number of households in each housing
    # type (without RDP, which is exogenously set equal to data)

    distance_share = np.abs(
        housing_type_total.iloc[:, 2:5] - housing_type_data[None, 0:3])
    distance_share_score = (
        distance_share.iloc[:, 1] + distance_share.iloc[:, 2])

    which = np.argmin(distance_share_score)
    min_score = np.nanmin(distance_share_score)
    calibrated_amenities = housing_type_total.iloc[which, 0:2]

    param["amenity_backyard"] = calibrated_amenities[0]
    param["amenity_settlement"] = calibrated_amenities[1]

    try:
        os.mkdir(path_precalc_inp)
    except OSError as error:
        print(error)

    # Works the same as in paper
    np.save(path_precalc_inp + 'param_amenity_backyard.npy',
            param["amenity_backyard"])
    np.save(path_precalc_inp + 'param_amenity_settlement.npy',
            param["amenity_settlement"])

    if options["location_based_calib"] == 1:

        index = 0
        index_max = 50
        metrics = np.zeros(index_max)

        # We start from where we left (to gain time) and compute the
        # equilibrium again
        param["pockets"] = np.zeros(24014) + param["amenity_settlement"]
        save_param_informal_settlements = np.zeros((index_max, 24014))
        metrics_is = np.zeros(index_max)
        param["backyard_pockets"] = np.zeros(24014) + param["amenity_backyard"]
        save_param_backyards = np.zeros((index_max, 24014))
        metrics_ib = np.zeros(index_max)

        print("\n* City limits *")

        (initial_state_utility,
         initial_state_error,
         initial_state_simulated_jobs,
         initial_state_households_housing_types,
         initial_state_household_centers,
         initial_state_households,
         initial_state_dwelling_size,
         initial_state_housing_supply,
         initial_state_rent,
         initial_state_rent_matrix,
         initial_state_capital_land,
         initial_state_average_income,
         initial_state_limit_city
         ) = eqcmp.compute_equilibrium(
             fraction_capital_destroyed,
             amenities,
             param,
             housing_limit,
             population,
             households_per_income_class,
             total_RDP,
             coeff_land,
             income_net_of_commuting_costs,
             grid,
             options,
             agricultural_rent,
             interest_rate,
             number_properties_RDP,
             average_income,
             mean_income,
             income_class_by_housing_type,
             minimum_housing_supply,
             param["coeff_A"],
             income_2011)

        print("\n** ITERATIONS **")

        debut_iterations_time = time.process_time()
        number_total_iterations = index_max

        # Then we optimize over the number of households per housing type
        # PER PIXEL, and not just on the aggregate number (to acccount for
        # differing disamenities per location, e.g. eviction probability,
        # infrastructure networks, etc.)

        # To do so, we use granular housing_types (from SAL data) instead of
        # aggregate housing_types

        for index in range(0, index_max):

            # IS
            diff_is = np.zeros(24014)
            for i in range(0, 24014):
                diff_is[i] = (housing_types.informal_grid[i]
                              - initial_state_households_housing_types[2, :][i]
                              )
                # We apply an empirical reweighting that helps convergence
                adj = (diff_is[i] / 150000)
                # We increase the amenity score when we underestimate the nb of
                # HHs
                param["pockets"][i] = param["pockets"][i] + adj
            # We store iteration outcome and prevent extreme sorting from
            # happening due to the amenity score
            metrics_is[index] = sum(np.abs(diff_is))
            param["pockets"][param["pockets"] < 0.05] = 0.05
            param["pockets"][param["pockets"] > 0.99] = 0.99
            save_param_informal_settlements[index, :] = param["pockets"]

            # IB
            diff_ib = np.zeros(24014)
            for i in range(0, 24014):
                if options["actual_backyards"] == 1:
                    diff_ib[i] = (
                        housing_types.backyard_informal_grid[i]
                        + housing_types.backyard_formal_grid[i]
                        - initial_state_households_housing_types[1, :][i])
                elif options["actual_backyards"] == 0:
                    diff_ib[i] = (
                        housing_types.backyard_informal_grid[i]
                        - initial_state_households_housing_types[1, :][i])
                adj = (diff_ib[i] / 75000)
                param["backyard_pockets"][i] = (
                    param["backyard_pockets"][i] + adj)
            metrics_ib[index] = sum(np.abs(diff_ib))
            param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
            param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99
            save_param_backyards[index, :] = param["backyard_pockets"]

            metrics[index] = metrics_is[index] + metrics_ib[index]

            # We run the equilibrium again with updated values of
            # informal/backyard housing disamenity indices, then go to the next
            # iteration

            (initial_state_utility, initial_state_error,
             initial_state_simulated_jobs,
             initial_state_households_housing_types,
             initial_state_household_centers,
             initial_state_households, initial_state_dwelling_size,
             initial_state_housing_supply, initial_state_rent,
             initial_state_rent_matrix, initial_state_capital_land,
             initial_state_average_income, initial_state_limit_city
             ) = eqcmp.compute_equilibrium(
                 fraction_capital_destroyed, amenities, param, housing_limit,
                 population, households_per_income_class, total_RDP,
                 coeff_land, income_net_of_commuting_costs, grid, options,
                 agricultural_rent, interest_rate, number_properties_RDP,
                 average_income, mean_income, income_class_by_housing_type,
                 minimum_housing_supply, param["coeff_A"], income_2011)

            time_elapsed = time.process_time() - debut_iterations_time
            iteration_number = index + 1

            print(f"iteration {iteration_number}/{number_total_iterations}",
                  str(datetime.timedelta(seconds=round(time_elapsed))),
                  f"elapsed ({round(time_elapsed/iteration_number)}s / iter)",
                  "There remains:",
                  str(datetime.timedelta(seconds=round(
                      time_elapsed
                      / iteration_number
                      * (number_total_iterations-iteration_number))))
                  )

        # We pick the set of parameters that minimize the sum of absolute diffs
        # between data and simulation
        score_min = np.min(metrics)
        index_min = np.argmin(metrics)
        # metrics[index_min]
        param["pockets"] = save_param_informal_settlements[index_min]
        param["backyard_pockets"] = save_param_backyards[index_min]

        print(np.nanmin(param["pockets"]))
        print(np.nanmean(param["pockets"]))
        print(np.nanmax(param["pockets"]))
        print(np.nanmin(param["backyard_pockets"]))
        print(np.nanmean(param["backyard_pockets"]))
        print(np.nanmax(param["backyard_pockets"]))

        try:
            os.mkdir(path_precalc_inp)
        except OSError as error:
            print(error)

        np.save(path_precalc_inp + 'param_pockets.npy',
                param["pockets"])
        np.save(path_precalc_inp + 'param_backyards.npy',
                param["backyard_pockets"])

    print("Calibration process - end")


# %% Compute initial state

print("Compute initial state")

# NB: Note that we use a Cobb-Douglas production function all along!
# Also note that we simulate households as being a couple

# NB: Do some bootstrapping

(initial_state_utility,
 initial_state_error,
 initial_state_simulated_jobs,
 initial_state_households_housing_types,
 initial_state_household_centers,
 initial_state_households,
 initial_state_dwelling_size,
 initial_state_housing_supply,
 initial_state_rent,
 initial_state_rent_matrix,
 initial_state_capital_land,
 initial_state_average_income,
 initial_state_limit_city) = eqcmp.compute_equilibrium(
     fraction_capital_destroyed,
     amenities,
     param,
     housing_limit,
     population,
     households_per_income_class,
     total_RDP,
     coeff_land,
     income_net_of_commuting_costs,
     grid,
     options,
     agricultural_rent,
     interest_rate,
     number_properties_RDP,
     average_income,
     mean_income,
     income_class_by_housing_type,
     minimum_housing_supply,
     param["coeff_A"],
     income_2011)

# Reminder: income groups are ranked from poorer to richer, and housing types
# follow the following order: formal-backyard-informal-RDP

# Note on outputs (with dimensions in same order as axes):
# initial_state_utility = utility for each income group (no RDP)
#   after optimization
# initial_state_error = value of error term for each group after optimization
# initial_state_simulated_jobs = total number of households per housing type
#   (no RDP) and income group
# initial_state_households_housing_types = number of households
#   per housing type (with RDP) per pixel
# initial_state_household_centers = number of households per income group
#   per pixel
# initial_state_households = number of households in each housing type
#   and income group per pixel
# initial_state_dwelling_size = dwelling size (in m²) for each housing type
#   per pixel
# initial_state_housing_supply = housing surface built (in m²) per unit of
#   available land (in km²) for each housing type in each pixel
# initial_state_rent = average rent (in rands/m²) for each housing type
#   in each pixel
# initial_state_rent_matrix = average willingness to pay (in rands)
#   for each housing type (no RDP) and each income group in each pixel
# initial_state_capital_land = value of the (housing construction sector)
#   capital stock (in available-land unit equivalent) per unit of available
#   land (in km²) in each housing type (no RDP) and each selected pixel
# initial_state_average_income = average income per income group
#   (not an output of the model)
# initial_state_limit_city = indicator dummy for having strictly more
#   than one household per housing type and income group in each pixel

# Save outputs

try:
    os.mkdir(path_outputs + name)
except OSError as error:
    print(error)


np.save(path_outputs + name + '/initial_state_utility.npy',
        initial_state_utility)
np.save(path_outputs + name + '/initial_state_error.npy',
        initial_state_error)
np.save(path_outputs + name + '/initial_state_simulated_jobs.npy',
        initial_state_simulated_jobs)
np.save(path_outputs + name + '/initial_state_households_housing_types.npy',
        initial_state_households_housing_types)
np.save(path_outputs + name + '/initial_state_household_centers.npy',
        initial_state_household_centers)
np.save(path_outputs + name + '/initial_state_households.npy',
        initial_state_households)
np.save(path_outputs + name + '/initial_state_dwelling_size.npy',
        initial_state_dwelling_size)
np.save(path_outputs + name + '/initial_state_housing_supply.npy',
        initial_state_housing_supply)
np.save(path_outputs + name + '/initial_state_rent.npy',
        initial_state_rent)
np.save(path_outputs + name + '/initial_state_rent_matrix.npy',
        initial_state_rent_matrix)
np.save(path_outputs + name + '/initial_state_capital_land.npy',
        initial_state_capital_land)
np.save(path_outputs + name + '/initial_state_average_income.npy',
        initial_state_average_income)
np.save(path_outputs + name + '/initial_state_limit_city.npy',
        initial_state_limit_city)

# %% Scenarios

print("Compute simulations")

# RUN SIMULATION: time depends on the timeline (long with 30 years)
(simulation_households_center,
 simulation_households_housing_type,
 simulation_dwelling_size,
 simulation_rent,
 simulation_households,
 simulation_error,
 simulation_housing_supply,
 simulation_utility,
 simulation_deriv_housing,
 simulation_T) = eqsim.run_simulation(
     t,
     options,
     param,
     grid,
     initial_state_utility,
     initial_state_error,
     initial_state_households,
     initial_state_households_housing_types,
     initial_state_housing_supply,
     initial_state_household_centers,
     initial_state_average_income,
     initial_state_rent,
     initial_state_dwelling_size,
     fraction_capital_destroyed,
     amenities,
     housing_limit,
     spline_estimate_RDP,
     spline_land_constraints,
     spline_land_backyard,
     spline_land_RDP,
     spline_land_informal,
     income_class_by_housing_type,
     path_precalc_transp,
     spline_RDP,
     spline_agricultural_rent,
     spline_interest_rate,
     spline_population_income_distribution,
     spline_inflation,
     spline_income_distribution,
     spline_population,
     spline_income,
     spline_minimum_housing_supply,
     spline_fuel,
     income_2011
     )

# Save outputs

try:
    os.mkdir(path_outputs + name)
except OSError as error:
    print(error)

np.save(path_outputs + name + '/simulation_households_center.npy',
        simulation_households_center)
np.save(path_outputs + name + '/simulation_households_housing_type.npy',
        simulation_households_housing_type)
np.save(path_outputs + name + '/simulation_dwelling_size.npy',
        simulation_dwelling_size)
np.save(path_outputs + name + '/simulation_rent.npy',
        simulation_rent)
np.save(path_outputs + name + '/simulation_households.npy',
        simulation_households)
np.save(path_outputs + name + '/simulation_error.npy',
        simulation_error)
np.save(path_outputs + name + '/simulation_housing_supply.npy',
        simulation_housing_supply)
np.save(path_outputs + name + '/simulation_utility.npy',
        simulation_utility)
np.save(path_outputs + name + '/simulation_deriv_housing.npy',
        simulation_deriv_housing)
np.save(path_outputs + name + '/simulation_T.npy',
        simulation_T)
