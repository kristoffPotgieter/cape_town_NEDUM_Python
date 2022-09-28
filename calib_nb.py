# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
# ---

# # Notebook: run calibration

# ## Preamble
#
# ### Import packages

# region
# We import standard Python libraries
import numpy as np
import pandas as pd
import os

# We also import our own packages
import inputs.data as inpdt
import inputs.parameters_and_options as inpprm
import equilibrium.compute_equilibrium as eqcmp
import equilibrium.functions_dynamic as eqdyn
import calibration.calib_main_func as calmain
# endregion

# ### Define file paths

path_code = '..'
path_folder = path_code + '/2. Data/'
path_precalc_inp = path_folder + '0. Precalculated inputs/'
path_data = path_folder + 'data_Cape_Town/'
path_precalc_transp = path_folder + 'precalculated_transport/'
path_scenarios = path_folder + 'data_Cape_Town/Scenarios/'
path_outputs = path_code + '/4. Output/'
path_floods = path_folder + "FATHOM/"


# ## Import parameters and options

# ### We import default parameter and options

options = inpprm.import_options()
param = inpprm.import_param(
    path_precalc_inp, path_outputs, path_folder, options)

# ### We also set custom options for this simulation

# #### We first set options regarding structural assumptions used in the model

# Dummy for taking floods into account in agents' choices
options["agents_anticipate_floods"] = 1
# Dummy for preventing new informal settlement development
options["informal_land_constrained"] = 0

# #### Then we set options regarding flood data used

# Dummy for taking pluvial floods into account (on top of fluvial floods)
options["pluvial"] = 1
# Dummy for reducing pluvial risk for (better protected) formal structures
options["correct_pluvial"] = 1
# Dummy for taking coastal floods into account (on top of fluvial floods)
options["coastal"] = 1
# Digital elevation model to be used with coastal floods (MERITDEM or NASADEM)
# NB: MERITDEM is also the DEM used for fluvial and pluvial flood data
options["dem"] = "MERITDEM"
# Dummy for taking defended (vs. undefended) fluvial flood maps
# NB: FATHOM recommends to use undefended maps due to the high uncertainty
# in infrastructure modelling
options["defended"] = 1
# Dummy for taking sea-level rise into account in coastal flood data
# NB: Projections are up to 2050, based upon IPCC AR5 assessment for the
# RCP 8.5 scenario
options["slr"] = 1

# #### We also set options for scenarios on time-moving exogenous variables

# NB: Must be set to 1/2/3 for low/medium/high growth scenario
options["inc_ineq_scenario"] = 2
options["pop_growth_scenario"] = 3
options["fuel_price_scenario"] = 2

# #### Finally, we set options regarding data processing

# Default is set at zero to save computing time
# (data is simply loaded in the model)
#
# NB: this is only needed to create the data for the first time, or when the
# source is changed, so that pre-processed data is updated

# Dummy for converting small-area-level (SAL) data into grid-level data
# (used for result validation)
options["convert_sal_data"] = 0
# Dummy for computing expected income net of commuting costs on the basis
# of calibrated wages
options["compute_net_income"] = 0


# ## Load data

# ### Basic geographic data

grid, center = inpdt.import_grid(path_data)
amenities = inpdt.import_amenities(path_precalc_inp, options)


# ### Macro data

(interest_rate, population, housing_type_data, total_RDP
 ) = inpdt.import_macro_data(param, path_scenarios, path_folder)


# ### Households and income data

# region
income_class_by_housing_type = inpdt.import_hypothesis_housing_type()

(mean_income, households_per_income_class, average_income, income_mult,
 income_2011, households_per_income_and_housing
 ) = inpdt.import_income_classes_data(param, path_data)

# NB: we create this parameter to maintain money illusion in simulations
# (see eqsim.run_simulation function)
param["income_year_reference"] = mean_income

# Other data at SP (small place) level used for calibration and validation
(data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011,
 grid_formal_density_HFA, threshold_income_distribution, income_distribution,
 cape_town_limits) = inpdt.import_households_data(path_precalc_inp)

# Import nb of households per pixel, by housing type (from SAL data)
# NB: RDP housing is included in formal, and there are both formal and informal
# backyards
if options["convert_sal_data"] == 1:
    housing_types = inpdt.import_sal_data(grid, path_folder, path_data,
                                          housing_type_data)
housing_types = pd.read_excel(path_folder + 'housing_types_grid_sal.xlsx')
housing_types[np.isnan(housing_types)] = 0
# endregion

# ### Land use projections

# region
# We import basic projections
(spline_RDP, spline_estimate_RDP, spline_land_RDP,
 spline_land_backyard, spline_land_informal, spline_land_constraints,
 number_properties_RDP) = (
     inpdt.import_land_use(grid, options, param, data_rdp, housing_types,
                           housing_type_data, path_data, path_folder)
     )

# We correct areas for each housing type at baseline year for the amount of
# constructible land in each type
coeff_land = inpdt.import_coeff_land(
    spline_land_constraints, spline_land_backyard, spline_land_informal,
    spline_land_RDP, param, 0)

# We import housing heigth limits
housing_limit = inpdt.import_housing_limit(grid, param)

# We update parameter vector with construction parameters
# (relies on loaded data) and compute other variables
(param, minimum_housing_supply, agricultural_rent
 ) = inpprm.import_construction_parameters(
    param, grid, housing_types_sp, data_sp["dwelling_size"],
    mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land,
    interest_rate, options
    )
# endregion

# ### Import flood data (takes some time when agents anticipate floods)

# region
# If agents anticipate floods, we return output from damage functions
if options["agents_anticipate_floods"] == 1:
    (fraction_capital_destroyed, structural_damages_small_houses,
     structural_damages_medium_houses, structural_damages_large_houses,
     content_damages, structural_damages_type1, structural_damages_type2,
     structural_damages_type3a, structural_damages_type3b,
     structural_damages_type4a, structural_damages_type4b
     ) = inpdt.import_full_floods_data(options, param, path_folder,
                                       housing_type_data)

# Else, we set those outputs as zero
# NB: 24014 is the number of grid pixels
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
# endregion

# ### Import scenarios (for time-moving variables)

(spline_agricultural_rent, spline_interest_rate,
 spline_population_income_distribution, spline_inflation,
 spline_income_distribution, spline_population,
 spline_income, spline_minimum_housing_supply, spline_fuel
 ) = eqdyn.import_scenarios(income_2011, param, grid, path_scenarios,
                            options)

# ### Import income net of commuting costs (for all time periods)

# region
if options["compute_net_income"] == 1:
    (incomeNetOfCommuting, modalShares, ODflows, averageIncome
     ) = inpdt.import_transport_data(
         grid, param, 0, households_per_income_class, average_income,
         spline_inflation, spline_fuel,
         spline_population_income_distribution, spline_income_distribution,
         path_precalc_inp, path_precalc_transp, 'GRID', options)

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')
# endregion


# ## Prepare data

# ### Define dominant income group in each census block (SP)

# region
# In one case, we base our definition on the median income at the SP-level:
# we consider as dominant the group corresponding to the highest income
# threshold is crossed
if options["correct_dominant_incgrp"] == 0:
    data_income_group = np.zeros(len(data_sp["income"]))
    for j in range(0, param["nb_of_income_classes"] - 1):
        data_income_group[data_sp["income"] >
                          threshold_income_distribution[j]] = j+1

# In the other case, we just consider the most numerous income group in each
# census block
elif options["correct_dominant_incgrp"] == 1:
    data_income_group = np.zeros(len(income_distribution))
    for i in range(0, len(income_distribution)):
        data_income_group[i] = np.argmax(income_distribution[i])
# endregion

# Although the second option seems more logical, we take the first one as
# default since we are going to consider median SP prices, and we want
# associated net income to be in line with those values to avoid a sample
# selection bias in our regressions.

# ### Obtain number of formal private housing units per SP

# NB: it is not clear whether RDP are included in SP formal count, and
# if they should be taken out based on imperfect cadastral estimations.
# For reference, we include the two options.

# region
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
# endregion

# Given the uncertainty surrounding RDP counts, we take the second option as
# default and prefer to rely on sample selection (see below) to exclude the SPs
# where RDP housing is likely to drive most of our results

# ### Sample selection

# As the relations we are going to estimate are only true for the
# formal private sector, we exclude SPs in the bottom quintile of property
# prices and for which more than 5% of households are reported to live in
# informal housing (settlements + backyards). We also exclude "rural" SPs
# (i.e., those that are large, with a small share than can be urbanized).

# We also add options to consider other criteria, namely we offer to
# exclude poorest income group (which is in effect crowded out from the
# formal sector), as well as Mitchell's Plain (as its housing market is
# very specific) and far-away land (for which we have few observations)

# region
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
# endregion

# We pick the second choice as our default since it is more conservative than
# the first, and less ad hoc than the third one.


# ## Calibrate construction function parameters

# We estimate construction function parameters based on a log-linearization
# of the housing market clearing condition
coeff_b, coeff_a, coeffKappa = calmain.estim_construct_func_param(
    options, param, data_sp, threshold_income_distribution,
    income_distribution, data_rdp, housing_types_sp,
    data_number_formal, data_income_group, selected_density,
    path_data, path_precalc_inp, path_folder)

# NB: The results are automatically saved for later use in simulations

# We update parameter vector
param["coeff_a"] = coeff_a
param["coeff_b"] = coeff_b
param["coeff_A"] = coeffKappa


# ## Calibrate incomes and gravity parameter

# We scan values for the gravity parameter to estimate incomes as a
# function of it. The value range is set by trial and error: the wider the
# range you want to test, the longer. In principle, we should find a value
# within a coarse interval before going to the finer level: this may require
# several iterations if the underlying data changes.
# NB: we do that as it is too long and complex to run a solver directly

# Then, we select the income-gravity pair that best fits the distribution
# of commuters over distance from the CBD.
# NB: we need to proceed in twos steps as there is no separate identification
# of the gravity parameter and the net incomes

# We start by selecting the range over which we want to scan
if options["scan_type"] == "rough":
    list_lambda = 10 ** np.arange(0.40, 0.51, 0.05)
if options["scan_type"] == "normal":
    list_lambda = 10 ** np.arange(0.42, 0.441, 0.01)
if options["scan_type"] == "fine":
    list_lambda = 10 ** np.arange(0.427, 0.4291, 0.001)

# We then run the function that returns the calibrated outputs
(incomeCentersKeep, lambdaKeep, cal_avg_income, scoreKeep,
 bhattacharyyaDistances) = (
    calmain.estim_incomes_and_gravity(
        param, grid, list_lambda, households_per_income_class,
        average_income, income_distribution, spline_inflation, spline_fuel,
        spline_population_income_distribution, spline_income_distribution,
        path_data, path_precalc_inp, path_precalc_transp, options)
    )

# We update the parameter vector
param["lambda"] = np.array(lambdaKeep)


# ## Calibrate utility function parameters

# We compute local incomes net of commuting costs at the SP (not grid)
# level that is used in calibration.

# region
# Note that lambda and calibrated incomes have an impact here: from now on,
# we will stop loading precalibrated parameters to rely on the newly
# calibrated parameters that we just saved
options["load_precal_param"] = 0

(incomeNetOfCommuting, *_
 ) = inpdt.import_transport_data(
     grid, param, 0, households_per_income_class, average_income,
     spline_inflation, spline_fuel,
     spline_population_income_distribution, spline_income_distribution,
     path_precalc_inp, path_precalc_transp, 'SP', options)
# endregion

# Then we calibrate utility function parameters based on the maximization
# of a composite likelihood that reproduces the fit on exogenous amenities,
# dwelling sizes, and income sorting

# NB: Here, we also have an impact from construction parameters and sample
# selection (+ number of formal units)
(calibratedUtility_beta, calibratedUtility_q0, cal_amenities
 ) = calmain.estim_util_func_param(
     data_number_formal, data_income_group, housing_types_sp, data_sp,
     coeff_a, coeff_b, coeffKappa, interest_rate,
     incomeNetOfCommuting, selected_density, path_data, path_precalc_inp,
     options, param)

# We update parameter vector
param["beta"] = calibratedUtility_beta
param["q0"] = calibratedUtility_q0


# ## Calibrate disamenity index for informal backyards + settlements

# region
# We first need to recompute income net of commuting costs at baseline
# year since calibrated income has changed
(incomeNetOfCommuting, modalShares, ODflows, averageIncome
 ) = inpdt.import_transport_data(
     grid, param, 0, households_per_income_class, average_income,
     spline_inflation, spline_fuel,
     spline_population_income_distribution, spline_income_distribution,
     path_precalc_inp, path_precalc_transp, 'GRID', options)

income_net_of_commuting_costs = np.load(
    path_precalc_transp + 'GRID_incomeNetOfCommuting_0.npy')
# endregion

# Then, we do the same for the amenity index
amenities = inpdt.import_amenities(path_precalc_inp, options)

# NB: Since disamenity index calibration relies on the model fit and is not
# computed a priori (contrary to other parameters), the options set in the
# preamble should be the same as the ones used in the main script, so that
# the calibrated values are in line with the structural assumptions used

# ### We start with a general (not location-specific) calibration

# We define a range of disamenity values which we would like to scan,
# and arrange them in a grid
list_amenity_backyard = np.arange(0.64, 0.681, 0.01)
list_amenity_settlement = np.arange(0.60, 0.641, 0.01)
housing_type_total = pd.DataFrame(np.array(np.meshgrid(
    list_amenity_backyard, list_amenity_settlement)).T.reshape(-1, 2))
housing_type_total.columns = ["param_backyard", "param_settlement"]

# We initialize output vector
housing_type_total["formal"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["backyard"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["informal"] = np.zeros(
    len(housing_type_total.param_backyard))
housing_type_total["subsidized"] = np.zeros(
    len(housing_type_total.param_backyard))

# We print the number of total iterations (to have an intuition of how long
# the process will take)
number_total_iterations = (
    len(list_amenity_backyard) * len(list_amenity_settlement))
print(f"** Calibration: {number_total_iterations} iterations **")

# We are going to compute the initial state equilibrium for each pair of
# parameters, and retain the one that best fits the observed number of
# households in informal settlements + backyards

for i in range(0, len(list_amenity_backyard)):
    for j in range(0, len(list_amenity_settlement)):

        # We set input values
        param["amenity_backyard"] = list_amenity_backyard[i]
        param["amenity_settlement"] = list_amenity_settlement[j]
        param["pockets"] = np.ones(24014) * param["amenity_settlement"]
        param["backyard_pockets"] = (np.ones(24014)
                                     * param["amenity_backyard"])

        # We run the algorithm
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

        # We update the iteration count and print progress made
        iteration_number = i * len(list_amenity_settlement) + j + 1
        print(f"iteration {iteration_number}/{number_total_iterations}")

# region
# We compute the error between simulated and observed number of households
# in each housing type (without RDP, which is exogenously set equal to data)
distance_share = np.abs(
    housing_type_total.iloc[:, 2:5] - housing_type_data[None, 0:3])

# We define the score that we want to minimize as the sum of the errors for
# informal backyards and informal settlements
distance_share_score = (
    distance_share.iloc[:, 1] + distance_share.iloc[:, 2])

# We select the arguments associated with the minimum
which = np.argmin(distance_share_score)
min_score = np.nanmin(distance_share_score)
calibrated_amenities = housing_type_total.iloc[which, 0:2]
# endregion

# We update parameter vector
param["amenity_backyard"] = calibrated_amenities[0]
param["amenity_settlement"] = calibrated_amenities[1]

# region
# We create the output directory and save the values
try:
    os.mkdir(path_precalc_inp)
except OSError as error:
    print(error)

np.save(path_precalc_inp + 'param_amenity_backyard.npy',
        param["amenity_backyard"])
np.save(path_precalc_inp + 'param_amenity_settlement.npy',
        param["amenity_settlement"])
# endregion

# ### Calibrate location-specific disamenity index

# Default is set to 1 but can be changed if we fear overfit of the model
if options["location_based_calib"] == 1:

    # We start from where we left (to gain time) and compute the
    # equilibrium again

    # We first initialize input values

    index = 0
    index_max = 50
    metrics = np.zeros(index_max)

    param["pockets"] = np.zeros(24014) + param["amenity_settlement"]
    save_param_informal_settlements = np.zeros((index_max, 24014))
    metrics_is = np.zeros(index_max)
    param["backyard_pockets"] = np.zeros(24014) + param["amenity_backyard"]
    save_param_backyards = np.zeros((index_max, 24014))
    metrics_ib = np.zeros(index_max)

    # We run the algorithm
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

    # We set the maximum number of iterations
    number_total_iterations = index_max

    # Then we optimize over the number of households per housing type
    # PER PIXEL, and not just on the aggregate number (to acccount for
    # differing disamenities per location, e.g. eviction probability,
    # infrastructure networks, etc.)

    # To do so, we use granular housing_types variable (from SAL data) instead
    # of aggregate housing_types variable

    for index in range(0, index_max):

        # INFORMAL SETTLEMENTS

        # We initialize output vector
        diff_is = np.zeros(24014)
        for i in range(0, 24014):
            # We store the error term
            diff_is[i] = (housing_types.informal_grid[i]
                          - initial_state_households_housing_types[2, :][i]
                          )
            # We apply an empirical reweighting that helps convergence
            adj = (diff_is[i] / 150000)
            # We increase the amenity score when we underestimate the nb of
            # households
            param["pockets"][i] = param["pockets"][i] + adj
        # We store iteration outcome and prevent extreme sorting from
        # happening due to the amenity score
        metrics_is[index] = sum(np.abs(diff_is))
        param["pockets"][param["pockets"] < 0.05] = 0.05
        param["pockets"][param["pockets"] > 0.99] = 0.99
        save_param_informal_settlements[index, :] = param["pockets"]

        # INFORMAL BACKYARDS

        # We initialize output vector
        diff_ib = np.zeros(24014)
        for i in range(0, 24014):
            # Note that we add an option depending on whether we restrict
            # ourselves to informal backyards (default) or all kinds of
            # backyards (not warranted given the standardized structure
            # assumed in the model)
            if options["actual_backyards"] == 1:
                diff_ib[i] = (
                    housing_types.backyard_informal_grid[i]
                    + housing_types.backyard_formal_grid[i]
                    - initial_state_households_housing_types[1, :][i])
            elif options["actual_backyards"] == 0:
                diff_ib[i] = (
                    housing_types.backyard_informal_grid[i]
                    - initial_state_households_housing_types[1, :][i])
            # We help convergence and update parameter
            adj = (diff_ib[i] / 75000)
            param["backyard_pockets"][i] = (
                param["backyard_pockets"][i] + adj)
        # We store iteration output and prevent extreme sorting
        metrics_ib[index] = sum(np.abs(diff_ib))
        param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
        param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99
        save_param_backyards[index, :] = param["backyard_pockets"]

        # We retain the sum of the errors as our minimization objective
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

        iteration_number = index + 1

        print(f"iteration {iteration_number}/{number_total_iterations}")

    # We pick the set of parameters that minimize the sum of absolute diffs
    # between data and simulation
    score_min = np.min(metrics)
    index_min = np.argmin(metrics)

    # We update the parameter vector
    param["pockets"] = save_param_informal_settlements[index_min]
    param["backyard_pockets"] = save_param_backyards[index_min]

    # We print the basic distribution of the calibrated parameters
    print(np.nanmin(param["pockets"]))
    print(np.nanmean(param["pockets"]))
    print(np.nanmax(param["pockets"]))
    print(np.nanmin(param["backyard_pockets"]))
    print(np.nanmean(param["backyard_pockets"]))
    print(np.nanmax(param["backyard_pockets"]))

    # We create the output directory and save values
    try:
        os.mkdir(path_precalc_inp)
    except OSError as error:
        print(error)
    np.save(path_precalc_inp + 'param_pockets.npy',
            param["pockets"])
    np.save(path_precalc_inp + 'param_backyards.npy',
            param["backyard_pockets"])
