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
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Notebook: run model

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
import equilibrium.run_simulations as eqsim
import equilibrium.functions_dynamic as eqdyn
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

# ### Set timeline for simulations

t = np.arange(0, 30)


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
# TODO: Explain the difference between defended vs undefended
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


# ## Give name to simulation to export the results

# NB: this changes according to custom parameters of interest
name = ('floods' + str(options["agents_anticipate_floods"])
        + str(options["informal_land_constrained"])
        + '_F' + str(options["defended"])
        + '_P' + str(options["pluvial"]) + str(options["correct_pluvial"])
        + '_C' + str(options["coastal"]) + str(options["slr"])
        + '_scenario' + str(options["inc_ineq_scenario"])
        + str(options["pop_growth_scenario"])
        + str(options["fuel_price_scenario"]))


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
# endregion


# ## Compute initial state equilibrium

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

# Reminder: income groups are ranked from poorer to richer, and housing types
# follow the following order: formal-backyard-informal-RDP

# Note on outputs (with dimensions in same order as axes):
#
# initial_state_utility = utility for each income group (no RDP)
#   after optimization
#
# initial_state_error = value of error term for each group after optimization
#
# initial_state_simulated_jobs = total number of households per housing type
#   (no RDP) and income group
#
# initial_state_households_housing_types = number of households
#   per housing type (with RDP) per pixel
#
# initial_state_household_centers = number of households per income group
#   per pixel
#
# initial_state_households = number of households in each housing type
#   and income group per pixel
#
# initial_state_dwelling_size = dwelling size (in m²) for each housing type
#   per pixel
#
# initial_state_housing_supply = housing surface built (in m²) per unit of
#   available land (in km²) for each housing type in each pixel
#
# initial_state_rent = average rent (in rands/m²) for each housing type
#   in each pixel
#
# initial_state_rent_matrix = average willingness to pay (in rands)
#   for each housing type (no RDP) and each income group in each pixel
#
# initial_state_capital_land = value of the (housing construction sector)
#   capital stock (in available-land unit equivalent) per unit of available
#   land (in km²) in each housing type (no RDP) and each selected pixel
#
# initial_state_average_income = average income per income group
#   (not an output of the model)
#
# initial_state_limit_city = indicator dummy for having strictly more
#   than one household per housing type and income group in each pixel

# We create the associated output directory
try:
    os.mkdir(path_outputs + name)
except OSError as error:
    print(error)

# We save the output
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


# ## Run simulations for subsequent periods (time depends on timeline length)

# We run the algorithm
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

# We create the associated output directory
try:
    os.mkdir(path_outputs + name)
except OSError as error:
    print(error)

# We save the output
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
