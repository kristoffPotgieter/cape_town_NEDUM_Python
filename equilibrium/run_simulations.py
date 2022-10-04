# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:23:16 2020.

@author: Charlotte Liotta
"""

import copy
import numpy as np

import equilibrium.functions_dynamic as eqdyn
import equilibrium.compute_equilibrium as eqcmp
import inputs.parameters_and_options as inpprm
import inputs.data as inpdt


def run_simulation(t, options, param, grid, initial_state_utility,
                   initial_state_error, initial_state_households,
                   initial_state_households_housing_types,
                   initial_state_housing_supply,
                   initial_state_household_centers,
                   initial_state_average_income, initial_state_rent,
                   initial_state_dwelling_size, fraction_capital_destroyed,
                   amenities, housing_limit, spline_estimate_RDP,
                   spline_land_constraints, spline_land_backyard,
                   spline_land_RDP, spline_land_informal,
                   income_class_by_housing_type,
                   precalculated_transport, spline_RDP,
                   spline_agricultural_price, spline_interest_rate,
                   spline_population_income_distribution, spline_inflation,
                   spline_income_distribution, spline_population,
                   spline_income,
                   spline_minimum_housing_supply, spline_fuel,
                   income_baseline):
    """
    Run simulations over several years according to exogenous scenarios.

    After accounting for the changes in time-moving exogenous variables each
    period, the function returns key equilibrium values (computed with
    equilibrium.compute_equilibrium) under a housing supply constraint
    described in equilibrium.functions_dynamic and the technical
    documentation. This allows to account for inertia in formal private
    developers' actions and return more realistic simulated outputs.

    Parameters
    ----------
    t : int
        Year for which we want to run the function
    options : dict
        Dictionary of default options
    param : dict
        Dictionary of default parameters
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    initial_state_utility : ndarray(float64)
        Utility levels for each income group (4) at baseline year
        (2011)
    initial_state_error : ndarray(float64)
        Ratio (in %) of simulated number of households per income group over
        target population per income group at baseline year (2011)
    initial_state_households : ndarray(float64, ndim=3)
        Number of households per grid cell in each income group (4) and
        each housing type (4) at baseline year (2011)
    initial_state_households_housing_types : ndarray(float64, ndim=2)
        Number of households per grid cell in each housing type (4)
        at baseline year (2011)
    initial_state_housing_supply : ndarray(float64, ndim=2)
        Housing supply per unit of available land (in m² per km²)
        for each housing type (4) in each grid cell at baseline year (2011)
    initial_state_household_centers : ndarray(float64, ndim=2)
        Number of households per grid cell in each income group (4)
        at baseline year (2011)
    initial_state_average_income : ndarray(float64)
        Not an output of the model per se : it is just the average median
        income for each income group in the model (4), that may change
        over time
    initial_state_rent : ndarray(float64, ndim=2)
        Average annual rent (in rands) per grid cell for each housing type
        (4) at baseline year (2011)
    initial_state_dwelling_size : ndarray(float64, ndim=2)
        Average dwelling size (in m²) per grid cell in each housing
        type (4) at baseline year (2011)
    fraction_capital_destroyed : DataFrame
        Data frame of expected fractions of capital destroyed, for housing
        structures and contents in different housing types, in each
        grid cell (24,014)
    amenities : ndarray(float64)
        Normalized amenity index (relative to the mean) for each grid cell
        (24,014)
    housing_limit : Series
        Maximum housing supply (in m² per km²) in each grid cell (24,014)
    spline_estimate_RDP : interp1d
        Linear interpolation for the grid-level number of formal subsidized
        dwellings over the years (baseline year set at 0)
    spline_land_constraints : interp1d
        Linear interpolation for the grid-level overall land availability,
        (in %) over the years (baseline year set at 0)
    spline_land_backyard : interp1d
        Linear interpolation for the grid-level land availability (in %)
        for informal backyards over the years (baseline year set at 0)
    spline_land_RDP : interp1d
        Linear interpolation for the grid-level land availability (in %)
        for formal subsidized housing over the years (baseline year set at 0)
    spline_land_informal : interp1d
        Linear interpolation for the grid-level land availability (in %)
        for informal settlements over the years (baseline year set at 0)
    income_class_by_housing_type : DataFrame
        Set of dummies coding for housing market access (across 4 housing
        submarkets) for each income group (4, from poorest to richest)
    precalculated_transport : str
        Path for precalcuted transport inputs (intermediate outputs from
        commuting choice model)
    spline_RDP : interp1d
        Linear interpolation for the total number of formal subsidized
        dwellings over the years (baseline year set at 0)
    spline_agricultural_price : interp1d
        Linear interpolation for the agricultural land price (in rands)
        over the years (baseline year set at 0)
    spline_interest_rate : interp1d
        Linear interpolation for the interest rate (in %) over the years
    spline_population_income_distribution : interp1d
        Linear interpolation for total population per income group in the data
        (12) over the years (baseline year set at 0)
    spline_inflation : interp1d
        Linear interpolation for inflation rate (in base 100 relative to
        baseline year) over the years (baseline year set at 0)
    spline_income_distribution : interp1d
        Linear interpolation for median annual income (in rands) per income
        group in the data (12) over the years (baseline year set at 0)
    spline_population : interp1d
        Linear interpolation for total population over the years
        (baseline year set at 0)
    spline_income : interp1d
        Linear interpolation for overall average (annual) income over the years
        (baseline year set at 0), used to avoid money illusion in future
        simulations when computing the housing supply
        (see equilibrium.run_simulations)
    spline_minimum_housing_supply : interp1d
        Linear interpolation for minimum housing supply (in m²) over the years
        (baseline year set at 0)
    spline_fuel : interp1d
        Linear interpolation for fuel price (in rands per km)
        over the years (baseline year set at 0)
    income_baseline : DataFrame
        Table summarizing, for each income group in the data (12, including
        people out of employment), the number of households living in each
        endogenous housing type (3), their total number at baseline year (2011)
        in retrospect (2001), as well as the distribution of their average
        income (at baseline year)

    Returns
    -------
    simulation_households_center : ndarray(float64, ndim=3)
        Number of households per grid cell (24,014) in each income group (4)
        over all simulation years (30)
    simulation_households_housing_type : ndarray(float64, ndim=3)
        Number of households per grid cell (24,014) in each housing type (4)
        over all simulation years (30)
    simulation_dwelling_size : ndarray(float64, ndim=3)
        Average dwelling size (in m²) per grid cell (24,014) in each housing
        type (4) over all simulation years (30)
    simulation_rent : ndarray(float64, ndim=3)
        Average annual rent (in rands) per grid cell (24,014) for each housing
        type (4) over all simulation years (30)
    simulation_households : ndarray(float64, ndim=4)
        Number of households per grid cell (24,014) in each income group (4)
        and each housing type (4) at baseline year (2011) over all simulation
        years (30)
    simulation_error : ndarray(float64, ndim=2)
        Ratio (in %) of simulated number of households per income group over
        target population per income group over all simulation years (30)
    simulation_housing_supply : ndarray(float64, ndim=3)
        Housing supply per unit of available land (in m² per km²)
        for each housing type (4) in each grid cell (24,014) over all
        simulation years (30)
    simulation_utility : ndarray(float64, ndim=2)
        Utility levels for each income group (4) over all simulation years (30)
    simulation_deriv_housing : ndarray(float64, ndim=2)
        Difference between simulated next and current period values for housing
        supply per unit of available land (in m² per km²) per grid cell
        (24,014), for all simulation years (30).
    simulation_T : ndarray(float64)
        Years (relative to baseline set at 0) used for the simulations

    """
    #  Number of iterations per year
    freq_iter = param["iter_calc_lite"]
    #  Here, we are going to run separate simulations with and without
    #  dynamics, hence the need for a new parameter
    options["adjust_housing_init"] = copy.deepcopy(
        options["adjust_housing_supply"])
    options["adjust_housing_supply"] = 0
    #  Setting the timeline
    years_simulations = np.arange(
        t[0], t[len(t) - 1] + 1, (t[1] - t[0])/freq_iter)

    # Preallocating outputs
    simulation_dwelling_size = np.zeros((len(t), 4, len(grid.dist)))
    simulation_rent = np.zeros((len(t), 4, len(grid.dist)))
    simulation_households = np.zeros((len(t), 4, 4, len(grid.dist)))
    simulation_housing_supply = np.zeros((len(t), 4, len(grid.dist)))
    simulation_households_housing_type = np.zeros((len(t), 4, len(grid.dist)))
    simulation_households_center = np.zeros((len(t), 4, len(grid.dist)))
    simulation_error = np.zeros((len(t), 4))
    simulation_utility = np.zeros((len(t), 4))
    simulation_deriv_housing = np.zeros((len(t), len(grid.dist)))

    # Starting the simulation
    for index_iter in range(0, len(years_simulations)):

        print(index_iter)

        year_temp = copy.deepcopy(years_simulations[index_iter])
        stat_temp_housing_supply = copy.deepcopy(initial_state_housing_supply)
        stat_temp_rent = copy.deepcopy(initial_state_rent)

        if index_iter > 0:

            if index_iter == len(t):
                print('stop')

            # SIMULATION WITH EQUILIBRIUM HOUSING STOCK
            print('Simulation without constraint')
            options["adjust_housing_supply"] = 1

            # All that changes
            (average_income, households_per_income_class
             ) = eqdyn.compute_average_income(
                spline_population_income_distribution,
                spline_income_distribution, param, year_temp)
            income_net_of_commuting_costs = np.load(
                precalculated_transport + "GRID_incomeNetOfCommuting_"
                + str(int(year_temp)) + ".npy")
            (param["subsidized_structure_value"]
             ) = (param["subsidized_structure_value_ref"]
                  * (spline_inflation(year_temp) / spline_inflation(0)))
            (param["informal_structure_value"]
             ) = (param["informal_structure_value_ref"]
                  * (spline_inflation(year_temp) / spline_inflation(0)))
            mean_income = spline_income(year_temp)
            interest_rate = eqdyn.interpolate_interest_rate(
                spline_interest_rate, year_temp)
            population = spline_population(year_temp)
            total_RDP = spline_RDP(year_temp)
            minimum_housing_supply = spline_minimum_housing_supply(year_temp)
            number_properties_RDP = spline_estimate_RDP(year_temp)
            # Scale factor needs to move to avoid monetary illusion in the
            # model, e.g. housing supply should not change when currency
            # changes and all prices move: see technical documentation for
            # math formula
            construction_param = (
                (mean_income / param["income_year_reference"])
                ** (- param["coeff_b"]) * param["coeff_A"]
            )
            coeff_land = inpdt.import_coeff_land(
                spline_land_constraints, spline_land_backyard,
                spline_land_informal, spline_land_RDP, param, year_temp)
            agricultural_rent = inpprm.compute_agricultural_rent(
                spline_agricultural_price(year_temp), construction_param,
                interest_rate, param, options)

            # We compute a new static equilibrium for next period
            (tmpi_utility, tmpi_error, tmpi_simulated_jobs,
             tmpi_households_housing_types, tmpi_household_centers,
             tmpi_households, tmpi_dwelling_size, tmpi_housing_supply,
             tmpi_rent, tmpi_rent_matrix, tmpi_capital_land,
             tmpi_average_income, tmpi_limit_city) = eqcmp.compute_equilibrium(
                fraction_capital_destroyed, amenities, param, housing_limit,
                population, households_per_income_class, total_RDP, coeff_land,
                income_net_of_commuting_costs, grid, options,
                agricultural_rent, interest_rate, number_properties_RDP,
                average_income, mean_income, income_class_by_housing_type,
                minimum_housing_supply, construction_param, income_baseline)

            # Estimation of the housing supply difference between t and t+1
            # (only for formal private housing)
            deriv_housing_temp = eqdyn.evolution_housing_supply(
                housing_limit, param, years_simulations[index_iter],
                years_simulations[index_iter - 1], tmpi_housing_supply[0, :],
                stat_temp_housing_supply[0, :])
            # We update the initial housing parameter as it will give the
            # housing supply when developers do not adjust
            param["housing_in"] = stat_temp_housing_supply[0, :]
            + deriv_housing_temp

            # RUN A NEW SIMULATION WITH FIXED HOUSING
            # This allows to get a constrained "dynamic" equilibrium after
            # taking inertia and depreciation into account
            print('Simulation with constraint')
            options["adjust_housing_supply"] = 0
            (initial_state_utility, initial_state_error,
             initial_state_simulated_jobs,
             initial_state_households_housing_types,
             initial_state_household_centers, initial_state_households,
             initial_state_dwelling_size, initial_state_housing_supply,
             initial_state_rent, initial_state_rent_matrix,
             initial_state_capital_land, initial_state_average_income,
             initial_state_limit_city) = eqcmp.compute_equilibrium(
                fraction_capital_destroyed, amenities, param, housing_limit,
                population, households_per_income_class, total_RDP, coeff_land,
                income_net_of_commuting_costs, grid, options,
                agricultural_rent, interest_rate, number_properties_RDP,
                average_income, mean_income, income_class_by_housing_type,
                minimum_housing_supply, construction_param, income_baseline)

            stat_temp_deriv_housing = copy.deepcopy(deriv_housing_temp)

        # We initialize the output vector
        else:
            stat_temp_deriv_housing = np.zeros(len(stat_temp_rent[0, :]))

# The next condition was added to consider more than one simulation per year.
# This is not used in practice and was not fully coded, hence the function
# breaks down for frequencies bigger than one.

        if ((index_iter - 1) / param["iter_calc_lite"]
                - np.floor((index_iter - 1) / param["iter_calc_lite"])) == 0:
            # We retain the new constrained equilibrium with dynamic housing
            # supply as an output
            simulation_households_center[int(
                (index_iter - 1) / param["iter_calc_lite"] + 1), :, :
                    ] = copy.deepcopy(initial_state_household_centers)
            simulation_households_housing_type[int(
                (index_iter - 1) / param["iter_calc_lite"] + 1), :, :
                    ] = copy.deepcopy(initial_state_households_housing_types)
            simulation_dwelling_size[int(
                (index_iter - 1) / param["iter_calc_lite"] + 1), :, :
                    ] = copy.deepcopy(initial_state_dwelling_size)
            simulation_rent[
                int((index_iter - 1) / param["iter_calc_lite"] + 1), :, :
                    ] = copy.deepcopy(initial_state_rent)
            simulation_households[
                int((index_iter - 1) / param["iter_calc_lite"] + 1), :, :, :
                    ] = copy.deepcopy(initial_state_households)
            simulation_error[
                int((index_iter - 1) / param["iter_calc_lite"] + 1), :
                    ] = copy.deepcopy(initial_state_error)
            simulation_housing_supply[
                int((index_iter - 1) / param["iter_calc_lite"] + 1), :, :
                    ] = copy.deepcopy(initial_state_housing_supply)
            simulation_utility[
                int((index_iter - 1) / param["iter_calc_lite"] + 1), :
                    ] = copy.deepcopy(initial_state_utility)
            simulation_deriv_housing[
                int((index_iter - 1) / param["iter_calc_lite"] + 1), :
                    ] = copy.deepcopy(stat_temp_deriv_housing)

    # In case we have more than one simulation per year, we collapse timeline
    # to yearly
    if len(t) < len(years_simulations):
        T = copy.deepcopy(t)
    # Else, we just keep it as it is
    else:
        T = copy.deepcopy(years_simulations)
    # We retain this timeline as an output for our plots
    simulation_T = copy.deepcopy(T)

    # We also reinitialize parameter values
    options["adjust_housing_supply"] = copy.deepcopy(
        options["adjust_housing_init"])

    return (simulation_households_center, simulation_households_housing_type,
            simulation_dwelling_size, simulation_rent, simulation_households,
            simulation_error, simulation_housing_supply, simulation_utility,
            simulation_deriv_housing, simulation_T)
