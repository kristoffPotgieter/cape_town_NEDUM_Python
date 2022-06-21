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
                   spline_agricultural_rent, spline_interest_rate,
                   spline_population_income_distribution, spline_inflation,
                   spline_income_distribution, spline_population,
                   spline_income,
                   spline_minimum_housing_supply, spline_fuel, income_2011):
    """Run simulations over several years according to scenarios."""
    # Parameters and options of the scenario

    #  Nb of iterations per year
    freq_iter = param["iter_calc_lite"]
    #  Here, we are going to run separate simulations with and without
    #  dynamics, hence the need for a new parameter
    #  TODO: maybe do it before and plug this back in functions_solver
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
        # stat_temp_utility = copy.deepcopy(initial_state_utility)
        # Note that we actually need nothing more than the housing supply
        stat_temp_housing_supply = copy.deepcopy(initial_state_housing_supply)
        stat_temp_rent = copy.deepcopy(initial_state_rent)
        # stat_temp_average_income = copy.deepcopy(
        # initial_state_average_income)

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
            # income_mult = average_income / mean_income
            number_properties_RDP = spline_estimate_RDP(year_temp)
            # Scale factor needs to move to create monetary illusion in the
            # model, e.g. housing supply should not change when currency
            # changes and all prices move: this is where the formula comes
            # from
            construction_param = (
                (mean_income / param["income_year_reference"])
                ** (- param["coeff_b"]) * param["coeff_A"]
            )
            coeff_land = inpdt.import_coeff_land(
                spline_land_constraints, spline_land_backyard,
                spline_land_informal, spline_land_RDP, param, year_temp)

            agricultural_rent = inpprm.compute_agricultural_rent(
                spline_agricultural_rent(year_temp), construction_param,
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
                minimum_housing_supply, construction_param, income_2011)

            # Estimation of the derivation of housing supply between t and t+1
            # (only for formal housing)
            deriv_housing_temp = eqdyn.evolution_housing_supply(
                housing_limit, param, options, years_simulations[index_iter],
                years_simulations[index_iter - 1], tmpi_housing_supply[0, :],
                stat_temp_housing_supply[0, :])
            # We update the initial housing parameter as it will give the
            # housing supply when developers do not adjust
            # TODO: create a copy to plug back in functions_solver
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
                minimum_housing_supply, construction_param, income_2011)

            # stat_temp_utility = copy.deepcopy(tmpi_utility)
            stat_temp_deriv_housing = copy.deepcopy(deriv_housing_temp)

        # We initialize the derivation vector
        else:
            stat_temp_deriv_housing = np.zeros(len(stat_temp_rent[0, :]))

###

        # TODO: isn't it always the case? What happens else?
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

###

    # In case we have more than one simulation per year, we collapse timeline
    # to yearly
    if len(t) < len(years_simulations):
        T = copy.deepcopy(t)
    # Else, we just keep it as it is
    else:
        T = copy.deepcopy(years_simulations)

    # We retain this timeline as an output for our plots
    simulation_T = copy.deepcopy(T)
    # We reinitialize parameter
    options["adjust_housing_supply"] = copy.deepcopy(
        options["adjust_housing_init"])

    return (simulation_households_center, simulation_households_housing_type,
            simulation_dwelling_size, simulation_rent, simulation_households,
            simulation_error, simulation_housing_supply, simulation_utility,
            simulation_deriv_housing, simulation_T)
