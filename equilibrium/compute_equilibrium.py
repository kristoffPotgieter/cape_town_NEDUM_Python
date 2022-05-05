# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:21:21 2020.

@author: Charlotte Liotta
"""

import numpy as np
from tqdm import tqdm
import copy

import equilibrium.sub.compute_outputs as eqout


def compute_equilibrium(fraction_capital_destroyed, amenities, param,
                        housing_limit, population, households_per_income_class,
                        total_RDP, coeff_land, income_net_of_commuting_costs,
                        grid, options, agricultural_rent, interest_rate,
                        number_properties_RDP, average_income, mean_income,
                        income_class_by_housing_type, minimum_housing_supply,
                        construction_param):
    """Determine static equilibrium allocation from iterative algorithm."""
    # Adjust the population to remove the population in RDP
    #  We augment the number of households per income class to include RDP
    #  TODO: shouldn't we only rescale income groups eligible to formal
    #  backyarding?
    ratio = population / sum(households_per_income_class)
    households_per_income_class = households_per_income_class * ratio
    #  Considering that all RDP belong to the poorest, we remove them from here
    households_per_income_class[0] = np.max(
        households_per_income_class[0] - total_RDP, 0)

    # Shorten the grid
    #  We select pixels with constructible shares for all housing types > 0.01
    #  and a positive maximum income net of commuting costs across classes
    #  NB: we will have no output in other pixels (numeric simplification)
    selected_pixels = (
        (np.sum(coeff_land, 0) > 0.01).squeeze()
        & (np.nanmax(income_net_of_commuting_costs, 0) > 0)
        )
    coeff_land_full = copy.deepcopy(coeff_land)
    coeff_land = coeff_land[:, selected_pixels]
    grid_temp = copy.deepcopy(grid)
    grid = grid.iloc[selected_pixels, :]
    housing_limit = housing_limit[selected_pixels]
    # param["multi_proba_group"] = param["multi_proba_group"][
    # :, selected_pixels]
    income_net_of_commuting_costs = income_net_of_commuting_costs[
        :, selected_pixels]
    minimum_housing_supply = minimum_housing_supply[selected_pixels]
    housing_in = copy.deepcopy(param["housing_in"][selected_pixels])
    amenities = amenities[selected_pixels]
    fraction_capital_destroyed = fraction_capital_destroyed.iloc[
        selected_pixels, :]
    param_pockets = param["pockets"][selected_pixels]
    param_backyards_pockets = param["backyard_pockets"][selected_pixels]

    # Useful variables for the solver
    # (3 is because we have 3 types of housing in the solver)
    diff_utility = np.zeros((param["max_iter"], param["nb_of_income_classes"]))
    simulated_people_housing_types = np.zeros(
        (param["max_iter"], 3, len(grid.dist)))
    simulated_people = np.zeros((3, 4, len(grid.dist)))
    simulated_jobs = np.zeros(
        (param["max_iter"], 3, param["nb_of_income_classes"]))
    total_simulated_jobs = np.zeros(
        (param["max_iter"], param["nb_of_income_classes"]))
    rent_matrix = np.zeros((param["max_iter"], 3, len(grid.dist)))
    error_max_abs = np.zeros(param["max_iter"])
    error_max = np.zeros(param["max_iter"])
    error_mean = np.zeros(param["max_iter"])
    nb_error = np.zeros(param["max_iter"])
    error = np.zeros((param["max_iter"], param["nb_of_income_classes"]))
    housing_supply = np.empty((3, len(grid.dist)))
    dwelling_size = np.empty((3, len(grid.dist)))
    R_mat = np.empty((3, 4, len(grid.dist)))
    housing_supply[:] = np.nan
    dwelling_size[:] = np.nan
    R_mat[:] = np.nan

    # Initialisation solver
    utility = np.zeros((param["max_iter"], param["nb_of_income_classes"]))
    #  We take arbitrary utility levels, not too far from what we would expect,
    #  to make computation quicker
    utility[0, :] = np.array([1501, 4819, 16947, 79809])
    index_iteration = 0
    #  We need to apply some convergence factor to our error terms to make them
    #  converge in our optimization: the formula comes from trial and error,
    #  but the intuition is that we put more weight on the error for households
    #  with high relative income as it will magnify the effect on rents, hence
    #  housing supply and resulting population distribution

    #  TODO: run simulations with different convergence parameters to ensure
    #  that we do not get stuck in some local optimum (unlikely as distribution
    #  of households is monotonous wrt utility changes)
    param["convergence_factor"] = (
        0.02 * (np.nanmean(average_income) / mean_income) ** 0.4
        )  # 0.045

    # Compute outputs solver - First iteration (for each housing type, no RDP)
    #  Formal housing
    (simulated_jobs[index_iteration, 0, :], rent_matrix[index_iteration, 0, :],
     simulated_people_housing_types[index_iteration, 0, :],
     simulated_people[0, :, :], housing_supply[0, :], dwelling_size[0, :],
     R_mat[0, :, :]) = eqout.compute_outputs(
         'formal', utility[index_iteration, :], amenities, param,
         income_net_of_commuting_costs, fraction_capital_destroyed, grid,
         income_class_by_housing_type, options, housing_limit,
         agricultural_rent, interest_rate, coeff_land[0, :],
         minimum_housing_supply, construction_param, housing_in, param_pockets,
         param_backyards_pockets
         )
    #  Backyard housing
    (simulated_jobs[index_iteration, 1, :], rent_matrix[index_iteration, 1, :],
     simulated_people_housing_types[index_iteration, 1, :],
     simulated_people[1, :, :], housing_supply[1, :], dwelling_size[1, :],
     R_mat[1, :, :]) = eqout.compute_outputs(
         'backyard', utility[index_iteration, :], amenities, param,
         income_net_of_commuting_costs, fraction_capital_destroyed, grid,
         income_class_by_housing_type, options, housing_limit,
         agricultural_rent, interest_rate, coeff_land[1, :],
         minimum_housing_supply, construction_param, housing_in, param_pockets,
         param_backyards_pockets
         )
    #  Informal housing
    (simulated_jobs[index_iteration, 2, :], rent_matrix[index_iteration, 2, :],
     simulated_people_housing_types[index_iteration, 2, :],
     simulated_people[2, :, :], housing_supply[2, :], dwelling_size[2, :],
     R_mat[2, :, :]) = eqout.compute_outputs(
         'informal', utility[index_iteration, :], amenities, param,
         income_net_of_commuting_costs, fraction_capital_destroyed, grid,
         income_class_by_housing_type, options, housing_limit,
         agricultural_rent, interest_rate, coeff_land[2, :],
         minimum_housing_supply, construction_param, housing_in, param_pockets,
         param_backyards_pockets
         )

    # Compute error and adjust utility

    #  We first update the first iteration of output vector with the sum of
    #  simulated people per income group across housing types (no RDP)
    total_simulated_jobs[index_iteration, :] = np.sum(
        simulated_jobs[index_iteration, :, :], 0)

    #  diff_utility will be used to adjust the utility levels
    #  Note that optimization is made to stick to households_per_income_class,
    #  which does not include people living in RDP (as we take this as
    #  exogenous), see equilibrium condition (i)

    #  We compare total population for each income group obtained from
    #  equilibrium condition (total_simulated_jobs) with target population
    #  allocation (households_per_income_class)

    #  We arbitrarily set a strictly positive minimum utility level at 10
    #  (as utility will be adjusted multiplicatively, we do not want to break
    #  the model with zero terms)
    diff_utility[index_iteration, :] = np.log(
        (total_simulated_jobs[index_iteration, :] + 10)
        / (households_per_income_class + 10))
    diff_utility[index_iteration, :] = (
        diff_utility[index_iteration, :] * param["convergence_factor"])
    (diff_utility[index_iteration, diff_utility[index_iteration, :] > 0]
     ) = (diff_utility[index_iteration, diff_utility[index_iteration, :] > 0]
          * 1.1)

    # Difference with reality
    error[index_iteration, :] = (total_simulated_jobs[index_iteration, :]
                                 / households_per_income_class - 1) * 100
    #  This is the parameter of interest for optimization
    error_max_abs[index_iteration] = np.nanmax(
        np.abs(total_simulated_jobs[index_iteration, :]
               / households_per_income_class - 1))
    error_max[index_iteration] = -1
    error_mean[index_iteration] = np.nanmean(
        np.abs(total_simulated_jobs[index_iteration, :]
               / (households_per_income_class + 0.001) - 1))
    nb_error[index_iteration] = np.nansum(
        np.abs(total_simulated_jobs[index_iteration, :]
               / households_per_income_class - 1) > param["precision"])

    # Iteration (no RDP)
    # with alive_bar(param["max_iter"],title='compute equilibrium') as bar:
    # We use a progression bar
    with tqdm(total=param["max_iter"],
              desc="stops when error_max_abs <" + str(param["precision"])
              ) as pbar:
        while ((index_iteration < param["max_iter"] - 1)
               & (error_max_abs[index_iteration] > param["precision"])):

            # Adjust parameters to how close we are from the objective
            index_iteration = index_iteration + 1
            # When population is overestimated, we augment utility to reduce
            # population (cf. population constraint from standard model)
            utility[index_iteration, :] = np.exp(
                np.log(utility[index_iteration - 1, :])
                + diff_utility[index_iteration - 1, :])
            # This is a precaution as utility cannot be negative
            utility[index_iteration, utility[index_iteration, :] < 0] = 10

            # We augment the convergence factor at each iteration in propotion
            # with the estimation error to augment the importance of later
            # compared to earlier errors (as algorithm should improve across
            # iterations)

            # NB: we assume the minimum error is 100 not to break model with
            # zeros
            convergence_factor = (
                param["convergence_factor"] / (
                    1 + 0.5 * np.abs((
                        total_simulated_jobs[index_iteration, :] + 100
                        ) / (households_per_income_class + 100) - 1))
                )

            # At the same time, we also reduce it while time passes, not to
            # demand too much of the algorithm and to help convergence
            convergence_factor = (
                convergence_factor
                * (1 - 0.6 * index_iteration / param["max_iter"])
                )

            # Now, we do the same as in the initalization phase

            # Compute outputs solver - first iteration

            #  Formal housing
            (simulated_jobs[index_iteration, 0, :],
             rent_matrix[index_iteration, 0, :],
             simulated_people_housing_types[index_iteration, 0, :],
             simulated_people[0, :, :],
             housing_supply[0, :],
             dwelling_size[0, :],
             R_mat[0, :, :]) = eqout.compute_outputs(
                 'formal', utility[index_iteration, :], amenities, param,
                 income_net_of_commuting_costs, fraction_capital_destroyed,
                 grid, income_class_by_housing_type, options, housing_limit,
                 agricultural_rent, interest_rate, coeff_land[0, :],
                 minimum_housing_supply, construction_param, housing_in,
                 param_pockets, param_backyards_pockets
                 )

            #  Backyard housing
            (simulated_jobs[index_iteration, 1, :],
             rent_matrix[index_iteration, 1, :],
             simulated_people_housing_types[index_iteration, 1, :],
             simulated_people[1, :, :],
             housing_supply[1, :],
             dwelling_size[1, :],
             R_mat[1, :, :]) = eqout.compute_outputs(
                 'backyard', utility[index_iteration, :], amenities, param,
                 income_net_of_commuting_costs, fraction_capital_destroyed,
                 grid, income_class_by_housing_type, options, housing_limit,
                 agricultural_rent, interest_rate, coeff_land[1, :],
                 minimum_housing_supply, construction_param, housing_in,
                 param_pockets, param_backyards_pockets
                 )

            #  Informal housing
            (simulated_jobs[index_iteration, 2, :],
             rent_matrix[index_iteration, 2, :],
             simulated_people_housing_types[index_iteration, 2, :],
             simulated_people[2, :, :],
             housing_supply[2, :],
             dwelling_size[2, :],
             R_mat[2, :, :]) = eqout.compute_outputs(
                 'informal', utility[index_iteration, :], amenities, param,
                 income_net_of_commuting_costs, fraction_capital_destroyed,
                 grid, income_class_by_housing_type, options, housing_limit,
                 agricultural_rent, interest_rate, coeff_land[2, :],
                 minimum_housing_supply, construction_param, housing_in,
                 param_pockets, param_backyards_pockets
                 )

            # Compute error and adjust utility
            total_simulated_jobs[index_iteration, :] = np.sum(
                simulated_jobs[index_iteration, :, :], 0)

            # diff_utility will be used to adjust the utility levels
            diff_utility[index_iteration, :] = np.log(
                (total_simulated_jobs[index_iteration, :] + 10)
                / (households_per_income_class + 10))
            diff_utility[index_iteration, :] = (
                diff_utility[index_iteration, :] * convergence_factor)
            diff_utility[
                index_iteration, diff_utility[index_iteration, :] > 0] = (
                    diff_utility[index_iteration,
                                 diff_utility[index_iteration, :] > 0] * 1.1
                    )

            # Variables to display
            error[index_iteration, :] = (
                total_simulated_jobs[index_iteration, :]
                / households_per_income_class - 1) * 100
            error_max_abs[index_iteration] = np.max(np.abs(
                total_simulated_jobs[index_iteration,
                                     households_per_income_class != 0]
                / households_per_income_class - 1))
            m = np.argmax(np.abs(total_simulated_jobs[index_iteration, :]
                                 / households_per_income_class - 1))
            erreur_temp = (total_simulated_jobs[index_iteration, :]
                           / households_per_income_class - 1)
            error_max[index_iteration] = erreur_temp[m]
            error_mean[index_iteration] = np.mean(np.abs(
                total_simulated_jobs[index_iteration, :]
                / (households_per_income_class + 0.001) - 1))
            nb_error[index_iteration] = np.sum(np.abs(
                total_simulated_jobs[index_iteration, :]
                / households_per_income_class - 1) > param["precision"])
            pbar.set_postfix({'error_max_abs': error_max_abs[index_iteration]})
            pbar.update()

    # RDP houses
    #  We correct output coming from data_RDP with more reliable estimations
    #  from Claus
    households_RDP = (number_properties_RDP * total_RDP
                      / sum(number_properties_RDP))
    #  Share of housing (no backyard) in RDP surface (with land in km²)
    construction_RDP = np.matlib.repmat(
        param["RDP_size"] / (param["RDP_size"] + param["backyard_size"]),
        1, len(grid_temp.dist))
    #  RDP dwelling size
    dwelling_size_RDP = np.matlib.repmat(
        param["RDP_size"], 1, len(grid_temp.dist))

    # We fill the vector for each housing type
    simulated_people_with_RDP = np.zeros((4, 4, len(grid_temp.dist)))
    simulated_people_with_RDP[0, :, selected_pixels] = np.transpose(
        simulated_people[0, :, :, ])
    simulated_people_with_RDP[1, :, selected_pixels] = np.transpose(
        simulated_people[1, :, :, ])
    simulated_people_with_RDP[2, :, selected_pixels] = np.transpose(
        simulated_people[2, :, :, ])
    simulated_people_with_RDP[3, 0, :] = households_RDP

    # Outputs of the solver

    initial_state_error = error[index_iteration, :]
    #  Note that this does not contain RDP
    initial_state_simulated_jobs = simulated_jobs[index_iteration, :, :]
    #  We sum across income groups (axis=1)
    initial_state_households_housing_types = np.sum(simulated_people_with_RDP,
                                                    1)
    #  We sum across housing types (axis=0)
    initial_state_household_centers = np.sum(simulated_people_with_RDP, 0)
    #  We keep both dimensions
    initial_state_households = simulated_people_with_RDP

    # Housing stock and dwelling size (fill with RDP)
    housing_supply_export = np.zeros((3, len(grid_temp.dist)))
    dwelling_size_export = np.zeros((3, len(grid_temp.dist)))
    housing_supply_export[:, selected_pixels] = housing_supply
    dwelling_size_export[:, selected_pixels] = copy.deepcopy(dwelling_size)
    dwelling_size_export[dwelling_size_export <= 0] = np.nan
    # TODO: See equilibrium condition (iv)
    # TODO: choose between right and original specification
    # housing_supply_RDP = (
    #     construction_RDP * dwelling_size_RDP * households_RDP
    #     / (coeff_land_full[3, :] * 0.25)  # * 1000000
    #     )
    # housing_supply_RDP[np.isnan(housing_supply_RDP)] = 0
    housing_supply_RDP = (
        construction_RDP * 1000000
        )
    initial_state_dwelling_size = np.vstack(
        [dwelling_size_export, dwelling_size_RDP])
    # Note that RDP housing supply per unit of land has nothing to do with
    # backyard housing supply per unit of land
    initial_state_housing_supply = np.vstack(
        [housing_supply_export, housing_supply_RDP]
        )

    # Rents (HHs in RDP pay a rent of 0)
    rent_temp = copy.deepcopy(rent_matrix[index_iteration, :, :])
    rent_export = np.zeros((3, len(grid_temp.dist)))
    rent_export[:, selected_pixels] = copy.deepcopy(rent_temp)
    rent_export[:, selected_pixels == 0] = np.nan
    initial_state_rent = np.vstack(
        [rent_export, np.zeros(len(grid_temp.dist))])
    rent_matrix_export = np.zeros(
        (3, len(average_income), len(grid_temp.dist)))
    rent_matrix_export[:, :, selected_pixels] = copy.deepcopy(R_mat)
    rent_matrix_export[:, :, selected_pixels == 0] = np.nan
    initial_state_rent_matrix = copy.deepcopy(rent_matrix_export)

    # Other outputs
    #  See research note, p.10 (Cobb-Douglas)
    initial_state_capital_land = ((housing_supply / (construction_param))
                                  ** (1 / param["coeff_b"]))
    #  NB: this is not an output of the model
    initial_state_average_income = copy.deepcopy(average_income)
    initial_state_limit_city = [initial_state_households > 1]
    initial_state_utility = utility[index_iteration, :]

    return (initial_state_utility,
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
            initial_state_limit_city)
