# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:23:16 2020

@author: Charlotte Liotta
"""

import copy
import numpy as np

from equilibrium.functions_dynamic import *
from equilibrium.compute_equilibrium import *
from inputs.data import *

def run_simulation(t, options, income_2011, param, grid, initial_state_utility, initial_state_error, initial_state_households, initial_state_households_housing_types, initial_state_housing_supply, initial_state_household_centers, initial_state_average_income, initial_state_rent, initial_state_dwelling_size, fraction_capital_destroyed, amenities, housing_limit, spline_estimate_RDP, spline_land_constraints, spline_land_backyard, spline_land_RDP, spline_land_informal, income_class_by_housing_type, path_scenarios, precalculated_transport, spline_RDP):
    
    #Parameters and options of the scenario
    freq_iter = 1 #One iteration every year
    options["adjust_housing_init"] = copy.deepcopy(options["adjust_housing_supply"])
    options["adjust_housing_supply"] = 0
    years_simulations = np.arange(t[0], t[len(t) - 1] + 1, (t[1] - t[0])/freq_iter)

    #Preallocating outputs
    simulation_dwelling_size = np.zeros((len(t), 4, len(grid.dist)))
    simulation_rent = np.zeros((len(t), 4, len(grid.dist)))
    simulation_households = np.zeros((len(t), 4, 4, len(grid.dist)))
    simulation_housing_supply = np.zeros((len(t), 4, len(grid.dist)))
    simulation_households_housing_type = np.zeros((len(t), 4, len(grid.dist)))
    simulation_households_center = np.zeros((len(t), 4, len(grid.dist)))
    simulation_error = np.zeros((len(t), 4))
    simulation_utility = np.zeros((len(t), 4))
    simulation_deriv_housing = np.zeros((len(t), len(grid.dist)))
    
    #Import Scenarios
    spline_agricultural_rent, spline_interest_rate, spline_population_income_distribution, spline_inflation, spline_income_distribution, spline_population, spline_interest_rate, spline_income, spline_minimum_housing_supply, spline_fuel = import_scenarios(income_2011, param, grid, path_scenarios)

    for index_iter in range(0, len(years_simulations)):
    
        print(index_iter)
        
        year_temp = copy.deepcopy(years_simulations[index_iter])
        stat_temp_utility = copy.deepcopy(initial_state_utility)
        stat_temp_housing_supply = copy.deepcopy(initial_state_housing_supply)
        stat_temp_rent = copy.deepcopy(initial_state_rent)
        stat_temp_average_income = copy.deepcopy(initial_state_average_income)
        
        if index_iter > 0:
                
            if index_iter == len(t):
                print('stop')
            
            #Simulation with equilibrium housing stock
            print('Simulation without constraint')
            options["adjust_housing_supply"] = 1
        
            #Tout ce qui évolue
            average_income, households_per_income_class = compute_average_income(spline_population_income_distribution, spline_income_distribution, param, year_temp)
            income_net_of_commuting_costs = np.load(precalculated_transport + "incomeNetOfCommuting_" + str(int(year_temp)) + ".npy")
            param["subsidized_structure_value"] = param["subsidized_structure_value_ref"] * (spline_inflation(year_temp) / spline_inflation(0))
            param["informal_structure_value"] = param["informal_structure_value_ref"] * (spline_inflation(year_temp) / spline_inflation(0))
            mean_income = spline_income(year_temp)       
            interest_rate = interpolate_interest_rate(spline_interest_rate, year_temp)
            population = spline_population(year_temp)
            total_RDP = spline_RDP(year_temp)
            minimum_housing_supply = spline_minimum_housing_supply(year_temp)
            income_mult = average_income / mean_income
            number_properties_RDP = spline_estimate_RDP(year_temp)        
            # Why not just scale factor?
            construction_param = (mean_income / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"]        
            coeff_land = import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, year_temp)
            agricultural_rent = spline_agricultural_rent(year_temp) ** (param["coeff_a"]) * (interest_rate) / (construction_param * param["coeff_b"] ** param["coeff_b"])
            #  TODO: we had to remove fraction_capital_destroyed as a parameter: does it make sense?
            tmpi_utility, tmpi_error, tmpi_simulated_jobs, tmpi_households_housing_types, tmpi_household_centers, tmpi_households, tmpi_dwelling_size, tmpi_housing_supply, tmpi_rent, tmpi_rent_matrix, tmpi_capital_land, tmpi_average_income, tmpi_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, construction_param)

            #Estimation of the derivation of housing supply between t and t+1
            deriv_housing_temp = evolution_housing_supply(housing_limit, param, options, years_simulations[index_iter], years_simulations[index_iter - 1], tmpi_housing_supply[0, :], stat_temp_housing_supply[0, :])
            param["housing_in"] = stat_temp_housing_supply[0,:] + deriv_housing_temp
            
            #Run a new simulation with fixed housing
            print('Simulation with constraint')
            options["adjust_housing_supply"] = 0   
            initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, construction_param)

            #Ro de la simulation libre
            stat_temp_utility = copy.deepcopy(tmpi_utility)
            stat_temp_deriv_housing = copy.deepcopy(deriv_housing_temp)

        else:
        
            stat_temp_deriv_housing = np.zeros(len(stat_temp_rent[0,:]))
            
        if (index_iter - 1) / param["iter_calc_lite"] - np.floor((index_iter - 1) / param["iter_calc_lite"]) == 0:

            simulation_households_center[int((index_iter - 1) / param["iter_calc_lite"] + 1), :, :] = copy.deepcopy(initial_state_household_centers)
            simulation_households_housing_type[int((index_iter - 1) / param["iter_calc_lite"] + 1), :, :] = copy.deepcopy(initial_state_households_housing_types)
            simulation_dwelling_size[int((index_iter - 1) / param["iter_calc_lite"] + 1), :, :] = copy.deepcopy(initial_state_dwelling_size)
            simulation_rent[int((index_iter - 1) / param["iter_calc_lite"] + 1), :, :] = copy.deepcopy(initial_state_rent)
            simulation_households[int((index_iter - 1) / param["iter_calc_lite"] + 1), :, :, :] = copy.deepcopy(initial_state_households)
            simulation_error[int((index_iter - 1) / param["iter_calc_lite"] + 1), :] = copy.deepcopy(initial_state_error)
            simulation_housing_supply[int((index_iter - 1) / param["iter_calc_lite"] + 1), :, :] = copy.deepcopy(initial_state_housing_supply)
            simulation_utility[int((index_iter - 1) / param["iter_calc_lite"] + 1), :] = copy.deepcopy(initial_state_utility)
            simulation_deriv_housing[int((index_iter - 1) / param["iter_calc_lite"] + 1), :] = copy.deepcopy(stat_temp_deriv_housing)

    if len(t) < len(years_simulations):
        T = copy.deepcopy(t)
    else:
        T = copy.deepcopy(years_simulations)

    simulation_T = copy.deepcopy(T)
    options["adjust_housing_supply"] = copy.deepcopy(options["adjust_housing_init"])
    
    return simulation_households_center, simulation_households_housing_type, simulation_dwelling_size, simulation_rent, simulation_households, simulation_error, simulation_housing_supply, simulation_utility, simulation_deriv_housing, simulation_T