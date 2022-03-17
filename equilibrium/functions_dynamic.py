# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:16:26 2020.

@author: Charlotte Liotta
"""

import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import numpy.matlib


def import_scenarios(income_2011, param, grid, path_scenarios):
    """Return linear regression splines for various scenarios."""
    # Import Scenarios
    scenario_income_distribution = pd.read_csv(
        path_scenarios + 'Scenario_inc_distrib_2.csv', sep=';')
    scenario_population = pd.read_csv(
        path_scenarios + 'Scenario_pop_20201209.csv', sep=';')
    scenario_inflation = pd.read_csv(
        path_scenarios + 'Scenario_inflation_1.csv', sep=';')
    scenario_interest_rate = pd.read_csv(
        path_scenarios + 'Scenario_interest_rate_1.csv', sep=';')
    scenario_price_fuel = pd.read_csv(
        path_scenarios + 'Scenario_price_fuel_1.csv', sep=';')

    # Spline for population by income group in raw data
    spline_population_income_distribution = interp1d(
        np.array([2001, 2011, 2040]) - param["baseline_year"],
        np.transpose([income_2011.Households_nb_2001,
                      income_2011.Households_nb,
                      scenario_income_distribution.Households_nb_2040]),
        'linear')

    # Spline for inflation
    spline_inflation = interp1d(
        scenario_inflation.Year_infla[
            ~np.isnan(scenario_inflation.inflation_base_2010)
            ] - param["baseline_year"],
        scenario_inflation.inflation_base_2010[
            ~np.isnan(scenario_inflation.inflation_base_2010)],
        'linear')

    # Spline for median income by income group in raw data
    #  We initialize the vector for years 1996, 2001, 2011, and 2040
    income_distribution = np.array(
        [income_2011.INC_med, income_2011.INC_med, income_2011.INC_med,
         scenario_income_distribution.INC_med_2040]
        )
    #  After 2011, for each income group, we multiply the baseline by inflation
    #  growth rate over the period
    (income_distribution[scenario_population.Year_pop > 2011, :]
     ) = (income_distribution[scenario_population.Year_pop > 2011, :]
          * np.matlib.repmat(spline_inflation(scenario_population.Year_pop[
              scenario_population.Year_pop > 2011] - param["baseline_year"])
              / spline_inflation(2011 - param["baseline_year"]),
              1,
              income_distribution.shape[1])
          )
    #  Then we get the spline
    spline_income_distribution = interp1d(
        scenario_population.Year_pop[~np.isnan(scenario_population.Year_pop)]
        - param["baseline_year"],
        np.transpose(
            income_distribution[~np.isnan(scenario_population.Year_pop), :]),
        'linear')

    # Spline for overall population
    spline_population = interp1d(
        scenario_population.Year_pop[~np.isnan(scenario_population.HH_total)]
        - param["baseline_year"],
        scenario_population.HH_total[~np.isnan(scenario_population.HH_total)],
        'linear')

    # Spline for real interest rate
    spline_interest_rate = interp1d(
        scenario_interest_rate.Year_interest_rate[
            ~np.isnan(scenario_interest_rate.real_interest_rate)]
        - param["baseline_year"],
        scenario_interest_rate.real_interest_rate[
            ~np.isnan(scenario_interest_rate.real_interest_rate)],
        'linear')

    # Spline for RDP population
    #  Note that this is inconsistent with inpdt.import_land_use! Clarify...
    RDP_2011 = 2.2666e+05
    RDP_2001 = 1.1718e+05
    spline_RDP = interp1d(
        [2001 - param["baseline_year"],
         2011 - param["baseline_year"],
         2018 - param["baseline_year"],
         2041 - param["baseline_year"]],
        [RDP_2001, RDP_2011, RDP_2011 + 7*5000,
         RDP_2011 + 7*5000 + 23 * param["future_rate_public_housing"]],
        'linear')

    # Spline for minimum housing_supply per pixel (always zero?)
    minimum_housing_2011 = param["minimum_housing_supply"]
    spline_minimum_housing_supply = interp1d(
        np.array([2001, 2011, 2100])
        - param["baseline_year"],
        np.transpose(
            [np.zeros(len(grid.dist)), minimum_housing_2011,
             minimum_housing_2011])
        )

    # Spline for agricultural rent
    #  We get missing value for 2040 by accounting for inflation
    agricultural_rent_2040 = (param["agricultural_rent_2011"]
                              * spline_inflation(2040 - param["baseline_year"])
                              / spline_inflation(2011 - param["baseline_year"])
                              )
    #  Then we get the spline
    spline_agricultural_rent = interp1d(
        [2001 - param["baseline_year"],
         2011 - param["baseline_year"],
         2040 - param["baseline_year"]],
        [param["agricultural_rent_2001"],
         param["agricultural_rent_2011"],
         agricultural_rent_2040],
        'linear')

    # Spline for fuel prices (in 10*cm^3?)
    spline_fuel = interp1d(
        scenario_price_fuel.Year_fuel[
            ~np.isnan(scenario_price_fuel.price_fuel)]
        - param["baseline_year"],
        scenario_price_fuel.price_fuel[
            ~np.isnan(scenario_price_fuel.price_fuel)]
        / 100,
        'linear')

    # Spline for overall average income
    #  We first get initial values
    average_income_2001 = np.sum(
        income_2011.Households_nb_2001 * income_2011.INC_med
        ) / sum(income_2011.Households_nb_2001)
    average_income_2011 = np.sum(
        income_2011.Households_nb * income_2011.INC_med
        ) / sum(income_2011.Households_nb)
    average_income_2040 = np.sum(
        scenario_income_distribution.Households_nb_2040
        * scenario_income_distribution.INC_med_2040
        ) / sum(scenario_income_distribution.Households_nb_2040)
    #  Then we set the time frame according to inflation schedule
    year_inc = scenario_inflation.Year_infla[
        ~np.isnan(scenario_inflation.inflation_base_2010)]
    year_inc = year_inc[(year_inc > 2000) & (year_inc < 2041)]
    #  We get initial spline (not taking inflation into account)
    inc_year_noinfla = interp1d(
        np.array([2001, 2011, 2040]) - param["baseline_year"],
        [average_income_2001, average_income_2011, average_income_2040],
        'linear')
    #  We stock it into an array with the right number of periods
    inc_ref = inc_year_noinfla(
        year_inc - param["baseline_year"]
        )
    #  We get a schedule for inflation growth rates with respect to baseline
    noinfla_ref = np.ones(year_inc[(year_inc <= param["baseline_year"])].size)
    infla_ref = spline_inflation(
        year_inc[(year_inc > param["baseline_year"])] - param["baseline_year"]
        ) / spline_inflation(0)
    infla_schedule = np.append(noinfla_ref, infla_ref)
    #  Then we correct output from initial spline with inflation growth rates
    inc_year_infla = inc_ref * infla_schedule
    #  And we get the final spline
    spline_income = interp1d(
        year_inc - param["baseline_year"], inc_year_infla, 'linear')

    return (spline_agricultural_rent, spline_interest_rate, spline_RDP,
            spline_population_income_distribution, spline_inflation,
            spline_income_distribution, spline_population,
            spline_interest_rate, spline_income, spline_minimum_housing_supply,
            spline_fuel)

###

def compute_average_income(spline_population_income_distribution, spline_income_distribution, param, t):

    total_bracket = spline_population_income_distribution(t)
    avg_income_bracket = spline_income_distribution(t)

    avg_income_group = np.zeros(4)
    total_group = np.zeros(4)

    for j in range(0, 4):
        total_group[j] = sum(total_bracket[param["income_distribution"] == j + 1])
        avg_income_group[j] = sum(avg_income_bracket[param["income_distribution"] == j + 1] * total_bracket[param["income_distribution"] == j + 1]) / total_group[j]

    return avg_income_group, total_group

def interpolate_interest_rate(spline_interest_rate, t):
    nb_years = 3
    interest_rate_n_years = spline_interest_rate(np.arange(t - nb_years, t))
    interest_rate_n_years[interest_rate_n_years < 0] = np.nan
    return np.nanmean(interest_rate_n_years)/100

def evolution_housing_supply(housing_limit, param, option, t1, t0, housing_supply_1, housing_supply_0):
    
    #New housing supply (accounting for inertia and depreciation w/ time)
    if t1 - t0 > 0:
        diff_housing = (housing_supply_1 - housing_supply_0) * (housing_supply_1 > housing_supply_0) * (t1 - t0) / param["time_invest_housing"] - housing_supply_0 * (t1 - t0)  / param["time_depreciation_buildings"]
    else:
        diff_housing = (housing_supply_1 - housing_supply_0) * (housing_supply_1 < housing_supply_0) * (t1 - t0) / param["time_invest_housing"] - housing_supply_0 * (t1 - t0)  / param["time_depreciation_buildings"]

    housing_supply_target = housing_supply_0 + diff_housing

    #Housing height is limited by potential regulations
    housing_supply_target = np.minimum(housing_supply_target, housing_limit)
    minimum_housing_supply_interp = interp1d(np.array([2001, 2011, 2100]) - param["baseline_year"], np.transpose([np.zeros(len(param["minimum_housing_supply"])), param["minimum_housing_supply"], param["minimum_housing_supply"]]))
    minimum_housing_supply_interp = minimum_housing_supply_interp(t1)                                                                                       
    housing_supply_target = np.maximum(housing_supply_target, minimum_housing_supply_interp)

    return housing_supply_target - housing_supply_0
