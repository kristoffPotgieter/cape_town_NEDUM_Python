# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:16:26 2020.

@author: Charlotte Liotta
"""

import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import numpy.matlib


def import_scenarios(income_baseline, param, grid, path_scenarios,
                     options):
    """
    Define linear interpolations for time-moving exogenous variables.

    Parameters
    ----------
    income_baseline : DataFrame
        Table summarizing, for each income group in the data (12, including
        people out of employment), the number of households living in each
        endogenous housing type (3), their total number at baseline year (2011)
        in retrospect (2001), as well as the distribution of their average
        income (at baseline year)
    param : dict
        Dictionary of default parameters
    grid : DataFrame
        Table yielding, for each grid cell (24,014), its x and y
        (centroid) coordinates, and its distance (in km) to the city centre
    path_scenarios : str
        Path towards raw scenarios used for time-moving exogenous variables
    options : dict
        Dictionary of default options

    Returns
    -------
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

    """
    # Import Scenarios
    if options["inc_ineq_scenario"] == 2:
        scenario_income_distribution = pd.read_csv(
            path_scenarios + 'Scenario_inc_distrib_2.csv', sep=';')
    elif options["inc_ineq_scenario"] == 1:
        scenario_income_distribution = pd.read_csv(
            path_scenarios + 'Scenario_inc_distrib_1.csv', sep=';')
    elif options["inc_ineq_scenario"] == 3:
        scenario_income_distribution = pd.read_csv(
            path_scenarios + 'Scenario_inc_distrib_3.csv', sep=';')

    if options["pop_growth_scenario"] == 4:
        scenario_population = pd.read_csv(
            path_scenarios + 'Scenario_pop_20201209.csv', sep=';')
    elif options["pop_growth_scenario"] == 3:
        scenario_population = pd.read_csv(
            path_scenarios + 'Scenario_pop_3.csv', sep=';')
    elif options["pop_growth_scenario"] == 2:
        scenario_population = pd.read_csv(
            path_scenarios + 'Scenario_pop_2.csv', sep=';')
    elif options["pop_growth_scenario"] == 1:
        scenario_population = pd.read_csv(
            path_scenarios + 'Scenario_pop_1.csv', sep=';')
    scenario_inflation = pd.read_csv(
        path_scenarios + 'Scenario_inflation_1.csv', sep=';')
    scenario_interest_rate = pd.read_csv(
        path_scenarios + 'Scenario_interest_rate_1.csv', sep=';')
    if options["fuel_price_scenario"] == 2:
        scenario_price_fuel = pd.read_csv(
            path_scenarios + 'Scenario_price_fuel_2.csv', sep=';')
    elif options["fuel_price_scenario"] == 1:
        scenario_price_fuel = pd.read_csv(
            path_scenarios + 'Scenario_price_fuel_1.csv', sep=',')
    elif options["fuel_price_scenario"] == 3:
        scenario_price_fuel = pd.read_csv(
            path_scenarios + 'Scenario_price_fuel_3.csv', sep=',')

    # Spline for population by income group in raw data
    spline_population_income_distribution = interp1d(
        np.array([2001, 2011, 2040]) - param["baseline_year"],
        np.transpose([income_baseline.Households_nb_2001,
                      income_baseline.Households_nb,
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
        [income_baseline.INC_med, income_baseline.INC_med,
         income_baseline.INC_med, scenario_income_distribution.INC_med_2040]
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

    # Spline for minimum housing_supply per pixel
    minimum_housing_2011 = param["minimum_housing_supply"]
    spline_minimum_housing_supply = interp1d(
        np.array([2001, 2011, 2100])
        - param["baseline_year"],
        np.transpose(
            [np.zeros(len(grid.dist)), minimum_housing_2011,
             minimum_housing_2011])
        )

    # Spline for agricultural land price
    #  We get missing value for 2040 by accounting for inflation
    agricultural_price_long_fut = (
        param["agricultural_price_baseline"]
        * spline_inflation(2040 - param["baseline_year"])
        / spline_inflation(2011 - param["baseline_year"])
        )
    #  Then we get the spline
    spline_agricultural_price = interp1d(
        [2001 - param["baseline_year"],
         2011 - param["baseline_year"],
         2040 - param["baseline_year"]],
        [param["agricultural_price_retrospect"],
         param["agricultural_price_baseline"],
         agricultural_price_long_fut],
        'linear')

    # Spline for fuel prices (in rands per km)
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
    average_income_retrospect = np.sum(
        income_baseline.Households_nb_2001 * income_baseline.INC_med
        ) / sum(income_baseline.Households_nb_2001)
    average_income_baseline = np.sum(
        income_baseline.Households_nb * income_baseline.INC_med
        ) / sum(income_baseline.Households_nb)
    average_income_long_fut = np.sum(
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
        [average_income_retrospect, average_income_baseline,
         average_income_long_fut],
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

    return (spline_agricultural_price, spline_interest_rate,
            spline_population_income_distribution, spline_inflation,
            spline_income_distribution, spline_population,
            spline_income, spline_minimum_housing_supply,
            spline_fuel)


def compute_average_income(spline_population_income_distribution,
                           spline_income_distribution, param, t):
    """
    Compute average income and population per income group for a given year.

    This allows to update the relative distributions used to compute the
    equilibrium in subsequent periods (see equilibrium.compute_equilibrium).

    Parameters
    ----------
    spline_population_income_distribution : interp1d
        Linear interpolation for total population per income group in the data
        (12) over the years (baseline year set at 0)
    spline_income_distribution : interp1d
        Linear interpolation for median annual income (in rands) per income
        group in the data (12) over the years (baseline year set at 0)
    param : dict
        Dictionary of default parameters
    t : int
        Year for which we want to run the function

    Returns
    -------
    avg_income_group : ndarray(float64)
        Average median income for each income group in the model (4)
    total_group : ndarray(float64)
        Exogenous total number of households per income group (excluding people
        out of employment, for 4 groups)

    """
    total_bracket = spline_population_income_distribution(t)
    avg_income_bracket = spline_income_distribution(t)

    avg_income_group = np.zeros(4)
    total_group = np.zeros(4)

    for j in range(0, 4):
        total_group[j] = sum(
            total_bracket[param["income_distribution"] == j + 1])
        avg_income_group[j] = sum(
            avg_income_bracket[param["income_distribution"] == j + 1]
            * total_bracket[param["income_distribution"] == j + 1]
            ) / total_group[j]

    return avg_income_group, total_group


def interpolate_interest_rate(spline_interest_rate, t):
    """
    Return real interest rate used in model, for a given year.

    Parameters
    ----------
    spline_interest_rate : interp1d
        Linear interpolation for the interest rate (in %) over the years
    t : int
        Year for which we want to run the function

    Returns
    -------
    float64
        Real interest rate used in the model, and defined as the average over
        past (3) years to convey the structural (as opposed to conjonctural)
        component of the interest rate

    """
    nb_years = 3
    interest_rate_n_years = spline_interest_rate(np.arange(t - nb_years, t))
    interest_rate_n_years[interest_rate_n_years < 0] = np.nan
    return np.nanmean(interest_rate_n_years)/100


def evolution_housing_supply(housing_limit, param, t1, t0,
                             housing_supply_1, housing_supply_0):
    """
    Yield dynamic housing supply with time inertia and capital depreciation.

    We consider that formal private developers anticipate the unconstrained
    equilibrium value of their housing supply in future periods. Only if it is
    bigger than current values do we allow them to build more housing. In all
    cases, housing capital depreciates. This function computes a new housing
    supply including a time inertia parameter, that will be the (more
    realistic) housing supply simulated for target year (all the other
    equilibrium values will be updated correspondingly under this constraint).
    Then, the function returns the difference between simulated future and
    current values for housing supply.

    Parameters
    ----------
    housing_limit : Series
        Maximum housing supply (in m² per km²) in each grid cell (24,014)
    param : Dict
        Dictionary of default parameters
    t1 : float64
        Target year (relative to baseline set at 0) for evolution of housing
        supply
    t0 : float64
        Origin year (relative to baseline set at 0) for evolution of housing
        supply
    housing_supply_1 : ndarray(float64)
        (Unconstrained) equilibrium housing supply per unit of available land
        (in m² per km²) for target year, per grid cell (24,014)
    housing_supply_0 : TYPE
        Equilibrium housing supply per unit of available land
        (in m² per km²) for origin year, per grid cell (24,014)

    Returns
    -------
    Series
        Difference between simulated future and current values for housing
        supply per unit of available land (in m² per km²), per grid cell
        (24,014).

    """
    # New housing supply (accounting for inertia and depreciation with time)
    # See technical documentation for math formula
    if t1 - t0 > 0:
        # Yields the difference in housing supply (if growing) weighted by
        # time inertia, minus housing stock depreciation
        diff_housing = ((housing_supply_1 - housing_supply_0)
                        * (housing_supply_1 > housing_supply_0)
                        * (t1 - t0) / param["time_invest_housing"]
                        - housing_supply_0 * (t1 - t0)
                        * param["depreciation_rate"])
    # This allows to run backward simulations
    else:
        diff_housing = ((housing_supply_1 - housing_supply_0)
                        * (housing_supply_1 < housing_supply_0)
                        * (t1 - t0) / param["time_invest_housing"]
                        - housing_supply_0 * (t1 - t0)
                        * param["depreciation_rate"])

    # We set the target constrained equilibrium housing supply target
    housing_supply_target = housing_supply_0 + diff_housing
    # We bound it upwards as housing height is limited by potential regulations
    housing_supply_target = np.minimum(housing_supply_target, housing_limit)
    # We also consider minimum housing supply in the future (useful for ad hoc
    # corrections on Mitchells Plain)
    minimum_housing_supply_interp = interp1d(
        np.array([2001, 2011, 2100]) - param["baseline_year"],
        np.transpose([np.zeros(len(param["minimum_housing_supply"])),
                      param["minimum_housing_supply"],
                      param["minimum_housing_supply"]])
        )
    minimum_housing_supply_interp = minimum_housing_supply_interp(t1)
    # We also bound it downwards
    housing_supply_target = np.maximum(
        housing_supply_target, minimum_housing_supply_interp)

    # Finally, we return diff_housing after taking corrections into account
    return housing_supply_target - housing_supply_0
