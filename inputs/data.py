# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:57:41 2020.

@author: Charlotte Liotta
"""

import numpy as np
import scipy.io
import pandas as pd
import copy
from scipy.interpolate import interp1d

import equilibrium.functions_dynamic as eqdyn


def import_grid(path_data):
    """Import pixel coordinates and distances to center."""
    data = pd.read_csv(path_data + 'grid_NEDUM_Cape_Town_500.csv', sep=';')
    grid = pd.DataFrame()
    grid["id"] = data.ID
    # Reduce dimensionality of data by 1,000
    grid["x"] = data.X/1000
    grid["y"] = data.Y/1000
    # Compute distance to grid center
    x_center = -53267.944572790904/1000
    y_center = -3754855.1309322729/1000
    grid["dist"] = (((grid.x - x_center) ** 2)
                    + ((grid.y - y_center) ** 2)) ** 0.5

    return grid, np.array([x_center, y_center])


def import_amenities(path_precalc_inp):
    """Import amenity index for each pixel."""
    # Follow calibration from Pfeiffer et al. (appendix C4)
    precalculated_amenities = scipy.io.loadmat(
        path_precalc_inp + 'calibratedAmenities.mat')
    # Normalize index by mean of values
    amenities = (precalculated_amenities["amenities"]
                 / np.nanmean(precalculated_amenities["amenities"])).squeeze()

    return amenities


def import_hypothesis_housing_type():
    """Import dummies to select income classes into housing types."""
    income_class_by_housing_type = pd.DataFrame()
    # Select which income class can live in formal settlements
    income_class_by_housing_type["formal"] = np.array([1, 1, 1, 1])
    # Select which income class can live in backyard settlements
    income_class_by_housing_type["backyard"] = np.array([1, 1, 0, 0])
    # Select which income class can live in informal settlements
    income_class_by_housing_type["settlement"] = np.array([1, 1, 0, 0])

    return income_class_by_housing_type


def import_income_classes_data(param, path_data):
    """Import population and average income per income class in the model."""
    # Import population distribution according to housing type (no RDP) and
    # income class
    income_2011 = pd.read_csv(path_data + 'Income_distribution_2011.csv')

    # Compute overall average income
    mean_income = np.sum(income_2011.Households_nb * income_2011.INC_med
                         ) / sum(income_2011.Households_nb)

    # Get income classes from data (12)
    nb_of_hh_bracket = income_2011.Households_nb
    avg_income_bracket = income_2011.INC_med

    # Initialize income classes in model (4)
    average_income = np.zeros(param["nb_of_income_classes"])
    households_per_income_class = np.zeros(param["nb_of_income_classes"])

    # Compute population and average income for each class in the model
    for j in range(0, param["nb_of_income_classes"]):
        households_per_income_class[j] = np.sum(
            nb_of_hh_bracket[(param["income_distribution"] == j + 1)])
        average_income[j] = np.sum(
            avg_income_bracket[(param["income_distribution"] == j + 1)]
            * nb_of_hh_bracket[param["income_distribution"] == j + 1]
            ) / households_per_income_class[j]

    #  Compute ratio of average income per class over global income average
    income_mult = average_income / mean_income

    return (mean_income, households_per_income_class, average_income,
            income_mult, income_2011)


def import_households_data(path_precalc_inp):
    """Import geographic data with class distributions for households."""
    # Import a structure of characteristics (for pixels and SPs mostly)
    data = scipy.io.loadmat(path_precalc_inp + 'data.mat')['data']

    #  Get maximum thresholds for model income classes (4)
    threshold_income_distribution = data[
        'thresholdIncomeDistribution'][0][0].squeeze()

    # Get data from RDP for pixels (24,014)
    data_rdp = pd.DataFrame()
    #  Number of RDP units
    data_rdp["count"] = data['gridCountRDPfromGV'][0][0].squeeze()
    #  Surface of RDP units
    data_rdp["area"] = data['gridAreaRDPfromGV'][0][0].squeeze()

    # Get other data for pixels
    #  Dummy indicating wheter pixel belongs to Mitchell's Plain district
    mitchells_plain_grid_2011 = data['MitchellsPlain'][0][0].squeeze()
    #  Population density in formal housing
    grid_formal_density_HFA = data['gridFormalDensityHFA'][0][0].squeeze()

    # Get housing type data for SPs (1,046)
    housing_types_sp = pd.DataFrame()
    #  Number of backyard settlements
    housing_types_sp["backyard_SP_2011"] = data[
        'spInformalBackyard'][0][0].squeeze()
    #  Number of informal settlements
    housing_types_sp["informal_SP_2011"] = data[
        'spInformalSettlement'][0][0].squeeze()
    #  Number of dwellings
    housing_types_sp["total_dwellings_SP_2011"] = data[
        'spTotalDwellings'][0][0].squeeze()
    #  Coordinates
    housing_types_sp["x_sp"] = data['spX'][0][0].squeeze()
    housing_types_sp["y_sp"] = data['spY'][0][0].squeeze()

    # Get other data for SPs
    data_sp = pd.DataFrame()
    #  Avg dwelling size
    data_sp["dwelling_size"] = data['spDwellingSize'][0][0].squeeze()
    #  Avg price of real estate (per m^2) for 2011
    #  (we do not consider data for 2001 and 2006)
    data_sp["price"] = data['spPrice'][0][0].squeeze()[2, :]
    #  Other aggregate statistics
    data_sp["income"] = data['sp2011AverageIncome'][0][0].squeeze()
    data_sp["unconstrained_area"] = data["spUnconstrainedArea"][0][0].squeeze()
    data_sp["area"] = data["sp2011Area"][0][0].squeeze()
    data_sp["distance"] = data["sp2011Distance"][0][0].squeeze()
    #  Dummy indicating wheter SP belongs to Mitchell's Plain district
    data_sp["mitchells_plain"] = data["sp2011MitchellsPlain"][0][0].squeeze()
    #  SP codes
    data_sp["sp_code"] = data["spCode"][0][0].squeeze()

    # Nb of househods of each model income class in each SP
    income_distribution = data[
        "sp2011IncomeDistributionNClass"][0][0].squeeze()
    # Dummy indicating whether SP belongs to Cape Town
    cape_town_limits = data["sp2011CapeTown"][0][0].squeeze()

    return (data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011,
            grid_formal_density_HFA, threshold_income_distribution,
            income_distribution, cape_town_limits)


def import_macro_data(param, path_scenarios):
    """Import interest rate and population per housing type."""
    # Interest rate
    #  Import interest_rate history + scenario until 2040
    scenario_interest_rate = pd.read_csv(
        path_scenarios + 'Scenario_interest_rate_1.csv', sep=';')
    #  Fit linear regression spline centered around baseline year
    spline_interest_rate = interp1d(
        scenario_interest_rate.Year_interest_rate[
            ~np.isnan(scenario_interest_rate.real_interest_rate)]
        - param["baseline_year"],
        scenario_interest_rate.real_interest_rate[
            ~np.isnan(scenario_interest_rate.real_interest_rate)],
        'linear'
        )
    #  Get interest rate as the mean (in %) over x last years
    nb_years_interest_rate = 3
    interest_rate_n_years = spline_interest_rate(
        np.arange(0 - nb_years_interest_rate, 0))
    #  We correct the obtained values
    interest_rate_n_years[interest_rate_n_years < 0] = np.nan
    interest_rate = np.nanmean(interest_rate_n_years)/100

    # Population
    # Raw figures come from Claus (to be updated)
    total_RDP = 194258
    total_formal = 626770
    total_informal = 143765
    total_backyard = 91132
    housing_type_data = np.array([total_formal, total_backyard, total_informal,
                                  total_RDP])
    population = sum(housing_type_data)

    return interest_rate, population, housing_type_data, total_RDP


def import_land_use(grid, options, param, data_rdp, housing_types,
                    housing_type_data, path_data, path_folder):
    """Return linear regression spline estimates for housing building paths."""
# 0. Import data

    # Social housing

    # New RDP construction projects
    construction_rdp = pd.read_csv(path_data + 'grid_new_RDP_projects.csv')

    # RDP population data
    #  We take total population in RDP at baseline year
    RDP_2011 = housing_type_data[3]
    #  Comes from Claus (to be updated)
    RDP_2001 = 1.1718e+05

    # Land cover for informal settlements (see R code for details)

    # Set the area of a pixel in m^2
    area_pixel = (0.5 ** 2) * 1000000

    # General surface data on land use (urban, agricultural...)
    land_use_data_old = pd.read_csv(
        path_data + 'grid_NEDUM_Cape_Town_500.csv', sep=';')

    # Surface data on scenarios for informal settlement building
    informal_risks_short = pd.read_csv(
        path_folder + 'Land occupation/informal_settlements_risk_SHORT.csv',
        sep=',')
    informal_risks_long = pd.read_csv(
        path_folder + 'Land occupation/informal_settlements_risk_LONG.csv',
        sep=',')
    informal_risks_medium = pd.read_csv(
        path_folder + 'Land occupation/informal_settlements_risk_MEDIUM.csv',
        sep=',')
    informal_risks_HIGH = pd.read_csv(
        path_folder + 'Land occupation/informal_settlements_risk_pHIGH.csv',
        sep=',')
    informal_risks_VERYHIGH = pd.read_csv(
        path_folder
        + 'Land occupation/informal_settlements_risk_pVERYHIGH.csv',
        sep=',')

    # Nb of informal dwellings per pixel (realized scenario)
    informal_settlements_2020 = pd.read_excel(
        path_folder + 'Flood plains - from Claus/inf_dwellings_2020.xlsx')
    # TODO: Why do we correct by max_land here? Supposedly to make it
    # comparable with land_use_data_old...
    informal_risks_2020 = (
        informal_settlements_2020.inf_dwellings_2020
        * param["shack_size"] * (1 / param["max_land_use_settlement"]))
    # We neglect building risks smaller than 1% of a pixel area
    informal_risks_2020[informal_risks_2020 < area_pixel/100] = 0
    informal_risks_2020[np.isnan(informal_risks_2020)] = 0

    # TODO: Ask Basile where this comes from
    polygon_medium_timing = pd.read_excel(
        path_folder + 'Land occupation/polygon_medium_timing.xlsx',
        header=None)

    # We take the selected pixels from medium scenario and make their risk
    # area equal to the short scenario, then take the same selection in the
    # short scenario and make their risk area equal to zero
    for item in list(polygon_medium_timing.squeeze()):
        informal_risks_medium.loc[
            informal_risks_medium["grid.data.ID"] == item,
            "area"
            ] = informal_risks_short.loc[
                informal_risks_short["grid.data.ID"] == item,
                "area"
                ]
        informal_risks_short.loc[
            informal_risks_short["grid.data.ID"] == item,
            "area"] = 0


# 1. RDP population

    # We compute linear regression spline for 4 years centered around baseline
    # Construction rate comes from working paper's median scenario, then a
    # lower rate after the end of the programme in 2020
    spline_RDP = interp1d(
        [2001 - param["baseline_year"], 2011 - param["baseline_year"],
         2020 - param["baseline_year"], 2041 - param["baseline_year"]],
        [RDP_2001, RDP_2011, RDP_2011 + 9*5000,
         RDP_2011 + 9*5000 + 21 * param["future_rate_public_housing"]],
        'linear'
        )

    # We capture the output of the function to be used later on
    #  Start of the programme
    year_begin_RDP = 2015
    #  We "center" scenarios around baseline year
    year_RDP = np.arange(year_begin_RDP, 2040) - param["baseline_year"]
    #  We take the spline output for this timeline
    number_RDP = spline_RDP(year_RDP)


# 2. RDP nb of houses

    # Setting the timeline

    #  We take the absolute difference between projected and actual RDP
    #  constructions over the years (RDP left to build) and return the year
    #  index for the minimum: this define the short-term horizon of the pgrm
    year_short_term = np.argmin(
        np.abs(sum(construction_rdp.total_yield_DU_ST)
               - (number_RDP - number_RDP[0]))
        )
    #  We just take size of the projection array for long-term horizon
    year_long_term = 30

    #  We save the timeline into a list
    year_data_informal = [
        2000 - param["baseline_year"],
        year_begin_RDP - param["baseline_year"],
        year_short_term,
        year_long_term
        ]

    # Getting the outcome

    #  We weight by closeness to the center to get retrospective number of RDP
    #  in 2001 (by assuming that central areas where built before)
    number_properties_2000 = (
        data_rdp["count"]
        * (1 - grid.dist / max(grid.dist[data_rdp["count"] > 0]))
        )
    #  Actual nb of RDP houses
    RDP_houses_estimates = data_rdp["count"]

    # Regression spline

    spline_estimate_RDP = interp1d(
        year_data_informal,
        np.transpose(
            [number_properties_2000,
             RDP_houses_estimates,
             RDP_houses_estimates + construction_rdp.total_yield_DU_ST,
             RDP_houses_estimates + construction_rdp.total_yield_DU_ST
             + construction_rdp.total_yield_DU_LT]
            ),
        'linear')

    number_properties_RDP = spline_estimate_RDP(0)


# 3. RDP pixel share

    # Getting areas

    #  % of the pixel area dedicated to RDP (after accounting for backyard)
    area_RDP = (data_rdp["area"] * param["RDP_size"]
                / (param["backyard_size"] + param["RDP_size"])
                / area_pixel)

    #  For the RDP constructed area, we take the min between declared value and
    #  extrapolation from our initial size parameters

    #  We do it for the ST
    area_RDP_short_term = np.minimum(
        construction_rdp.area_ST,
        (param["backyard_size"] + param["RDP_size"])
        * construction_rdp.total_yield_DU_ST
        )
    #  Then for the LT, while capping the constructed area at the pixel size
    #  (just in case)
    area_RDP_long_term = np.minimum(
        np.minimum(
            construction_rdp.area_ST + construction_rdp.area_LT,
            (param["backyard_size"] + param["RDP_size"])
            * (construction_rdp.total_yield_DU_ST
               + construction_rdp.total_yield_DU_LT)
            ),
        area_pixel
        )

    # Regression spline

    spline_land_RDP = interp1d(
        year_data_informal,
        np.transpose(
            [area_RDP, area_RDP, area_RDP_short_term, area_RDP_long_term]
            ),
        'linear')


# 4. Backyarding pixel share

    # Getting areas

    #  We get share of pixel backyard area by reweighting total area by the
    #  share of backyards in social housing units
    area_backyard = (data_rdp["area"] * param["backyard_size"]
                     / (param["backyard_size"] + param["RDP_size"])
                     / area_pixel)

    #  Share of pixel urban land
    urban = np.transpose(land_use_data_old.urban) / area_pixel
    #  We consider that the potential for backyard building cannot exceed that
    #  of urban area
    coeff_land_backyard = np.fmin(urban, area_backyard)

    #  We reweight max pixel share available for backyarding (both formal and
    #  informal) by a ratio of how densely populated the pixel is: this yields
    #  an alternative definition of coeef_land_backyard
    actual_backyards = (
        (housing_types.backyard_formal_grid
         + housing_types.backyard_informal_grid)
        / np.nanmax(housing_types.backyard_formal_grid
                    + housing_types.backyard_informal_grid)
        ) * np.max(coeff_land_backyard)

    #  We take the max from two different definitions to be conservative: we
    #  consider the maximum risk of backyard settlements
    #  This yields a pixel share potential for new structures
    # coeff_land_backyard = np.fmax(coeff_land_backyard, actual_backyards)
    #  TODO: should we multiply by the max share of land available here?
    # coeff_land_backyard = coeff_land_backyard * param["max_land_use_backyard"]
    # coeff_land_backyard[coeff_land_backyard < 0] = 0

    #  To project backyard share of pixel area on the ST, we add the potential
    #  backyard construction from RDP projects
    area_backyard_short_term = (
        area_backyard
        + np.maximum(
            area_RDP_short_term
            - construction_rdp.total_yield_DU_ST * param["RDP_size"],
            0
            ) / area_pixel
        )

    #  We do the same for RDP share of pixel area, by substracting backyarding
    area_RDP_short_term = (
        area_RDP
        + np.minimum(
            construction_rdp.total_yield_DU_ST * param["RDP_size"],
            construction_rdp.area_ST
            ) / area_pixel
        )

    #  We make sure that pixel share of backyard area does not exceed max
    #  available land after RDP construction
    area_backyard_short_term = np.minimum(
        area_backyard_short_term, param["max_land_use"] - area_RDP_short_term
        )

    #  We repeat the process for backyarding over the LT
    area_backyard_long_term = (
        area_backyard
        + np.maximum(
            area_RDP_long_term
            - (construction_rdp.total_yield_DU_LT
               + construction_rdp.total_yield_DU_ST) * param["RDP_size"],
            0
            ) / area_pixel
        )

    area_RDP_long_term = (
        area_RDP
        + np.minimum(
            (construction_rdp.total_yield_DU_LT
             + construction_rdp.total_yield_DU_ST) * param["RDP_size"],
            area_RDP_long_term
            ) / area_pixel
        )

    area_backyard_long_term = np.minimum(
        area_backyard_long_term, param["max_land_use"] - area_RDP_long_term
        )

    # Regression spline

    spline_land_backyard = interp1d(
        year_data_informal,
        np.transpose(
            [np.fmax(area_backyard, actual_backyards),
             np.fmax(area_backyard, actual_backyards),
             np.fmax(area_backyard_short_term, actual_backyards),
             np.fmax(area_backyard_long_term, actual_backyards)]
            ),
        'linear')


# 5. Unconstrained land pixel share

    # We get pixel share with and without urban edge
    coeff_land_no_urban_edge = (
        np.transpose(land_use_data_old.unconstrained_out)
        + np.transpose(land_use_data_old.unconstrained_UE)) / area_pixel
    coeff_land_urban_edge = np.transpose(
        land_use_data_old.unconstrained_UE) / area_pixel

    # Regression spline (with or without urban edge)

    if options["urban_edge"] == 0:
        year_constraints = np.array(
            [1990, param["year_urban_edge"] - 1, param["year_urban_edge"],
             2040]
            ) - param["baseline_year"]
        spline_land_constraints = interp1d(
            year_constraints,
            np.transpose(
                np.array(
                    [coeff_land_urban_edge, coeff_land_urban_edge,
                     coeff_land_no_urban_edge, coeff_land_no_urban_edge]
                    )
                ), 'linear'
            )
    else:
        year_constraints = np.array([1990, 2040]) - param["baseline_year"]
        spline_land_constraints = interp1d(
            year_constraints,
            np.transpose(
                np.array([coeff_land_urban_edge, coeff_land_urban_edge])
                )
            )


# 6. Informal settlement pixel share

    # Getting areas

    #  We get pixel share for informal settlement area at baseline
    informal_2011 = np.transpose(land_use_data_old.informal) / area_pixel

    #  We consider construction risk as the higher bound between initial
    #  conditions and prospective scenario
    informal_2020 = np.fmax(informal_risks_2020 / area_pixel, informal_2011)

    #  We also get area for high risk scenario
    high_proba = informal_risks_VERYHIGH.area + informal_risks_HIGH.area

    #  We consider some scenario for 2023 and correct for RDP construction
    #  (as informal predictions are not precise enough to account for such
    #  unavailable area)
    informal_2023 = np.fmin(
        coeff_land_no_urban_edge,
        np.fmax(
            informal_2020,
            np.transpose(np.fmin(informal_risks_short.area, high_proba))
            / area_pixel
            )
        ) - spline_land_RDP(12)
    informal_2023[informal_2023 < 0] = 0

    #  We do the same for 2025
    informal_2025 = np.fmin(
        coeff_land_no_urban_edge,
        np.fmax(
            informal_2023,
            np.transpose(np.fmin(informal_risks_medium.area, high_proba))
            / area_pixel
            )
        ) - spline_land_RDP(14)
    informal_2025[informal_2025 < 0] = 0

    #  And again for 2030
    informal_2030 = np.fmin(
        coeff_land_no_urban_edge,
        np.fmax(
            informal_2025,
            np.transpose(np.fmin(informal_risks_long.area, high_proba))
            / area_pixel
            )
        ) - spline_land_RDP(19)
    informal_2030[informal_2030 < 0] = 0

    # Regression spline (with land constraints)

    if options["informal_land_constrained"] == 0:
        spline_land_informal = interp1d(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 19, 29],
            np.transpose(
                [informal_2011, informal_2011, informal_2011, informal_2011,
                 informal_2011, informal_2011, informal_2011, informal_2011,
                 informal_2011, informal_2020, informal_2023, informal_2025,
                 informal_2030, informal_2030]
                ),
            'linear'
            )
    elif options["informal_land_constrained"] == 1:
        spline_land_informal = interp1d(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 19, 29],
            np.transpose(
                [informal_2011, informal_2011, informal_2011, informal_2011,
                 informal_2011, informal_2011, informal_2011, informal_2011,
                 informal_2011, informal_2020, informal_2020, informal_2020,
                 informal_2020, informal_2020]
                ),
            'linear'
            )


# 7. Function output

    return (spline_RDP, spline_estimate_RDP, spline_land_RDP,
            spline_land_backyard, spline_land_informal,
            spline_land_constraints, number_properties_RDP)


def import_coeff_land(spline_land_constraints, spline_land_backyard,
                      spline_land_informal, spline_land_RDP, param, t):
    """Return pixel share for housing scenarios, weighted by max building %."""
    coeff_land_private = (spline_land_constraints(t)
                          - spline_land_backyard(t)
                          - spline_land_informal(t)
                          - spline_land_RDP(t)) * param["max_land_use"]
    coeff_land_private[coeff_land_private < 0] = 0
    coeff_land_backyard = (spline_land_backyard(t)
                           * param["max_land_use_backyard"])
    # We do not need to reweight RDP available pixel share as we directly have
    # the true value from construction plans
    coeff_land_RDP = spline_land_RDP(t)
    coeff_land_settlement = (spline_land_informal(t)
                             * param["max_land_use_settlement"])
    coeff_land = np.array([coeff_land_private, coeff_land_backyard,
                           coeff_land_settlement, coeff_land_RDP])

    return coeff_land


def import_housing_limit(grid, param):
    """Return height limit within and out of historic city radius."""
    center_regulation = (grid["dist"] <= param["historic_radius"])
    outside_regulation = (grid["dist"] > param["historic_radius"])
    # Set high height multiplier to make as if no constraints
    housing_limit = (
        param["limit_height_center"] * 1000000 * center_regulation
        + param["limit_height_out"] * 1000000 * outside_regulation
                     )

    return housing_limit


# TODO: Study underlying assumptions

def import_init_floods_data(options, param, path_folder):
    """Import initial floods data and damage functions."""
    # Import floods data
    fluvial_floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr',
                      'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr',
                      'FD_1000yr']
    pluvial_floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr',
                      'P_100yr', 'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr']
    path_data = path_folder + "FATHOM/"

    d_pluvial = {}
    d_fluvial = {}
    for flood in fluvial_floods:
        d_fluvial[flood] = np.squeeze(
            pd.read_excel(path_data + flood + ".xlsx")
            )
    for flood in pluvial_floods:
        d_pluvial[flood] = np.squeeze(
            pd.read_excel(path_data + flood + ".xlsx")
            )

    # Depth-damage functions (from de Villiers, 2007)
    structural_damages_small_houses = interp1d(
        [0, 0.1, 0.6, 1.2, 2.4, 6, 10],
        [0, 0.0479, 0.1312, 0.1795, 0.3591, 1, 1]
        )
    structural_damages_medium_houses = interp1d(
        [0, 0.1, 0.6, 1.2, 2.4, 6, 10],
        [0, 0.083, 0.2273, 0.3083, 0.62, 1, 1]
        )
    structural_damages_large_houses = interp1d(
        [0, 0.1, 0.6, 1.2, 2.4, 6, 10],
        [0, 0.0799, 0.2198, 0.2997, 0.5994, 1, 1]
        )
    content_damages = interp1d(
        [0, 0.1, 0.3, 0.6, 1.2, 1.5, 2.4, 10],
        [0, 0.06, 0.15, 0.35, 0.77, 0.95, 1, 1]
        )

    # Depth-damage functions (from Englhardt, 2019)
    structural_damages_type1 = interp1d(
        [0, 0.5, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10],
        [0, 0.5, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )
    structural_damages_type2 = interp1d(
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10],
        [0, 0.45, 0.65, 0.82, 0.95, 1, 1, 1, 1, 1, 1, 1]
        )
    structural_damages_type3a = interp1d(
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10],
        [0, 0.4, 0.55, 0.7, 0.78, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81]
        )
    structural_damages_type3b = interp1d(
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10],
        [0, 0.25, 0.4, 0.48, 0.58, 0.62, 0.65, 0.75, 0.78, 0.8, 0.81, 0.81]
        )
    structural_damages_type4a = interp1d(
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10],
        [0, 0.31, 0.45, 0.55, 0.62, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65]
        )
    structural_damages_type4b = interp1d(
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10],
        [0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.62, 0.64, 0.65, 0.65]
        )

    if options["WBUS2"] == 1:
        WBUS2_20yr = pd.read_excel(
            path_folder + "Flood plains - from Claus/WBUS2_20yr.xlsx")
        WBUS2_50yr = pd.read_excel(
            path_folder + "Flood plains - from Claus/WBUS2_50yr.xlsx")
        WBUS2_100yr = pd.read_excel(
            path_folder + "Flood plains - from Claus/WBUS2_100yr.xlsx")
        d_fluvial['FD_20yr'].prop_flood_prone = np.fmax(
            d_fluvial['FD_20yr'].prop_flood_prone, WBUS2_20yr.prop_flood_prone)
        d_fluvial['FD_50yr'].prop_flood_prone = np.fmax(
            d_fluvial['FD_50yr'].prop_flood_prone, WBUS2_50yr.prop_flood_prone)
        d_fluvial['FD_100yr'].prop_flood_prone = np.fmax(
            d_fluvial['FD_100yr'].prop_flood_prone,
            WBUS2_100yr.prop_flood_prone)
        d_fluvial['FD_20yr'].flood_depth = np.maximum(
            d_fluvial['FD_20yr'].prop_flood_prone, param["depth_WBUS2_20yr"])
        d_fluvial['FD_50yr'].flood_depth = np.maximum(
            d_fluvial['FD_50yr'].prop_flood_prone, param["depth_WBUS2_50yr"])
        d_fluvial['FD_100yr'].flood_depth = np.maximum(
            d_fluvial['FD_100yr'].prop_flood_prone, param["depth_WBUS2_100yr"])

    return (structural_damages_small_houses, structural_damages_medium_houses,
            structural_damages_large_houses, content_damages,
            structural_damages_type1, structural_damages_type2,
            structural_damages_type3a, structural_damages_type3b,
            structural_damages_type4a, structural_damages_type4b,
            d_fluvial, d_pluvial)


def compute_fraction_capital_destroyed(d, type_flood, damage_function,
                                       housing_type):
    """Define function used to get fraction of capital destroyed by floods."""
    interval0 = 1 - (1/5)
    interval1 = (1/5) - (1/10)
    interval2 = (1/10) - (1/20)
    interval3 = (1/20) - (1/50)
    interval4 = (1/50) - (1/75)
    interval5 = (1/75) - (1/100)
    interval6 = (1/100) - (1/200)
    interval7 = (1/200) - (1/250)
    interval8 = (1/250) - (1/500)
    interval9 = (1/500) - (1/1000)
    interval10 = (1/1000)

    if ((type_flood == 'P') & (housing_type == 'formal')):
        d[type_flood + '_5yr'].prop_flood_prone = np.zeros(24014)
        d[type_flood + '_10yr'].prop_flood_prone = np.zeros(24014)
        d[type_flood + '_20yr'].prop_flood_prone = np.zeros(24014)
        d[type_flood + '_5yr'].flood_depth = np.zeros(24014)
        d[type_flood + '_10yr'].flood_depth = np.zeros(24014)
        d[type_flood + '_20yr'].flood_depth = np.zeros(24014)
    elif ((type_flood == 'P')
          & ((housing_type == 'subsidized') | (housing_type == 'backyard'))):
        d[type_flood + '_5yr'].prop_flood_prone = np.zeros(24014)
        d[type_flood + '_10yr'].prop_flood_prone = np.zeros(24014)
        d[type_flood + '_5yr'].flood_depth = np.zeros(24014)
        d[type_flood + '_10yr'].flood_depth = np.zeros(24014)

    damages0 = ((d[type_flood + '_5yr'].prop_flood_prone
                * damage_function(d[type_flood + '_5yr'].flood_depth))
                + (d[type_flood + '_5yr'].prop_flood_prone
                   * damage_function(d[type_flood + '_10yr'].flood_depth)))
    damages1 = ((d[type_flood + '_5yr'].prop_flood_prone
                 * damage_function(d[type_flood + '_5yr'].flood_depth))
                + (d[type_flood + '_10yr'].prop_flood_prone
                   * damage_function(d[type_flood + '_10yr'].flood_depth)))
    damages2 = ((d[type_flood + '_10yr'].prop_flood_prone
                 * damage_function(d[type_flood + '_10yr'].flood_depth))
                + (d[type_flood + '_20yr'].prop_flood_prone
                   * damage_function(d[type_flood + '_20yr'].flood_depth)))
    damages3 = ((d[type_flood + '_20yr'].prop_flood_prone
                 * damage_function(d[type_flood + '_20yr'].flood_depth))
                + (d[type_flood + '_50yr'].prop_flood_prone
                   * damage_function(d[type_flood + '_50yr'].flood_depth)))
    damages4 = ((d[type_flood + '_50yr'].prop_flood_prone
                 * damage_function(d[type_flood + '_50yr'].flood_depth))
                + (d[type_flood + '_75yr'].prop_flood_prone
                   * damage_function(d[type_flood + '_75yr'].flood_depth)))
    damages5 = ((d[type_flood + '_75yr'].prop_flood_prone
                 * damage_function(d[type_flood + '_75yr'].flood_depth))
                + (d[type_flood + '_100yr'].prop_flood_prone
                   * damage_function(d[type_flood + '_100yr'].flood_depth)))
    damages6 = ((d[type_flood + '_100yr'].prop_flood_prone
                 * damage_function(d[type_flood + '_100yr'].flood_depth))
                + (d[type_flood + '_200yr'].prop_flood_prone
                   * damage_function(d[type_flood + '_200yr'].flood_depth)))
    damages7 = ((d[type_flood + '_200yr'].prop_flood_prone
                 * damage_function(d[type_flood + '_200yr'].flood_depth))
                + (d[type_flood + '_250yr'].prop_flood_prone
                   * damage_function(d[type_flood + '_250yr'].flood_depth)))
    damages8 = ((d[type_flood + '_250yr'].prop_flood_prone
                 * damage_function(d[type_flood + '_250yr'].flood_depth))
                + (d[type_flood + '_500yr'].prop_flood_prone
                   * damage_function(d[type_flood + '_500yr'].flood_depth)))
    damages9 = ((d[type_flood + '_500yr'].prop_flood_prone
                 * damage_function(d[type_flood + '_500yr'].flood_depth))
                + (d[type_flood + '_1000yr'].prop_flood_prone
                   * damage_function(d[type_flood + '_1000yr'].flood_depth)))
    damages10 = ((d[type_flood + '_1000yr'].prop_flood_prone
                  * damage_function(d[type_flood + '_1000yr'].flood_depth))
                 + (d[type_flood + '_1000yr'].prop_flood_prone
                    * damage_function(d[type_flood + '_1000yr'].flood_depth)))

    return (0.5
            * ((interval0 * damages0) + (interval1 * damages1)
               + (interval2 * damages2) + (interval3 * damages3)
               + (interval4 * damages4) + (interval5 * damages5)
               + (interval6 * damages6) + (interval7 * damages7)
               + (interval8 * damages8) + (interval9 * damages9)
               + (interval10 * damages10)))


def import_full_floods_data(options, param, path_folder):
    """Add fraction of capital destroyed by floods to initial floods data."""
    fraction_capital_destroyed = pd.DataFrame()

    (structural_damages_small_houses, structural_damages_medium_houses,
     structural_damages_large_houses, content_damages,
     structural_damages_type1, structural_damages_type2,
     structural_damages_type3a, structural_damages_type3b,
     structural_damages_type4a, structural_damages_type4b,
     d_fluvial, d_pluvial) = import_init_floods_data(
         options, param, path_folder)

    if options["pluvial"] == 0:
        (fraction_capital_destroyed["contents_formal"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'formal')
        (fraction_capital_destroyed["contents_informal"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'informal')
        (fraction_capital_destroyed["contents_backyard"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'backyard')
        (fraction_capital_destroyed["contents_subsidized"]
         ) = compute_fraction_capital_destroyed
        (d_fluvial, 'FD', content_damages, 'subsidized')
        (fraction_capital_destroyed["structure_formal_1"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'formal')
        (fraction_capital_destroyed["structure_formal_2"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'formal')
        (fraction_capital_destroyed["structure_subsidized_1"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'subsidized')
        (fraction_capital_destroyed["structure_subsidized_2"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'subsidized')
        (fraction_capital_destroyed["structure_informal_settlements"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'informal')
        (fraction_capital_destroyed["structure_informal_backyards"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'backyard')
        (fraction_capital_destroyed["structure_formal_backyards"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type3a, 'backyard')
    elif options["pluvial"] == 1:
        (fraction_capital_destroyed["contents_formal"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'formal')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'formal'))
        (fraction_capital_destroyed["contents_informal"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'informal')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'informal'))
        (fraction_capital_destroyed["contents_backyard"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'backyard')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'backyard'))
        (fraction_capital_destroyed["contents_subsidized"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'subsidized')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'subsidized'))
        (fraction_capital_destroyed["structure_formal_1"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'formal')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4a, 'formal'))
        (fraction_capital_destroyed["structure_formal_2"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'formal')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4b, 'formal'))
        (fraction_capital_destroyed["structure_subsidized_1"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'subsidized')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4a, 'subsidized'))
        (fraction_capital_destroyed["structure_subsidized_2"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'subsidized')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4b, 'subsidized'))
        (fraction_capital_destroyed["structure_informal_settlements"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'informal')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type2, 'informal'))
        (fraction_capital_destroyed["structure_informal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'backyard')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type2, 'backyard'))
        (fraction_capital_destroyed["structure_formal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type3a, 'backyard')
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type3a, 'backyard'))

    (fraction_capital_destroyed["structure_backyards"]
     ) = (
         (16216 * fraction_capital_destroyed["structure_formal_backyards"])
         + (74916 * fraction_capital_destroyed["structure_informal_backyards"])
         ) / 91132

    return (fraction_capital_destroyed, structural_damages_small_houses,
            structural_damages_medium_houses, structural_damages_large_houses,
            content_damages, structural_damages_type1,
            structural_damages_type2, structural_damages_type3a,
            structural_damages_type3b, structural_damages_type4a,
            structural_damages_type4b)


def infer_WBUS2_depth(housing_types, param, path_folder):
    """Update parameters with flood depth."""
    path_data = path_folder + "FATHOM/"
    FATHOM_20yr = np.squeeze(pd.read_excel(path_data + 'FD_20yr' + ".xlsx"))
    FATHOM_50yr = np.squeeze(pd.read_excel(path_data + 'FD_50yr' + ".xlsx"))
    FATHOM_100yr = np.squeeze(pd.read_excel(path_data + 'FD_100yr' + ".xlsx"))

    FATHOM_20yr['pop_flood_prone'] = (
        FATHOM_20yr.prop_flood_prone
        * (housing_types.informal_grid
           + housing_types.formal_grid
           + housing_types.backyard_formal_grid
           + housing_types.backyard_informal_grid)
        )
    FATHOM_50yr['pop_flood_prone'] = (
        FATHOM_50yr.prop_flood_prone
        * (housing_types.informal_grid
           + housing_types.formal_grid
           + housing_types.backyard_formal_grid
           + housing_types.backyard_informal_grid)
        )
    FATHOM_100yr['pop_flood_prone'] = (
        FATHOM_100yr.prop_flood_prone
        * (housing_types.informal_grid
           + housing_types.formal_grid
           + housing_types.backyard_formal_grid
           + housing_types.backyard_informal_grid)
        )

    param["depth_WBUS2_20yr"] = (np.nansum(
        FATHOM_20yr.pop_flood_prone * FATHOM_20yr.flood_depth)
        / np.nansum(FATHOM_20yr.pop_flood_prone))
    param["depth_WBUS2_50yr"] = (np.nansum(
        FATHOM_50yr.pop_flood_prone * FATHOM_50yr.flood_depth)
        / np.nansum(FATHOM_50yr.pop_flood_prone))
    param["depth_WBUS2_100yr"] = (np.nansum(
        FATHOM_100yr.pop_flood_prone * FATHOM_100yr.flood_depth)
        / np.nansum(FATHOM_100yr.pop_flood_prone))

    return param


# TODO: Determine if the following is still useful

def import_basile_simulation():
    """Import obsolete data."""
    mat1 = scipy.io.loadmat(
        'C:/Users/charl/OneDrive/Bureau/Cape Town - pour Charlotte/Modèle/'
        + 'projet_le_cap/simulations scenarios - 201908.mat')
    simul1 = mat1["simulation_noUE"]
    simul1_error = simul1["error"][0][0]
    simul1_utility = simul1["utility"][0][0]
    simul1_households_housing_type = simul1["householdsHousingType"][0][0]
    simul1_rent = simul1["rent"][0][0]
    simul1_dwelling_size = simul1["dwellingSize"][0][0]
    simul1_households_center = simul1["householdsCenter"][0][0]
    data = scipy.io.loadmat(
        'C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/'
        + '0. Precalculated inputs/data.mat')['data']
    SP_code = data["spCode"][0][0].squeeze()

    return (simul1_error, simul1_utility, simul1_households_housing_type,
            simul1_rent, simul1_dwelling_size, simul1_households_center,
            SP_code)


def SP_to_grid_2011_1(data_SP, grid, path_data):
    """Adapt SP data to grid dimension."""
    grid_intersect = pd.read_csv(path_data + 'grid_SP_intersect.csv', sep=';')
    data_grid = np.zeros(len(grid.dist))
    for index in range(0, len(grid.dist)):
        intersect = np.unique(
            grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.id[index]]
            )
        area_exclu = 0
        for i in range(0, len(intersect)):
            if len(data_SP['sp_code' == intersect[i]]) == 0:
                area_exclu = (
                    area_exclu
                    + sum(grid_intersect.Area[(
                        grid_intersect.ID_grille == grid.id[index])
                        & (grid_intersect.SP_CODE == intersect[i])])
                    )
            else:
                data_grid[index] = (
                    data_grid[index]
                    + sum(grid_intersect.Area[(
                        grid_intersect.ID_grille == grid.id[index])
                        & (grid_intersect.SP_CODE == intersect[i])])
                    * data_SP['sp_code' == intersect[i]]
                    )
        if area_exclu > (0.9 * sum(
                grid_intersect.Area[grid_intersect.ID_grille == grid.id[index]]
                )):
            data_grid[index] = np.nan
        elif sum(
               grid_intersect.Area[grid_intersect.ID_grille == grid.id[index]]
               ) - area_exclu > 0:
            data_grid[index] = (
                data_grid[index]
                / (sum(grid_intersect.Area[
                    grid_intersect.ID_grille == grid.id[index]]
                    ) - area_exclu))
        else:
            data_grid[index] = np.nan

    return data_grid


def import_transport_data(grid, param, yearTraffic, spline_inflation,
                          spline_fuel, path_precalc_inp,
                          spline_population_income_distribution,
                          spline_income_distribution,
                          households_per_income_class, average_income):
    """Compute travel times and costs."""
    # STEP 1: IMPORT TRAVEL TIMES AND COSTS

    # Import travel times and distances
    transport_times = scipy.io.loadmat(path_precalc_inp
                                       + 'Transport_times_GRID.mat')

    # Price per km
    priceTrainPerKMMonth = (
        0.164 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
                            )
    priceTrainFixedMonth = (
        4.48 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceTaxiPerKMMonth = (
        0.785 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceTaxiFixedMonth = (
        4.32 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceBusPerKMMonth = (
        0.522 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    priceBusFixedMonth = (
        6.24 * 40 * spline_inflation(2011 - param["baseline_year"])
        / spline_inflation(2013 - param["baseline_year"])
        )
    inflation = spline_inflation(yearTraffic)
    infla_2012 = spline_inflation(2012 - param["baseline_year"])
    priceTrainPerKMMonth = priceTrainPerKMMonth * inflation / infla_2012
    priceTrainFixedMonth = priceTrainFixedMonth * inflation / infla_2012
    priceTaxiPerKMMonth = priceTaxiPerKMMonth * inflation / infla_2012
    priceTaxiFixedMonth = priceTaxiFixedMonth * inflation / infla_2012
    priceBusPerKMMonth = priceBusPerKMMonth * inflation / infla_2012
    priceBusFixedMonth = priceBusFixedMonth * inflation / infla_2012
    priceFuelPerKMMonth = spline_fuel(yearTraffic)
    if yearTraffic > 8:
        priceFuelPerKMMonth = priceFuelPerKMMonth * 1.2
        # priceBusPerKMMonth = priceBusPerKMMonth * 1.2
        # priceTaxiPerKMMonth = priceTaxiPerKMMonth * 1.2

    # Fixed costs
    priceFixedVehiculeMonth = 400
    priceFixedVehiculeMonth = priceFixedVehiculeMonth * inflation / infla_2012

    # STEP 2: TRAVEL TIMES AND COSTS AS MATRIX

    # Parameters
    numberDaysPerYear = 235
    numberHourWorkedPerDay = 8
    annualToHourly = 1 / (8*20*12)

    # Time taken by each mode in both direction (in min)
    timeOutput = np.empty(
        (transport_times["durationTrain"].shape[0],
         transport_times["durationTrain"].shape[1], 5)
        )
    timeOutput[:] = np.nan
    timeOutput[:, :, 0] = (transport_times["distanceCar"]
                           / param["walking_speed"] * 60 * 1.2 * 2)
    timeOutput[:, :, 0][np.isnan(transport_times["durationCar"])] = np.nan
    timeOutput[:, :, 1] = copy.deepcopy(transport_times["durationTrain"])
    timeOutput[:, :, 2] = copy.deepcopy(transport_times["durationCar"])
    timeOutput[:, :, 3] = copy.deepcopy(transport_times["durationMinibus"])
    timeOutput[:, :, 4] = copy.deepcopy(transport_times["durationBus"])

    # Length (in km) using each mode
    multiplierPrice = np.empty((timeOutput.shape))
    multiplierPrice[:] = np.nan
    multiplierPrice[:, :, 0] = np.zeros((timeOutput[:, :, 0].shape))
    multiplierPrice[:, :, 1] = transport_times["distanceCar"]
    multiplierPrice[:, :, 2] = transport_times["distanceCar"]
    multiplierPrice[:, :, 3] = transport_times["distanceCar"]
    multiplierPrice[:, :, 4] = transport_times["distanceCar"]

    # Multiplying by 235 (nb of working days per year)
    pricePerKM = np.empty(5)
    pricePerKM[:] = np.nan
    pricePerKM[0] = np.zeros(1)
    pricePerKM[1] = priceTrainPerKMMonth*numberDaysPerYear
    pricePerKM[2] = priceFuelPerKMMonth*numberDaysPerYear
    pricePerKM[3] = priceTaxiPerKMMonth*numberDaysPerYear
    pricePerKM[4] = priceBusPerKMMonth*numberDaysPerYear

    # Distances (not useful to calculate price but useful output)
    distanceOutput = np.empty((timeOutput.shape))
    distanceOutput[:] = np.nan
    distanceOutput[:, :, 0] = transport_times["distanceCar"]
    distanceOutput[:, :, 1] = transport_times["distanceCar"]
    distanceOutput[:, :, 2] = transport_times["distanceCar"]
    distanceOutput[:, :, 3] = transport_times["distanceCar"]
    distanceOutput[:, :, 4] = transport_times["distanceCar"]

    # Monetary price per year
    monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
    trans_monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
    for index2 in range(0, 5):
        monetaryCost[:, :, index2] = (pricePerKM[index2]
                                      * multiplierPrice[:, :, index2])

    #  Train (monthly fare)
    monetaryCost[:, :, 1] = monetaryCost[:, :, 1] + priceTrainFixedMonth * 12
    #  Private car
    monetaryCost[:, :, 2] = (monetaryCost[:, :, 2] + priceFixedVehiculeMonth
                             * 12)
    #  Minibus/taxi
    monetaryCost[:, :, 3] = monetaryCost[:, :, 3] + priceTaxiFixedMonth * 12
    #  Bus
    monetaryCost[:, :, 4] = monetaryCost[:, :, 4] + priceBusFixedMonth * 12
    trans_monetaryCost = copy.deepcopy(monetaryCost)

    # STEP 3: COMPUTE PROBA TO WORK IN C, EXPECTED INCOME, AND EXPECTED NB OF
    # RESIDENTS OF INCOME GROUP I WORKING IN C

    # In transport hours per working hour
    costTime = ((timeOutput * param["time_cost"])
                / (60 * numberHourWorkedPerDay))
    costTime[np.isnan(costTime)] = 10 ** 2
    param_lambda = param["lambda"].squeeze()

    incomeNetOfCommuting = np.zeros(
        (param["nb_of_income_classes"],
         transport_times["durationCar"].shape[1])
        )
    averageIncome = np.zeros(
        (param["nb_of_income_classes"],
         transport_times["durationCar"].shape[1])
        )
    modalShares = np.zeros(
        (185, transport_times["durationCar"].shape[1], 5,
         param["nb_of_income_classes"])
        )
    ODflows = np.zeros(
        (185, transport_times["durationCar"].shape[1],
         param["nb_of_income_classes"])
        )

    # Income
    incomeGroup, households_per_income_class = eqdyn.compute_average_income(
        spline_population_income_distribution, spline_income_distribution,
        param, yearTraffic)
    # Income centers
    income_centers_init = scipy.io.loadmat(
        path_precalc_inp + 'incomeCentersKeep.mat')['incomeCentersKeep']
    incomeCenters = income_centers_init * incomeGroup / average_income

    # Switch to hourly
    monetaryCost = trans_monetaryCost * annualToHourly
    monetaryCost[np.isnan(monetaryCost)] = 10**3 * annualToHourly
    incomeCenters = incomeCenters * annualToHourly

    # xInterp = grid.x
    # yInterp = grid.y

    # If changes?
    # (monetaryCost[:, (grid.dist < 15) & (grid.dist > 10), :]
    #  ) = monetaryCost[:, (grid.dist < 15) & (grid.dist > 10), :] * 1.2
    # (monetaryCost[:, (grid.dist < 30) & (grid.dist > 22), :]
    #  ) = monetaryCost[:, (grid.dist < 30) & (grid.dist > 22), :] * 0.7
    # (costTime[:, (grid.dist < 15) & (grid.dist > 10), :]
    #  ) = costTime[:, (grid.dist < 15) & (grid.dist > 10), :] * 1.2
    # (costTime[:, (grid.dist < 30) & (grid.dist > 22), :]
    #  ) = costTime[:, (grid.dist < 30) & (grid.dist > 22), :] * 0.7
    # (monetaryCost[:, (grid.dist < 25) & (grid.dist > 22), :]
    #  ) = monetaryCost[:, (grid.dist < 25) & (grid.dist > 22), :] * 0.8
    # (costTime[:, (grid.dist < 25) & (grid.dist > 22), :]
    #  ) = costTime[:, (grid.dist < 25) & (grid.dist > 22), :] * 0.8
    # (monetaryCost[:, (grid.dist < 11) & (grid.dist > 8), :]
    #  ) = monetaryCost[:, (grid.dist < 11) & (grid.dist > 8), :] * 0.8
    # (costTime[:, (grid.dist < 11) & (grid.dist > 8), :]
    #  ) = costTime[:, (grid.dist < 11) & (grid.dist > 8), :] * 0.8
    # (monetaryCost[:, (grid.dist < 22) & (grid.dist > 14), :]
    #  ) = monetaryCost[:, (grid.dist < 22) & (grid.dist > 14), :] * 0.8
    # (costTime[:, (grid.dist < 22) & (grid.dist > 14), :]
    #  ) = costTime[:, (grid.dist < 22) & (grid.dist > 14), :] * 0.8

    for j in range(0, param["nb_of_income_classes"]):

        # Household size varies with transport costs
        householdSize = param["household_size"][j]
        whichCenters = incomeCenters[:, j] > -100000
        incomeCentersGroup = incomeCenters[whichCenters, j]

        # Transport costs and employment allocation (cout par heure)
        transportCostModes = (
            (householdSize * monetaryCost[whichCenters, :, :]
             + (costTime[whichCenters, :, :]
                * incomeCentersGroup[:, None, None]))
            )

        # To prevent exp from diverging to infinity?
        valueMax = (np.min(param_lambda * transportCostModes, axis=2) - 500)

        # Modal shares
        modalShares[whichCenters, :, :, j] = (np.exp(
            - param_lambda * transportCostModes + valueMax[:, :, None])
            / np.nansum(np.exp(- param_lambda * transportCostModes
                               + valueMax[:, :, None]), 2)[:, :, None]
            )

        # Transport costs
        transportCost = (
            - 1 / param_lambda
            * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes
                                       + valueMax[:, :, None]), 2)) - valueMax)
            )

        # To prevent diverging exponentials?
        minIncome = (np.nanmax(
            param_lambda * (incomeCentersGroup[:, None] - transportCost), 0)
            - 700)

        # OD flows
        ODflows[whichCenters, :, j] = (
            np.exp(param_lambda * (incomeCentersGroup[:, None] - transportCost)
                   - minIncome)
            / np.nansum(np.exp(param_lambda * (incomeCentersGroup[:, None]
                                               - transportCost) - minIncome),
                        0)[None, :]
            )

        # Income net of commuting costs (correct formula)
        incomeNetOfCommuting[j, :] = (
            1/param_lambda * (np.log(np.nansum(np.exp(
                param_lambda * (incomeCentersGroup[:, None] - transportCost)
                - minIncome),
                0)) + minIncome)
            )

        # Average income earned per worker
        averageIncome[j, :] = np.nansum(
            ODflows[whichCenters, :, j] * incomeCentersGroup[:, None], 0)

    incomeNetOfCommuting = incomeNetOfCommuting / annualToHourly
    averageIncome = averageIncome / annualToHourly

    return incomeNetOfCommuting, modalShares, ODflows, averageIncome
