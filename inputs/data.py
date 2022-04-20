# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:57:41 2020.

@author: Charlotte Liotta
"""

import numpy as np
import scipy.io
import pandas as pd
from scipy.interpolate import interp1d

import calibration.compute_income as calcmp
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
        path_precalc_inp + 'calibratedAmenities.mat')["amenities"]
    precalculated_amenities = np.load(
        path_precalc_inp + 'calibratedAmenities.npy')
    # Normalize index by mean of values
    amenities = (precalculated_amenities
                 / np.nanmean(precalculated_amenities)).squeeze()

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
    # income class (note that formal backyard is lacking)
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
    households_per_income_in_formal = np.zeros(param["nb_of_income_classes"])
    households_per_income_in_backyard = np.zeros(
        param["nb_of_income_classes"])
    households_per_income_in_informal = np.zeros(param["nb_of_income_classes"])

    # Compute population and average income for each class in the model
    for j in range(0, param["nb_of_income_classes"]):
        households_per_income_class[j] = np.sum(
            nb_of_hh_bracket[(param["income_distribution"] == j + 1)])
        average_income[j] = np.sum(
            avg_income_bracket[(param["income_distribution"] == j + 1)]
            * nb_of_hh_bracket[param["income_distribution"] == j + 1]
            ) / households_per_income_class[j]

        households_per_income_in_formal[j] = np.sum(
            income_2011.formal[(param["income_distribution"] == j + 1)])
        households_per_income_in_backyard[j] = np.sum(
            income_2011.informal_backyard[
                (param["income_distribution"] == j + 1)]
            )
        households_per_income_in_informal[j] = np.sum(
            income_2011.informal_settlement[
                (param["income_distribution"] == j + 1)]
            )

    #  Compute ratio of average income per class over global income average
    income_mult = average_income / mean_income

    # Store breakdown in unique array
    households_per_income_and_housing = np.vstack(
        [households_per_income_in_formal, households_per_income_in_backyard,
         households_per_income_in_informal])

    return (mean_income, households_per_income_class, average_income,
            income_mult, income_2011, households_per_income_and_housing)


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

    # TODO: does it make sense to use spline when we have accurate data?
    # Shouldn't we keep it for scenarios?
    # interest_rate = spline_interest_rate(0) / 100

    # Population
    # Raw figures come from Claus/ comes from housing_types
    # TODO: link with data
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

    # Set the area of a pixel in m²
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
    # Here we correct by max_land_use to make it comparable with
    # land_use_data_old (this will be removed afterwards as it does not
    # correspond to reality)
    informal_risks_2020 = (
        informal_settlements_2020.inf_dwellings_2020
        * param["shack_size"] * (1 / param["max_land_use_settlement"]))
    # We neglect building risks smaller than 1% of a pixel area
    informal_risks_2020[informal_risks_2020 < area_pixel/100] = 0
    informal_risks_2020[np.isnan(informal_risks_2020)] = 0

    # TODO: Ask Claus where this comes from
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

    # Getting areas: note that we divide by max_land_use as original data is
    # already corrected and we want to make the spline coherent with other
    # non-corrected housing types

    #  % of the pixel area dedicated to RDP (after accounting for backyard)
    area_RDP = (data_rdp["area"] * param["RDP_size"]
                / (param["backyard_size"] + param["RDP_size"])
                / area_pixel) / param["max_land_use"]

    #  For the RDP constructed area, we take the min between declared value and
    #  extrapolation from our initial size parameters

    #  We do it for the ST
    area_RDP_short_term = np.minimum(
        construction_rdp.area_ST,
        (param["backyard_size"] + param["RDP_size"])
        * construction_rdp.total_yield_DU_ST
        ) / param["max_land_use"]
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
        ) / param["max_land_use"]

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
    #  an alternative definition of coeff_land_backyard that includes formal
    #  backyarding and may be used for flood damage estimations
    #  TODO: check pb with floods
    # actual_backyards = (
    #     (housing_types.backyard_formal_grid
    #       + housing_types.backyard_informal_grid)
    #     / np.nanmax(housing_types.backyard_formal_grid
    #                 + housing_types.backyard_informal_grid)
    #     ) * np.max(coeff_land_backyard)
    actual_backyards = 0

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
    coeff_land_RDP = spline_land_RDP(t) * param["max_land_use"]
    coeff_land_settlement = (spline_land_informal(t)
                             * param["max_land_use_settlement"])
    coeff_land = np.array([coeff_land_private, coeff_land_backyard,
                           coeff_land_settlement, coeff_land_RDP])

    return coeff_land


def import_housing_limit(grid, param):
    """Return height limit within and out of historic city radius."""
    center_regulation = (grid["dist"] <= param["historic_radius"])
    outside_regulation = (grid["dist"] > param["historic_radius"])
    # We get the maximum amount of housing we can get per km² (hence the
    # multiplier as pixel area is given in m²)
    housing_limit = 1000000 * (
        param["limit_height_center"] * center_regulation
        + param["limit_height_out"] * outside_regulation
                     )

    return housing_limit


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

    # Damage functions give damage as a % of good considered, as a function
    # of flood depth in meters

    # Depth-damage functions (from de Villiers, 2007: ratios from table 3, and
    # table 4)
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

    # Depth-damage functions (from Englhardt, 2019: figure 2)
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

    # Take the max from the two sources to be conservative
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
    # This defines a probability rule (summing to 1) for each time interval
    # defined in FATHOM (the more distant, the less likely)
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

    # We consider that formal housing is not vulnerable to pluvial floods over
    # medium run, and that RDP and backyard are not over short run
    # This is based on CCT Minimum Standards for Stormwater Design 2014 (p.37)
    # and Govender 2011 (fig.8 and p.30): see Aux data and discussion w/ CLaus
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

    # Damage scenarios are incremented using damage functions multiplied by
    # flood-prone area (yields pixel share of destructed area), so as to
    # define damage intervals to be used in final computation

    # We take zero value at t = 0
    damages0 = (d[type_flood + '_5yr'].prop_flood_prone
                * damage_function(d[type_flood + '_5yr'].flood_depth))
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
    # We assume that value stays the same when t = +inf
    damages10 = ((d[type_flood + '_1000yr'].prop_flood_prone
                  * damage_function(d[type_flood + '_1000yr'].flood_depth))
                 + (d[type_flood + '_1000yr'].prop_flood_prone
                    * damage_function(d[type_flood + '_1000yr'].flood_depth)))

    # The formula for expected fraction of capital destroyed is given by the
    # integral of damage according to time (or rather, inverse probability).
    # Assuming that damage increase linearly with time (or inverse
    # probability), we can approximate this area as a sum of rectangles
    # defined for each of our intervals: this yields the following formula

    # NB: for more graphical intuition, see
    # https://storymaps.arcgis.com/stories/7878c89c592e4a78b45f03b4b696ccac

    return (0.5
            * ((interval0 * damages0) + (interval1 * damages1)
               + (interval2 * damages2) + (interval3 * damages3)
               + (interval4 * damages4) + (interval5 * damages5)
               + (interval6 * damages6) + (interval7 * damages7)
               + (interval8 * damages8) + (interval9 * damages9)
               + (interval10 * damages10)))


def import_full_floods_data(options, param, path_folder, housing_type_data):
    """Add fraction of capital destroyed by floods to initial floods data."""
    fraction_capital_destroyed = pd.DataFrame()

    (structural_damages_small_houses, structural_damages_medium_houses,
     structural_damages_large_houses, content_damages,
     structural_damages_type1, structural_damages_type2,
     structural_damages_type3a, structural_damages_type3b,
     structural_damages_type4a, structural_damages_type4b,
     d_fluvial, d_pluvial) = import_init_floods_data(
         options, param, path_folder)

# We take only fluvial as a baseline, and pluvial as an option.
# According to Claus, FATHOM data is less reliable for pluvial (which depends
# on many factors), hence we take it as an option.

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

    # We take a weighted average for structures in bricks and shacks among
    # all backyard structures

    backyards_by_material = pd.read_excel(
        path_folder
        + "CT Dwelling type data validation workbook 20201204 v2.xlsx",
        sheet_name="Analysis", header=None, names=None, usecols="G:H",
        skiprows=9, nrows=2)

    (fraction_capital_destroyed["structure_backyards"]
     ) = ((backyards_by_material.iloc[0, 0]
           * fraction_capital_destroyed["structure_formal_backyards"])
          + (backyards_by_material.iloc[1, 0]
              * fraction_capital_destroyed["structure_informal_backyards"])
          ) / housing_type_data[1]

    return (fraction_capital_destroyed, structural_damages_small_houses,
            structural_damages_medium_houses, structural_damages_large_houses,
            content_damages, structural_damages_type1,
            structural_damages_type2, structural_damages_type3a,
            structural_damages_type3b, structural_damages_type4a,
            structural_damages_type4b)


def infer_WBUS2_depth(housing_types, param, path_floods):
    """Update parameters with flood depth."""
    FATHOM_20yr = np.squeeze(pd.read_excel(path_floods + 'FD_20yr' + ".xlsx"))
    FATHOM_50yr = np.squeeze(pd.read_excel(path_floods + 'FD_50yr' + ".xlsx"))
    FATHOM_100yr = np.squeeze(pd.read_excel(
        path_floods + 'FD_100yr' + ".xlsx"))

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


def import_transport_data(grid, param, yearTraffic,
                          households_per_income_class, average_income,
                          spline_inflation,
                          spline_fuel,
                          spline_population_income_distribution,
                          spline_income_distribution,
                          path_precalc_inp, path_precalc_transp, dim):
    """Compute job center distribution, commuting and net income."""
    (timeOutput, distanceOutput, monetaryCost, costTime
     ) = calcmp.import_transport_costs(
         grid, param, yearTraffic, households_per_income_class,
         spline_inflation, spline_fuel, spline_population_income_distribution,
         spline_income_distribution, path_precalc_inp, path_precalc_transp,
         dim)

    param_lambda = param["lambda"].squeeze()

    incomeNetOfCommuting = np.zeros(
        (param["nb_of_income_classes"],
         timeOutput[:, :, 0].shape[1])
        )
    averageIncome = np.zeros(
        (param["nb_of_income_classes"],
         timeOutput[:, :, 0].shape[1])
        )
    modalShares = np.zeros(
        (185, timeOutput[:, :, 0].shape[1], 5,
         param["nb_of_income_classes"])
        )
    ODflows = np.zeros(
        (185, timeOutput[:, :, 0].shape[1],
         param["nb_of_income_classes"])
        )

    # Income
    incomeGroup, households_per_income_class = eqdyn.compute_average_income(
        spline_population_income_distribution, spline_income_distribution,
        param, yearTraffic)
    # Income centers: corresponds to expected income associated with each
    # income center and income group
    income_centers_init = scipy.io.loadmat(
        path_precalc_inp + 'incomeCentersKeep.mat')['incomeCentersKeep']
    # income_centers_init = np.load(path_precalc_inp + 'incomeCentersKeep.npy')
    # This allows to correct incomes for RDP people not taken into account in
    # initial income data (just in scenarios)
    incomeCenters = income_centers_init * incomeGroup / average_income

    # Switch to hourly
    annualToHourly = 1 / (8*20*12)
    # TODO: why?
    monetaryCost = monetaryCost * annualToHourly
    # We assume that people not taking some transport mode have a extra high
    # cost of doing so
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

        # Household size varies with income group / transport costs
        householdSize = param["household_size"][j]
        # Here, -100000 corresponds to an arbitrary value given to incomes in
        # centers with too few jobs to have convergence in calibration (could
        # have been nan): we exclude those centers from the analysis
        whichCenters = incomeCenters[:, j] > -100000
        incomeCentersGroup = incomeCenters[whichCenters, j]

        # Transport costs and employment allocation (cost per hour)

        (transportCostModes, transportCost, _, valueMax, minIncome
         ) = calcmp.compute_ODflows(
            householdSize, monetaryCost, costTime, incomeCentersGroup,
            whichCenters, param_lambda)

        # Modal shares
        # This comes from the multinomial model resulting from extreme value
        # theory with a Gumbel distribution (generalized EV type-I)
        # NB: here, we consider minimum Gumbel
        modalShares[whichCenters, :, :, j] = (np.exp(
            - param_lambda * transportCostModes + valueMax[:, :, None])
            / np.nansum(np.exp(- param_lambda * transportCostModes
                               + valueMax[:, :, None]), 2)[:, :, None]
            )

        # OD flows: corresponds to pi_c|ix (we redefine it as a full matrix)
        # NB: here, we consider maximum Gumbel
        ODflows[whichCenters, :, j] = (
            np.exp(param_lambda * (incomeCentersGroup[:, None] - transportCost)
                   - minIncome)
            / np.nansum(np.exp(param_lambda * (incomeCentersGroup[:, None]
                                               - transportCost) - minIncome),
                        0)[None, :]
            )

        # Expected income net of commuting costs (correct formula): cf. log-sum
        # calculation ans selection bias with weighted sum and error terms
        # NB: here, we consider maximum Gumbel
        incomeNetOfCommuting[j, :] = (
            1/param_lambda * (np.log(np.nansum(np.exp(
                param_lambda * (incomeCentersGroup[:, None] - transportCost)
                - minIncome),
                0)) + minIncome)
            )

        # Average income (not net of commuting costs) earned per worker
        # NB: here, we just take the weighted average as there is no error
        averageIncome[j, :] = np.nansum(
            ODflows[whichCenters, :, j] * incomeCentersGroup[:, None], 0)

    incomeNetOfCommuting = incomeNetOfCommuting / annualToHourly
    averageIncome = averageIncome / annualToHourly

    np.save(path_precalc_transp + dim + "_averageIncome_" + str(yearTraffic),
            averageIncome)
    np.save(path_precalc_transp + dim + "_incomeNetOfCommuting_"
            + str(yearTraffic), incomeNetOfCommuting)
    np.save(path_precalc_transp + dim + "_modalShares_" + str(yearTraffic),
            modalShares)
    np.save(path_precalc_transp + dim + "_ODflows_" + str(yearTraffic),
            ODflows)

    return incomeNetOfCommuting, modalShares, ODflows, averageIncome


def import_sal_data(grid, path_folder, path_data, housing_type_data):
    """Import SAL data for population density by housing type."""
    sal_data = pd.read_excel(
        path_folder
        + "CT Dwelling type data validation workbook 20201204 v2.xlsx",
        header=6)
    sal_data["informal"] = sal_data[
        "Informal dwelling (shack; not in backyard; e.g. in an"
        + " informal/squatter settlement or on a farm)"]
    sal_data["backyard_formal"] = sal_data["House/flat/room in backyard"]
    sal_data["backyard_informal"] = sal_data[
        "Informal dwelling (shack; in backyard)"]
    sal_data["formal"] = np.nansum(sal_data.iloc[:, 3:15], 1)

    grid_intersect = pd.read_csv(
        path_data + 'grid_SAL_intersect.csv', sep=';')
    grid_intersect.rename(columns={"Area_inter": "Area"}, inplace=True)

    informal_grid = gen_small_areas_to_grid(
        grid, grid_intersect, sal_data["informal"],
        sal_data["Small Area Code"], 'SAL')
    backyard_formal_grid = gen_small_areas_to_grid(
        grid, grid_intersect, sal_data["backyard_formal"],
        sal_data["Small Area Code"], 'SAL')
    backyard_informal_grid = gen_small_areas_to_grid(
        grid, grid_intersect, sal_data["backyard_informal"],
        sal_data["Small Area Code"], 'SAL')
    formal_grid = gen_small_areas_to_grid(
        grid, grid_intersect, sal_data["formal"], sal_data["Small Area Code"],
        'SAL')

    # We correct the number of dwellings per pixel by reweighting with the
    # ratio of total original number over total estimated number
    informal_grid = (informal_grid * (np.nansum(sal_data["informal"])
                                      / np.nansum(informal_grid)))
    backyard_formal_grid = (backyard_formal_grid
                            * (np.nansum(sal_data["backyard_formal"])
                               / np.nansum(backyard_formal_grid)))
    backyard_informal_grid = (backyard_informal_grid
                              * (np.nansum(sal_data["backyard_informal"])
                                 / np.nansum(backyard_informal_grid)))
    # We adapt the fraction given for formal housing to our initial data:
    # housing_type_data[0] + housing_type_data[3] = total_formal + total_RDP
    formal_grid = formal_grid * (
        (housing_type_data[0] + housing_type_data[3])
        / np.nansum(formal_grid))

    housing_types_grid_sal = pd.DataFrame()
    housing_types_grid_sal["informal_grid"] = informal_grid
    housing_types_grid_sal["backyard_formal_grid"] = backyard_formal_grid
    housing_types_grid_sal["backyard_informal_grid"] = backyard_informal_grid
    housing_types_grid_sal["formal_grid"] = formal_grid

    # Replace missing values by zero
    housing_types_grid_sal[np.isnan(housing_types_grid_sal)] = 0

    housing_types_grid_sal.to_excel(
        path_folder + 'housing_types_grid_sal.xlsx')

    return housing_types_grid_sal


# TODO: put deprecated functions in side script

def convert_income_distribution(income_distribution, grid, path_data, data_sp):
    """Import SP data for income distribution in grid form."""
    grid_intersect = pd.read_csv(
        path_data + 'grid_SP_intersect.csv', sep=';')

    income0_grid = gen_small_areas_to_grid(
        grid, grid_intersect, income_distribution[:, 0],
        data_sp["sp_code"], 'SP')
    income1_grid = gen_small_areas_to_grid(
        grid, grid_intersect, income_distribution[:, 1],
        data_sp["sp_code"], 'SP')
    income2_grid = gen_small_areas_to_grid(
        grid, grid_intersect, income_distribution[:, 2],
        data_sp["sp_code"], 'SP')
    income3_grid = gen_small_areas_to_grid(
        grid, grid_intersect, income_distribution[:, 3],
        data_sp["sp_code"], 'SP')

    # We correct the values per pixel by reweighting with the
    # ratio of total original number over total estimated number
    income0_grid = (income0_grid * (np.nansum(income_distribution[:, 0])
                                    / np.nansum(income0_grid)))
    income1_grid = (income1_grid * (np.nansum(income_distribution[:, 1])
                                    / np.nansum(income1_grid)))
    income2_grid = (income2_grid * (np.nansum(income_distribution[:, 2])
                                    / np.nansum(income2_grid)))
    income3_grid = (income3_grid * (np.nansum(income_distribution[:, 3])
                                    / np.nansum(income3_grid)))

    income_grid = np.stack(
        [income0_grid, income1_grid, income2_grid, income3_grid])

    # Replace missing values by zero
    income_grid[np.isnan(income_grid)] = 0

    np.save(path_data + "income_distrib_grid.npy", income_grid)

    return income_grid


def gen_small_areas_to_grid(grid, grid_intersect, small_area_data,
                            small_area_code, unit):
    """Convert SAL/SP to grid dimensions."""
    grid_data = np.zeros(len(grid.dist))
    for index in range(0, len(grid.dist)):
        intersect = np.unique(
            grid_intersect[unit + '_CODE'][grid_intersect.ID_grille
                                           == grid.id[index]]
            )
        if len(intersect) == 0:
            grid_data[index] = np.nan
        else:
            for i in range(0, len(intersect)):
                small_code = intersect[i]
                small_area_intersect = np.nansum(
                    grid_intersect.Area[
                        (grid_intersect.ID_grille == grid.id[index])
                        & (grid_intersect[unit + '_CODE'] == small_code)
                        ].squeeze())
                small_area = np.nansum(
                    grid_intersect.Area[
                        (grid_intersect[unit + '_CODE'] == small_code)]
                    )
                if len(small_area_data[
                        small_area_code == small_code]) > 0:
                    # Yields number of dwellings/people given by the
                    # intersection
                    add = (small_area_data[small_area_code == small_code]
                           * (small_area_intersect / small_area))
                else:
                    add = 0
                grid_data[index] = grid_data[index] + add

    return grid_data


# TODO: check if deprecated with Basile

def SP_to_grid_2011_1(data_SP, grid, path_data):
    """Adapt SP data to grid dimension."""
    grid_intersect = pd.read_csv(path_data + 'grid_SP_intersect.csv', sep=';')
    data_grid = np.zeros(len(grid.dist))
    for index in range(0, len(grid.dist)):
        # A priori, each SP code is associated to several pixels
        intersect = np.unique(
            grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.id[index]]
            )
        area_exclu = 0
        for i in range(0, len(intersect)):
            if len(data_SP['sp_code' == intersect[i]]) == 0:
                # We exclude the area of (all) pixel(s) corresponding to
                # unmatched SP
                area_exclu = (
                    area_exclu
                    + sum(grid_intersect.Area[(
                        grid_intersect.ID_grille == grid.id[index])
                        & (grid_intersect.SP_CODE == intersect[i])])
                    )
            else:
                # We add the SP data times the area of matched pixel(s)
                data_grid[index] = (
                    data_grid[index]
                    + sum(grid_intersect.Area[(
                        grid_intersect.ID_grille == grid.id[index])
                        & (grid_intersect.SP_CODE == intersect[i])])
                    * data_SP['sp_code' == intersect[i]]
                    )
        # We do not update data if excluded area is bigger than 90% of the
        # matched SP area (not used in practice)
        if area_exclu > (0.9 * sum(
                grid_intersect.Area[grid_intersect.ID_grille == grid.id[index]]
                )):
            data_grid[index] = np.nan
        # Else, if there is some positive matching, we update data with
        # nonsense?
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
