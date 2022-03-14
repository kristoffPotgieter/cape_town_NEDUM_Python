# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:53:40 2022.

@author: monni
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


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
    #  We take estimate for total population in RDP in 2001 (why?)
    #  (estimated as sum(data.gridFormal(data.countRDPfromGV > 0)))  % 262452;
    #  % Estimated by nb inc_1 - BY - settlement in 2001
    RDP_2001 = 1.1718e+05

    # Land cover for informal settlements (see R code for details)

    # Set the area of a pixel (shouldn't it be set as parameter?) in m^2
    area_pixel = (0.5 ** 2) * 1000000

    # General surface data on land use (urban, agricultural...)
    land_use_data_old = pd.read_csv(
        path_data + 'grid_NEDUM_Cape_Town_500.csv', sep=';')

    # Surface data on scenarios for informal settlement building?
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

    # Nb of informal dwellings per pixel (realized scenario?)
    informal_settlements_2020 = pd.read_excel(
        path_folder + 'Flood plains - from Claus/inf_dwellings_2020.xlsx')
    # Why do we divide by the max % of buildable land? To account for risk?
    informal_risks_2020 = (
        informal_settlements_2020.inf_dwellings_2020
        * param["shack_size"] * (1 / param["max_land_use_settlement"]))
    # We neglect building risks smaller than 1% of a pixel area (why?)
    informal_risks_2020[informal_risks_2020 < area_pixel/100] = 0
    informal_risks_2020[np.isnan(informal_risks_2020)] = 0

    # Is it a selection of pixels? On which criterion?
    polygon_medium_timing = pd.read_excel(
        path_folder + 'Land occupation/polygon_medium_timing.xlsx',
        header=None)

    # We take the selected pixels from medium scenario and make their risk
    # area equal to the short scenario, then take the same selection in the
    # short scenario and make their risk area equal to zero (why?)
    for item in list(polygon_medium_timing.squeeze()):
        informal_risks_medium.area[
            informal_risks_medium["grid.data.ID"] == item
            ] = informal_risks_short.area[
                informal_risks_short["grid.data.ID"] == item]
        informal_risks_short.area[
            informal_risks_short["grid.data.ID"] == item] = 0


# 1. RDP population

    # We compute linear regression spline for 4 years centered around baseline
    # Where does growth rate come from? Set as parameter?
    spline_RDP = interp1d(
        [2001 - param["baseline_year"], 2011 - param["baseline_year"],
         2020 - param["baseline_year"], 2041 - param["baseline_year"]],
        [RDP_2001, RDP_2011, RDP_2011 + 9*5000,
         RDP_2011 + 9*5000 + 21 * param["future_rate_public_housing"]],
        'linear'
        )

    # We capture the output of the function to be used later on
    #  Should be set as param?
    year_begin_RDP = 2015
    #  We "center" scenarios around baseline year
    year_RDP = np.arange(year_begin_RDP, 2040) - param["baseline_year"]
    #  We take the spline output for this timeline
    number_RDP = spline_RDP(year_RDP)


# 2. RDP nb of houses

    # Setting the timeline

    #  We take the absolute difference between projected and actual RDP
    #  constructions over the years (RDP left to build?) and return the year
    #  index for the minimum... Meaning of variables?
    year_short_term = np.argmin(
        np.abs(sum(construction_rdp.total_yield_DU_ST)
               - (number_RDP - number_RDP[0]))
        )
    #  We just take size of the projection array for long term (set as param?)
    #  Why not use LT variables from construction_rdp?
    year_long_term = 30

    #  We save the timeline into a list
    year_data_informal = [
        2000 - param["baseline_year"],
        year_begin_RDP - param["baseline_year"],
        year_short_term,
        year_long_term
        ]

    # Getting the outcome

    #  Why do we weight by closeness to the center? Are we mixing data years?
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


# 3. RDP pixel share

    # Getting areas

    #  % of the pixel area dedicated to RDP (after accounting for backyarding)
    area_RDP = (data_rdp["area"] * param["RDP_size"]
                / (param["backyard_size"] + param["RDP_size"])
                / area_pixel)

    #  For the RDP constructed area, we take the min between declared value and
    #  extrapolation from our initial size parameters (why?)

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
    #  of urban area?
    coeff_land_backyard = np.fmin(urban, area_backyard)

    #  We take the population density from both formal and informal backyards
    #  (considered as backyarding) and reweight it by the ratio of max pixel
    #  share available for backyarding over max population density (why?)
    actual_backyards = (
        (housing_types.backyard_formal_grid
         + housing_types.backyard_informal_grid)
        / np.nanmax(housing_types.backyard_formal_grid
                    + housing_types.backyard_informal_grid)
        ) * np.max(coeff_land_backyard)

    #  We take the max from two different data sources?
    #  Idea is supposedly to have a pixel share potential for building
    coeff_land_backyard = np.fmax(coeff_land_backyard, actual_backyards)
    #  Why do we multiply by the max share of land available for backyarding
    #  although we have already been considering backyarding area?
    coeff_land_backyard = coeff_land_backyard * param["max_land_use_backyard"]
    coeff_land_backyard[coeff_land_backyard < 0] = 0

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

    #  We consider some scenario for 2023 (?) and correct for RDP construction
    #  (why?)
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
            coeff_land_backyard, spline_land_backyard, spline_land_informal,
            spline_land_constraints)
