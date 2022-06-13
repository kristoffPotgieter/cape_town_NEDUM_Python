# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:57:41 2020.

@author: Charlotte Liotta
"""

import numpy as np
import scipy.io
import pandas as pd
from scipy.interpolate import interp1d

import calibration.sub.compute_income as calcmp
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


def import_amenities(path_precalc_inp, options):
    """Import amenity index for each pixel."""
    # Follow calibration from Pfeiffer et al. (appendix C4)
    if options["load_precal_param"] == 1:
        precalculated_amenities = scipy.io.loadmat(
            path_precalc_inp + 'calibratedAmenities.mat')["amenities"]
    elif options["load_precal_param"] == 0:
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
    # Select which income class can live in informal settlements
    income_class_by_housing_type["subsidized"] = np.array([1, 0, 0, 0])

    return income_class_by_housing_type


def import_income_classes_data(param, path_data):
    """Import population and average income per income class in the model."""
    # Import population distribution according to housing type and income class
    # Note that RDP is included in formal
    # TODO: is formal backyard lacking or just included in formal housing?
    income_2011 = pd.read_csv(path_data + 'Income_distribution_2011.csv')

    # Compute overall median income
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

    # Compute population and median income for each class in the model
    for j in range(0, param["nb_of_income_classes"]):
        households_per_income_class[j] = np.sum(
            nb_of_hh_bracket[(param["income_distribution"] == j + 1)])
        # Note that this is in fact an average over median incomes
        average_income[j] = np.sum(
            avg_income_bracket[(param["income_distribution"] == j + 1)]
            * nb_of_hh_bracket[param["income_distribution"] == j + 1]
            ) / households_per_income_class[j]

        # We do the same across housing types for validation
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
    # NB: this is obtained from cadastre data (pre-processed by Claus)
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
    #  TODO: check if this includes formal backyards
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


def import_macro_data(param, path_scenarios, path_folder):
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
    interest_rate = eqdyn.interpolate_interest_rate(spline_interest_rate, 0)

    # Aggregate population: this come from SAL data and serves as a benchmark
    # to reweight aggregates obtained from diverse granular data sets so that
    # the same overall population is indeed considered

    sal_data = pd.read_excel(
        path_folder
        + "CT Dwelling type data validation workbook 20201204 v2.xlsx",
        header=6, nrows=5339)
    sal_data["informal"] = sal_data[
        "Informal dwelling (shack; not in backyard; e.g. in an"
        + " informal/squatter settlement or on a farm)"]
    sal_data["backyard_formal"] = sal_data["House/flat/room in backyard"]
    sal_data["backyard_informal"] = sal_data[
        "Informal dwelling (shack; in backyard)"]
    # NB: we do not include traditional houses, granny flats, caravans, and
    # others
    sal_data["formal"] = np.nansum(sal_data.iloc[:, [3, 5, 6, 7, 8]], 1)

    rdp_data = pd.read_excel(
        path_folder
        + "CT Dwelling type data validation workbook 20201204 v2.xlsx",
        sheet_name='Analysis', header=None, names=None, usecols="A:B",
        skiprows=18, nrows=6)

    total_RDP = int(rdp_data.iloc[5, 1])
    # Else, SR2 (incremental housing) properties according to 2012 General
    # Valuation yield 138,444 properties. By adding approximately 40,000
    # council housing units (similar to RDP/BNG in use), we get a "close"
    # estimate:
    # total_RDP = 138444 + 40000
    total_formal = sal_data["formal"].sum() - total_RDP
    total_informal = sal_data["informal"].sum()
    total_backyard = sal_data["backyard_informal"].sum()
    # Note that we only include informal backyards  by assumption in the model:
    # we are sure that formal backyards are note included in formal count here

    # Old raw figures from Claus
    # total_RDP = 194258
    # total_formal = 626770
    # total_informal = 143765
    # total_backyard = 91132

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
    #  Also comes from general validation for 2001
    #  TODO: ask Claus for references
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

    # Nb of informal dwellings per pixel
    informal_settlements_2020 = pd.read_excel(
        path_folder + 'Flood plains - from Claus/inf_dwellings_2020.xlsx')
    # Here we correct by 1 / max_land_use to make it comparable with other
    # informal risks (this will be removed afterwards as it does not
    # correspond to reality)
    informal_risks_2020 = (
        informal_settlements_2020.inf_dwellings_2020
        * param["shack_size"] * (1 / param["max_land_use_settlement"]))
    # We neglect building risks smaller than 1% of a pixel area
    informal_risks_2020[informal_risks_2020 < area_pixel/100] = 0
    informal_risks_2020[np.isnan(informal_risks_2020)] = 0

    # Pixel selection for scenario correction

    # We want to include pushback from farmers around Philipi, hence we delay
    # settlement of those areas compared to other short-term risks
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


# 1. Total RDP population

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
    #  in 2000 (by assuming that central areas where built before)
    number_properties_2000 = (
        data_rdp["count"]
        * (1 - grid.dist / max(grid.dist[data_rdp["count"] > 0]))
        )
    #  Actual nb of RDP houses in 2011
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

    #  % of the pixel area dedicated to RDP (after accounting for backyards)
    area_RDP = (data_rdp["area"] * param["RDP_size"]
                / (param["backyard_size"] + param["RDP_size"])
                / area_pixel) / param["max_land_use"]

    #  For the RDP constructed area, we take the min between declared value and
    #  extrapolation from our initial size parameters

    #  We do it for the short term
    area_RDP_short_term = np.minimum(
        construction_rdp.area_ST,
        (param["backyard_size"] + param["RDP_size"])
        * construction_rdp.total_yield_DU_ST
        ) / param["max_land_use"]
    #  Then for the long term, while capping the constructed area at the pixel
    #  size (just in case)
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

    #  NB: note that the pb with previous specification is not that we do not
    #  take formal backyards into account, but rather that we abstract from
    #  backyarding occurring within formal private houses

    actual_backyards = (
        (housing_types.backyard_formal_grid
         + housing_types.backyard_informal_grid)
        / np.nanmax(housing_types.backyard_formal_grid
                    + housing_types.backyard_informal_grid)
        ) * np.max(coeff_land_backyard)
    if options["actual_backyards"] == 0:
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

    #  We also get area for high risk scenario retained in projections
    #  TODO: add option to take into account less likely developments?
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
    coeff_land_RDP = spline_land_RDP(t)  # * param["max_land_use"]
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

    name = 'C_' + options["dem"] + '_' + str(options["slr"])

    # Coastal flood maps are extracted from DELTARES global flood maps that
    # use GTSMip6 water levels as inputs (Muis et al., 2020) for three distinct
    # DEMs. We only consider the two of them that have a fine resolution of
    # 90m / 3'' on a par with FATHOM DATA

    coastal_floods = [name + '_0000', name + '_0002', name + '_0005',
                      name + '_0010', name + '_0025', name + '_0050',
                      name + '_0100', name + '_0250']

    path_data = path_folder + "FATHOM/"

    d_pluvial = {}
    d_fluvial = {}
    d_coastal = {}
    for flood in fluvial_floods:
        print(flood)
        d_fluvial[flood] = np.squeeze(
            pd.read_excel(path_data + flood + ".xlsx")
            )
    for flood in pluvial_floods:
        print(flood)
        d_pluvial[flood] = np.squeeze(
            pd.read_excel(path_data + flood + ".xlsx")
            )
    for flood in coastal_floods:
        print(flood)
        d_coastal[flood] = np.squeeze(
            pd.read_excel(path_data + flood + ".xlsx")
            )

    # Damage functions give damage as a % of good considered, as a function
    # of flood depth in meters

    # Depth-damage functions (from de Villiers, 2007: ratios from table 3, and
    # table 4)
    structural_damages_small_houses = interp1d(
        [0, 0.1, 0.6, 1.2, 2.4, 6, 10],
        [0, 0.0479, 0.1317, 0.1795, 0.3591, 1, 1]
        )
    structural_damages_medium_houses = interp1d(
        [0, 0.1, 0.6, 1.2, 2.4, 6, 10],
        [0, 0.0830, 0.2273, 0.3083, 0.6166, 1, 1]
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

    # If WBUS2 is used, we take the max from the two sources to be conservative
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
            d_fluvial, d_pluvial, d_coastal)


def compute_fraction_capital_destroyed(d, type_flood, damage_function,
                                       housing_type, options):
    """Define function used to get fraction of capital destroyed by floods."""
    if type_flood == 'P' or type_flood == 'FD':
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

        # We consider that formal housing is not vulnerable to pluvial floods
        # over medium run, and that RDP and backyard are not over short run.
        # This is based on CCT Minimum Standards for Stormwater Design 2014
        # (p.37) and Govender 2011 (fig.8 and p.30)
        if options["correct_pluvial"] == 1:
            if ((type_flood == 'P') & (housing_type == 'formal')):
                d[type_flood + '_5yr'].prop_flood_prone = np.zeros(24014)
                d[type_flood + '_10yr'].prop_flood_prone = np.zeros(24014)
                d[type_flood + '_20yr'].prop_flood_prone = np.zeros(24014)
                d[type_flood + '_5yr'].flood_depth = np.zeros(24014)
                d[type_flood + '_10yr'].flood_depth = np.zeros(24014)
                d[type_flood + '_20yr'].flood_depth = np.zeros(24014)
            elif ((type_flood == 'P')
                  & ((housing_type == 'subsidized')
                     | (housing_type == 'backyard'))):
                d[type_flood + '_5yr'].prop_flood_prone = np.zeros(24014)
                d[type_flood + '_10yr'].prop_flood_prone = np.zeros(24014)
                d[type_flood + '_5yr'].flood_depth = np.zeros(24014)
                d[type_flood + '_10yr'].flood_depth = np.zeros(24014)

        # Damage scenarios are incremented using damage functions multiplied by
        # flood-prone area (yields pixel share of destructed area), so as to
        # define damage intervals to be used in final computation

        # TODO: should we take zero value at t = 0?
        # TODO: is continuous time risk the right framework?
        damages0 = (d[type_flood + '_5yr'].prop_flood_prone
                    * damage_function(d[type_flood + '_5yr'].flood_depth))
        # damages0 = (
        #     d[type_flood + '_5yr'].prop_flood_prone
        #     * damage_function(d[type_flood + '_5yr'].flood_depth)
        #     + d[type_flood + '_5yr'].prop_flood_prone
        #     * damage_function(d[type_flood + '_10yr'].flood_depth)
        #     )
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
                       * damage_function(d[type_flood + '_100yr'].flood_depth))
                    )
        damages6 = ((d[type_flood + '_100yr'].prop_flood_prone
                     * damage_function(d[type_flood + '_100yr'].flood_depth))
                    + (d[type_flood + '_200yr'].prop_flood_prone
                       * damage_function(d[type_flood + '_200yr'].flood_depth))
                    )
        damages7 = ((d[type_flood + '_200yr'].prop_flood_prone
                     * damage_function(d[type_flood + '_200yr'].flood_depth))
                    + (d[type_flood + '_250yr'].prop_flood_prone
                       * damage_function(d[type_flood + '_250yr'].flood_depth))
                    )
        damages8 = ((d[type_flood + '_250yr'].prop_flood_prone
                     * damage_function(d[type_flood + '_250yr'].flood_depth))
                    + (d[type_flood + '_500yr'].prop_flood_prone
                       * damage_function(d[type_flood + '_500yr'].flood_depth))
                    )
        damages9 = ((d[type_flood + '_500yr'].prop_flood_prone
                     * damage_function(d[type_flood + '_500yr'].flood_depth))
                    + (d[type_flood + '_1000yr'].prop_flood_prone
                       * damage_function(d[type_flood + '_1000yr'].flood_depth)
                       ))
        # We assume that value stays the same when t = +inf
        damages10 = ((d[type_flood + '_1000yr'].prop_flood_prone
                      * damage_function(d[type_flood + '_1000yr'].flood_depth))
                     + (d[type_flood + '_1000yr'].prop_flood_prone
                        * damage_function(d[type_flood + '_1000yr'].flood_depth
                                          )))

        # The formula for expected fraction of capital destroyed is given by
        # the integral of damage according to time (or rather, inverse
        # probability).
        # Assuming that damages increase linearly with time (or inverse
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

    elif type_flood == 'C':
        interval0 = 1 - (1/2)
        interval1 = (1/2) - (1/5)
        interval2 = (1/5) - (1/10)
        interval3 = (1/10) - (1/25)
        interval4 = (1/25) - (1/50)
        interval5 = (1/50) - (1/100)
        interval6 = (1/100) - (1/250)
        interval7 = (1/250)
    # NB: note that we do not use the same intervals for coastal as for
    # pluvial/fluvial since we do not have the same return periods available
    # in Deltares and FATHOM data

        name = type_flood + '_' + options["dem"] + '_' + str(options["slr"])
        damages0 = ((d[name + '_0000'].prop_flood_prone
                     * damage_function(d[name + '_0000'].flood_depth))
                    + (d[name + '_0002'].prop_flood_prone
                       * damage_function(d[name + '_0002'].flood_depth)))
        damages1 = ((d[name + '_0002'].prop_flood_prone
                     * damage_function(d[name + '_0002'].flood_depth))
                    + (d[name + '_0005'].prop_flood_prone
                       * damage_function(d[name + '_0005'].flood_depth)))
        damages2 = ((d[name + '_0005'].prop_flood_prone
                     * damage_function(d[name + '_0005'].flood_depth))
                    + (d[name + '_0010'].prop_flood_prone
                       * damage_function(d[name + '_0010'].flood_depth)))
        damages3 = ((d[name + '_0010'].prop_flood_prone
                     * damage_function(d[name + '_0010'].flood_depth))
                    + (d[name + '_0025'].prop_flood_prone
                       * damage_function(d[name + '_0025'].flood_depth)))
        damages4 = ((d[name + '_0025'].prop_flood_prone
                     * damage_function(d[name + '_0025'].flood_depth))
                    + (d[name + '_0050'].prop_flood_prone
                       * damage_function(d[name + '_0050'].flood_depth)))
        damages5 = ((d[name + '_0050'].prop_flood_prone
                     * damage_function(d[name + '_0050'].flood_depth))
                    + (d[name + '_0100'].prop_flood_prone
                       * damage_function(d[name + '_0100'].flood_depth)))
        damages6 = ((d[name + '_0100'].prop_flood_prone
                     * damage_function(d[name + '_0100'].flood_depth))
                    + (d[name + '_0250'].prop_flood_prone
                       * damage_function(d[name + '_0250'].flood_depth)))
        damages7 = ((d[name + '_0250'].prop_flood_prone
                     * damage_function(d[name + '_0250'].flood_depth))
                    + (d[name + '_0250'].prop_flood_prone
                       * damage_function(d[name + '_0250'].flood_depth)))

        return (0.5
                * ((interval0 * damages0) + (interval1 * damages1)
                   + (interval2 * damages2) + (interval3 * damages3)
                   + (interval4 * damages4) + (interval5 * damages5)
                   + (interval6 * damages6) + (interval7 * damages7)))


def import_full_floods_data(options, param, path_folder, housing_type_data):
    """Add fraction of capital destroyed by floods to initial floods data."""
    fraction_capital_destroyed = pd.DataFrame()

    (structural_damages_small_houses, structural_damages_medium_houses,
     structural_damages_large_houses, content_damages,
     structural_damages_type1, structural_damages_type2,
     structural_damages_type3a, structural_damages_type3b,
     structural_damages_type4a, structural_damages_type4b,
     d_fluvial, d_pluvial, d_coastal) = import_init_floods_data(
         options, param, path_folder)

    if options["pluvial"] == 0 and options["coastal"] == 0:
        print("Contents in private formal")
        (fraction_capital_destroyed["contents_formal"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'formal', options)
        print("Contents in informal settlements")
        (fraction_capital_destroyed["contents_informal"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'informal', options)
        print("Contents in (any) backyard")
        (fraction_capital_destroyed["contents_backyard"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'backyard', options)
        print("Contents in formal subsidized")
        (fraction_capital_destroyed["contents_subsidized"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'subsidized', options)
        print("Private formal structures (one floor)")
        (fraction_capital_destroyed["structure_formal_1"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'formal', options)
        print("Private formal structures (two floors)")
        (fraction_capital_destroyed["structure_formal_2"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'formal', options)
        print("Formal subsidized structures (one floor)")
        (fraction_capital_destroyed["structure_subsidized_1"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'subsidized', options)
        print("Formal subsidized structures (two floors)")
        (fraction_capital_destroyed["structure_subsidized_2"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'subsidized', options)
        print("Informal settlement structures")
        (fraction_capital_destroyed["structure_informal_settlements"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'informal', options)
        print("Informal backyard structures")
        (fraction_capital_destroyed["structure_informal_backyards"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'backyard', options)
        print("Formal backyard structures (one floor)")
        (fraction_capital_destroyed["structure_formal_backyards"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type3a, 'backyard', options)
        print("Formal backyard structures (two floors)")
        (fraction_capital_destroyed["structure_formal_backyards"]
         ) = compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type3b, 'backyard', options)

    elif options["pluvial"] == 1 and options["coastal"] == 0:
        print("Contents in private formal")
        (fraction_capital_destroyed["contents_formal"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'formal', options))
        print("Contents in informal settlements")
        (fraction_capital_destroyed["contents_informal"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'informal', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'informal', options))
        print("Contents in (any) backyard")
        (fraction_capital_destroyed["contents_backyard"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'backyard', options))
        print("Contents in formal subsidized")
        (fraction_capital_destroyed["contents_subsidized"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'subsidized', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'subsidized', options))
        print("Private formal structures (one floor)")
        (fraction_capital_destroyed["structure_formal_1"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4a, 'formal', options))
        print("Private formal structures (two floors)")
        (fraction_capital_destroyed["structure_formal_2"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4b, 'formal', options))
        print("Formal subsidized structures (one floor)")
        (fraction_capital_destroyed["structure_subsidized_1"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'subsidized', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4a, 'subsidized',
                 options))
        print("Formal subsidized structures (two floors)")
        (fraction_capital_destroyed["structure_subsidized_2"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'subsidized', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4b, 'subsidized',
                 options))
        print("Informal settlement structures")
        (fraction_capital_destroyed["structure_informal_settlements"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'informal', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type2, 'informal',
                 options))
        print("Informal backyard structures")
        (fraction_capital_destroyed["structure_informal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type2, 'backyard',
                 options))
        print("Formal backyard structures (one floor)")
        (fraction_capital_destroyed["structure_formal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type3a, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type3a, 'backyard',
                 options))
        print("Formal backyard structures (two floors)")
        (fraction_capital_destroyed["structure_formal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type3b, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type3b, 'backyard',
                 options))

    elif options["pluvial"] == 0 and options["coastal"] == 1:
        print("Contents in private formal")
        (fraction_capital_destroyed["contents_formal"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', content_damages, 'formal', options))
        print("Contents in informal settlements")
        (fraction_capital_destroyed["contents_informal"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'informal', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', content_damages, 'informal', options))
        print("Contents in (any) backyard")
        (fraction_capital_destroyed["contents_backyard"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', content_damages, 'backyard', options))
        print("Contents in formal subsidized")
        (fraction_capital_destroyed["contents_subsidized"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'subsidized', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', content_damages, 'subsidized', options))
        print("Private formal structures (one floor)")
        (fraction_capital_destroyed["structure_formal_1"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type4a, 'formal', options))
        print("Private formal structures (two floors)")
        (fraction_capital_destroyed["structure_formal_2"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'P', structural_damages_type4b, 'formal', options))
        print("Formal subsidized structures (one floor)")
        (fraction_capital_destroyed["structure_subsidized_1"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'subsidized', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type4a, 'subsidized',
                 options))
        print("Formal subsidized structures (two floors)")
        (fraction_capital_destroyed["structure_subsidized_2"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'subsidized', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type4b, 'subsidized',
                 options))
        print("Informal settlement structures")
        (fraction_capital_destroyed["structure_informal_settlements"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'informal', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type2, 'informal',
                 options))
        print("Informal backyard structures")
        (fraction_capital_destroyed["structure_informal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type2, 'backyard',
                 options))
        print("Formal backyard structures (one floor)")
        (fraction_capital_destroyed["structure_formal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type3a, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type3a, 'backyard',
                 options))
        print("Formal backyard structures (two floors)")
        (fraction_capital_destroyed["structure_formal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type3b, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type3b, 'backyard',
                 options))

    elif options["pluvial"] == 1 and options["coastal"] == 1:
        print("Contents in private formal")
        (fraction_capital_destroyed["contents_formal"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', content_damages, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'formal', options))
        print("Contents in informal settlements")
        (fraction_capital_destroyed["contents_informal"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'informal', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', content_damages, 'informal', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'informal', options))
        print("Contents in (any) backyard")
        (fraction_capital_destroyed["contents_backyard"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', content_damages, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'backyard', options))
        print("Contents in formal subsidized")
        (fraction_capital_destroyed["contents_subsidized"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', content_damages, 'subsidized', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', content_damages, 'subsidized', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', content_damages, 'subsidized', options))
        print("Private formal structures (one floor)")
        (fraction_capital_destroyed["structure_formal_1"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type4a, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4a, 'formal', options))
        print("Private formal structures (two floors)")
        (fraction_capital_destroyed["structure_formal_2"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type4b, 'formal', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4b, 'formal', options))
        print("Formal subsidized structures (one floor)")
        (fraction_capital_destroyed["structure_subsidized_1"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4a, 'subsidized', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type4a, 'subsidized',
                 options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4a, 'subsidized',
                 options))
        print("Formal subsidized structures (two floors)")
        (fraction_capital_destroyed["structure_subsidized_2"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type4b, 'subsidized', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type4b, 'subsidized',
                 options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type4b, 'subsidized',
                 options))
        print("Informal settlement structures")
        (fraction_capital_destroyed["structure_informal_settlements"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'informal', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type2, 'informal', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type2, 'informal',
                 options))
        print("Informal backyard structures")
        (fraction_capital_destroyed["structure_informal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type2, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type2, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type2, 'backyard',
                 options))
        print("Formal backyard structures (one floor)")
        (fraction_capital_destroyed["structure_formal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type3a, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type3a, 'backyard',
                 options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type3a, 'backyard',
                 options))
        print("Formal backyard structures (two floors)")
        (fraction_capital_destroyed["structure_formal_backyards"]
         ) = (compute_fraction_capital_destroyed(
             d_fluvial, 'FD', structural_damages_type3b, 'backyard', options)
             + compute_fraction_capital_destroyed(
                 d_coastal, 'C', structural_damages_type3b, 'backyard',
                 options)
             + compute_fraction_capital_destroyed(
                 d_pluvial, 'P', structural_damages_type3b, 'backyard',
                 options))

    # We take a weighted average for structures in bricks and shacks among
    # all backyard structures in case we include both in the model.
    # To do so, we rely on SAL-level housing type data.

    sal_data = pd.read_excel(
        path_folder
        + "CT Dwelling type data validation workbook 20201204 v2.xlsx",
        header=6, nrows=5339)
    sal_data["backyard_formal"] = sal_data["House/flat/room in backyard"]
    sal_data["backyard_informal"] = sal_data[
        "Informal dwelling (shack; in backyard)"]
    total_backyard_formal = sal_data["backyard_formal"].sum()
    total_backyard_informal = sal_data["backyard_informal"].sum()

    (fraction_capital_destroyed["structure_backyards"]
     ) = ((total_backyard_formal
           * fraction_capital_destroyed["structure_formal_backyards"])
          + (total_backyard_informal
              * fraction_capital_destroyed["structure_informal_backyards"])
          ) / (total_backyard_formal + total_backyard_informal)

    return (fraction_capital_destroyed, structural_damages_small_houses,
            structural_damages_medium_houses, structural_damages_large_houses,
            content_damages, structural_damages_type1,
            structural_damages_type2, structural_damages_type3a,
            structural_damages_type3b, structural_damages_type4a,
            structural_damages_type4b)


def infer_WBUS2_depth(housing_types, param, path_floods):
    """Update CoCT flood data with FATHOM flood depth (deprecated)."""
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
                          path_precalc_inp, path_precalc_transp, dim, options):
    """Compute job center distribution, commuting and net income."""
    # Import (monetary and time) transport costs
    (timeOutput, distanceOutput, monetaryCost, costTime
     ) = calcmp.import_transport_costs(
         grid, param, yearTraffic, households_per_income_class,
         spline_inflation, spline_fuel, spline_population_income_distribution,
         spline_income_distribution, path_precalc_inp, path_precalc_transp,
         dim, options)

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

    # Update average income and number of households per income group
    # for considered year
    incomeGroup, households_per_income_class = eqdyn.compute_average_income(
        spline_population_income_distribution, spline_income_distribution,
        param, yearTraffic)

    # We import expected income associated with each income center and income
    # group (from calibration)
    if options["load_precal_param"] == 1:
        income_centers_init = scipy.io.loadmat(
            path_precalc_inp + 'incomeCentersKeep.mat')['incomeCentersKeep']
    elif options["load_precal_param"] == 0:
        income_centers_init = np.load(
            path_precalc_inp + 'incomeCentersKeep.npy')
    # This allows to correct incomes for unemployed population not taken into
    # account in initial income data (just in scenarios)
    incomeCenters = income_centers_init * incomeGroup / average_income

    # Switch to hourly
    annualToHourly = 1 / (8*20*12)
    monetaryCost = monetaryCost * annualToHourly
    incomeCenters = incomeCenters * annualToHourly

    for j in range(0, param["nb_of_income_classes"]):

        # Household size varies with income group / transport costs
        householdSize = param["household_size"][j]
        # Here, -100000 corresponds to an arbitrary value given to incomes in
        # centers with too few jobs to have convergence in calibration (could
        # have been nan): we exclude those centers from the analysis
        whichCenters = incomeCenters[:, j] > -100000
        incomeCentersGroup = incomeCenters[whichCenters, j]

        # We compute transport costs for each mode and per chosen mode
        # (cost per hour)

        (transportCostModes, transportCost, _, valueMax, minIncome
         ) = calcmp.compute_ODflows(
            householdSize, monetaryCost, costTime, incomeCentersGroup,
            whichCenters, param_lambda)

        # NB: we compute OD flows again later to get the full matrix

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
        # Note that this should take unemployment into account
        averageIncome[j, :] = np.nansum(
            ODflows[whichCenters, :, j] * incomeCentersGroup[:, None], 0)

    # We go back to yearly format before saving results
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
        header=6, nrows=5339)
    sal_data["informal"] = sal_data[
        "Informal dwelling (shack; not in backyard; e.g. in an"
        + " informal/squatter settlement or on a farm)"]
    sal_data["backyard_formal"] = sal_data["House/flat/room in backyard"]
    sal_data["backyard_informal"] = sal_data[
        "Informal dwelling (shack; in backyard)"]
    # NB: we do not include traditional houses, granny flats, caravans, and
    # others
    sal_data["formal"] = np.nansum(sal_data.iloc[:, [3, 5, 6, 7, 8]], 1)

    sal_data = pd.read_excel(
        path_folder
        + "CT Dwelling type data validation workbook 20201204 v2.xlsx",
        header=6, nrows=5339)
    sal_data["informal"] = sal_data[
        "Informal dwelling (shack; not in backyard; e.g. in an"
        + " informal/squatter settlement or on a farm)"]
    sal_data["backyard_formal"] = sal_data["House/flat/room in backyard"]
    sal_data["backyard_informal"] = sal_data[
        "Informal dwelling (shack; in backyard)"]
    # NB: we do not include traditional houses, granny flats, caravans, and
    # others
    sal_data["formal"] = np.nansum(sal_data.iloc[:, [3, 5, 6, 7, 8]], 1)

    # We import information on intersection between SAL and grid pixels
    grid_intersect = pd.read_csv(
        path_data + 'grid_SAL_intersect.csv', sep=';')
    grid_intersect.rename(columns={"Area_inter": "Area"}, inplace=True)

    # We then proceed to the disaggregation of SAL data
    print("Informal settlements")
    informal_grid = gen_small_areas_to_grid(
        grid, grid_intersect, sal_data["informal"],
        sal_data["Small Area Code"], 'SAL')
    print("Formal backyards")
    backyard_formal_grid = gen_small_areas_to_grid(
        grid, grid_intersect, sal_data["backyard_formal"],
        sal_data["Small Area Code"], 'SAL')
    print("Informal backyards")
    backyard_informal_grid = gen_small_areas_to_grid(
        grid, grid_intersect, sal_data["backyard_informal"],
        sal_data["Small Area Code"], 'SAL')
    # NB: this does include RDP
    print("Private formal + subsidized formal housing")
    formal_grid = gen_small_areas_to_grid(
        grid, grid_intersect, sal_data["formal"], sal_data["Small Area Code"],
        'SAL')

    # We correct the number of dwellings per pixel by reweighting with the
    # ratio of total original number over total estimated number
    # NB: this allows to correct for potential errors or imperfect information
    # in the matching of small areas and grid pixels
    informal_grid = (informal_grid * (np.nansum(sal_data["informal"])
                                      / np.nansum(informal_grid)))
    backyard_formal_grid = (backyard_formal_grid
                            * (np.nansum(sal_data["backyard_formal"])
                               / np.nansum(backyard_formal_grid)))
    backyard_informal_grid = (backyard_informal_grid
                              * (np.nansum(sal_data["backyard_informal"])
                                 / np.nansum(backyard_informal_grid)))
    formal_grid = (formal_grid * (np.nansum(sal_data["formal"])
                                  / np.nansum(formal_grid)))

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


def gen_small_areas_to_grid(grid, grid_intersect, small_area_data,
                            small_area_code, unit):
    """Convert SAL/SP to grid dimensions."""
    grid_data = np.zeros(len(grid.dist))
    # We loop over grid pixels
    print("Looping over pixels")
    for index in range(0, len(grid.dist)):
        print(index)
        # We define a list of SAL/SP codes that do intersect considered pixel
        intersect = np.unique(
            grid_intersect[unit + '_CODE'][grid_intersect.ID_grille
                                           == grid.id[index]]
            )
        if len(intersect) == 0:
            grid_data[index] = np.nan
        else:
            # If the intersection is not empty, we loop over intersecting
            # SAL/SPs
            for i in range(0, len(intersect)):
                small_code = intersect[i]
                # Then, we get the corresponding area for the sub-intersection
                # and for the overall SAL/SP
                small_area_intersect = np.nansum(
                    grid_intersect.Area[
                        (grid_intersect.ID_grille == grid.id[index])
                        & (grid_intersect[unit + '_CODE'] == small_code)
                        ].squeeze())
                small_area = np.nansum(
                    grid_intersect.Area[
                        (grid_intersect[unit + '_CODE'] == small_code)]
                    )
                # If such SAL/SP code does indeed exist in the matching data,
                # we store a weighted average of the info it contains
                if len(small_area_data[
                        small_area_code == small_code]) > 0:
                    # More precisely, this yields the number of
                    # dwellings/households given by the intersection
                    add = (small_area_data[small_area_code == small_code]
                           * (small_area_intersect / small_area))
                else:
                    add = 0
                # Finally, we update our output data with each weighted info
                # corresponding to each of the SAL/SP intersection with
                # considered grid pixel
                grid_data[index] = grid_data[index] + add

    return grid_data
