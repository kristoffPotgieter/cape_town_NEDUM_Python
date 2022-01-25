# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:57:41 2020.

@author: Charlotte Liotta
"""

import numpy as np
import scipy.io
import pandas as pd
from scipy.interpolate import interp1d
import copy


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


def import_amenities(precalculated_inputs):
    """Import amenity index for each pixel."""
    # Follow calibration from Pfeiffer et al. (appendix C4)
    precalculated_amenities = scipy.io.loadmat(
        precalculated_inputs + 'calibratedAmenities.mat')
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


def import_income_classes_data(param, income_2011):
        
    nb_of_hh_bracket = income_2011.Households_nb
    avg_income_bracket = income_2011.INC_med    
    average_income = np.zeros(param["nb_of_income_classes"])
    households_per_income_class = np.zeros(param["nb_of_income_classes"])
    for j in range(0, param["nb_of_income_classes"]):
        households_per_income_class[j] = np.sum(nb_of_hh_bracket[(param["income_distribution"] == j + 1)])
        average_income[j] = np.sum(avg_income_bracket[(param["income_distribution"] == j + 1)] * nb_of_hh_bracket[param["income_distribution"] == j + 1]) / households_per_income_class[j]

    return households_per_income_class, average_income


def import_housig_limit(grid, param):
    center_regulation = (grid["dist"] <= param["historic_radius"])
    outside_regulation = (grid["dist"] > param["historic_radius"])
    housing_limit = param["limit_height_center"] * 1000000 * center_regulation + param["limit_height_out"] * 1000000 * outside_regulation 
    return housing_limit


def import_households_data(options, precalculated_inputs):
        
        if options["load_households_data"] == 0:
            data = scipy.io.loadmat(precalculated_inputs + 'data.mat')['data']
        
        data_rdp = pd.DataFrame()
        data_rdp["count"] = data['gridCountRDPfromGV'][0][0].squeeze() #Number of subsidized housing, 24014
        data_rdp["area"] = data['gridAreaRDPfromGV'][0][0].squeeze() #Surface of subsidized housing, 24014
        
        housing_types_sp = pd.DataFrame()
        housing_types_sp["backyard_SP_2011"] = data['spInformalBackyard'][0][0].squeeze() #Number of informal settlements in backyard per grid cell, 24014
        housing_types_sp["informal_SP_2011"] = data['spInformalSettlement'][0][0].squeeze() #Number of informal settlements in backyard per SP division, 1046. 85007 in total.
        housing_types_sp["total_dwellings_SP_2011"] = data['spTotalDwellings'][0][0].squeeze() #Number of dwellings per administrative division, 1046
        housing_types_sp["x_sp"] = data['spX'][0][0].squeeze()
        housing_types_sp["y_sp"] = data['spY'][0][0].squeeze()
        
        #housing_types_grid = pd.DataFrame()
        #housing_types_grid["backyard_grid_2011"] = data['gridInformalBackyard'][0][0].squeeze() #Number of informal settlements in backyard per grid cell, 24014
        #housing_types_grid["formal_grid_2011"] = data['gridFormal'][0][0].squeeze()
        #housing_types_grid["informal_grid_2011"] = data['gridInformalSettlement'][0][0].squeeze() #Number of informal settlements in backyard per grid cell, 24014
        
        data_sp = pd.DataFrame()
        data_sp["dwelling_size"] = data['spDwellingSize'][0][0].squeeze()
        data_sp["price"] = data['spPrice'][0][0].squeeze()[2, :]
        data_sp["income"] = data['sp2011AverageIncome'][0][0].squeeze()
        data_sp["unconstrained_area"] = data["spUnconstrainedArea"][0][0].squeeze()
        data_sp["area"] = data["sp2011Area"][0][0].squeeze()
        data_sp["distance"] = data["sp2011Distance"][0][0].squeeze()
        data_sp["mitchells_plain"] = data["sp2011MitchellsPlain"][0][0].squeeze()
        data_sp["sp_code"] = data["spCode"][0][0].squeeze()
        
        mitchells_plain_grid_2011 = data['MitchellsPlain'][0][0].squeeze() #Mitchells Plain or not ? True or False, 220/24014   
        grid_formal_density_HFA = data['gridFormalDensityHFA'][0][0].squeeze()
        threshold_income_distribution = data['thresholdIncomeDistribution'][0][0].squeeze()
        income_distribution = data["sp2011IncomeDistributionNClass"][0][0].squeeze()
        cape_town_limits = data["sp2011CapeTown"][0][0].squeeze()
        
        return data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, threshold_income_distribution, income_distribution, cape_town_limits
    
def import_land_use(grid, options, param, data_rdp, housing_types, total_RDP, path_data, path_folder):
    
    area_pixel = (0.5 ** 2) * 1000000

    #0. Import Land Cover Data (see R code for details)
    land_use_data_old = pd.read_csv(path_data + 'grid_NEDUM_Cape_Town_500.csv', sep = ';')
    informal_risks_short = pd.read_csv(path_folder + 'Land occupation/informal_settlements_risk_SHORT.csv', sep = ',')
    informal_risks_long = pd.read_csv(path_folder + 'Land occupation/informal_settlements_risk_LONG.csv', sep = ',')
    informal_risks_medium = pd.read_csv(path_folder + 'Land occupation/informal_settlements_risk_MEDIUM.csv', sep = ',')
    informal_risks_VERYHIGH = pd.read_csv(path_folder + 'Land occupation/informal_settlements_risk_pVERYHIGH.csv', sep = ',')
    informal_risks_HIGH = pd.read_csv(path_folder + 'Land occupation/informal_settlements_risk_pHIGH.csv', sep = ',')
    informal_settlements_2020 = pd.read_excel(path_folder + 'Flood plains - from Claus/inf_dwellings_2020.xlsx')
    informal_risks_2020 = informal_settlements_2020.inf_dwellings_2020 * param["shack_size"] * (1 / param["max_land_use_settlement"])
    informal_risks_2020[informal_risks_2020 < 2500] = 0
    informal_risks_2020[np.isnan(informal_risks_2020)] = 0
    
    polygon_medium_timing = pd.read_excel(path_folder + 'Land occupation/polygon_medium_timing.xlsx', header = None)
    
    for item in list(polygon_medium_timing.squeeze()):
        informal_risks_medium.area[informal_risks_medium["grid.data.ID"] == item] = informal_risks_short.area[informal_risks_short["grid.data.ID"] == item]
        informal_risks_short.area[informal_risks_short["grid.data.ID"] == item] = 0
    
    coeff_land_no_urban_edge = (np.transpose(land_use_data_old.unconstrained_out) + np.transpose(land_use_data_old.unconstrained_UE)) / area_pixel
    coeff_land_urban_edge = np.transpose(land_use_data_old.unconstrained_UE) / area_pixel
    
    #2. Backyard
    
    area_backyard = data_rdp["area"] * param["backyard_size"] / (param["backyard_size"] + param["RDP_size"]) / area_pixel            
    urban = np.transpose(land_use_data_old.urban) / area_pixel
    coeff_land_backyard = np.fmin(urban, area_backyard)
    actual_backyards = ((housing_types.backyard_formal_grid + housing_types.backyard_informal_grid) / np.nanmax(housing_types.backyard_formal_grid + housing_types.backyard_informal_grid)) * np.max(coeff_land_backyard)
    coeff_land_backyard = np.fmax(coeff_land_backyard, actual_backyards)
    coeff_land_backyard = coeff_land_backyard * param["max_land_use_backyard"]
    coeff_land_backyard[coeff_land_backyard < 0] = 0
    
    #3. RDP
    coeff_land_RDP = np.ones(len(coeff_land_backyard))
      
    #Area RDP/Backyard
    RDP_houses_estimates = data_rdp["count"] #actual nb of RDP houses
    area_RDP = data_rdp["area"] * param["RDP_size"] / (param["backyard_size"] + param["RDP_size"]) / area_pixel   
    number_properties_2000 = data_rdp["count"] * (1 - grid.dist / max(grid.dist[data_rdp["count"] > 0]))
    construction_rdp = pd.read_csv(path_data + 'grid_new_RDP_projects.csv')            
    
    year_begin_RDP = 2015
    year_RDP = np.arange(year_begin_RDP, 2040) - param["baseline_year"]
    
    #RDP_2011 = 2.2666e+05 #(estimated as sum(data.gridFormal(data.countRDPfromGV > 0)))    % RDP_2011 = 320969; %227409; % Where from?
    RDP_2011 = total_RDP
    RDP_2001 = 1.1718e+05 #(estimated as sum(data.gridFormal(data.countRDPfromGV > 0)))  % 262452; % Estimated by nb inc_1 - BY - settlement in 2001
    spline_RDP = interp1d([2001 - param["baseline_year"], 2011 - param["baseline_year"], 2020 -  param["baseline_year"], 2041 - param["baseline_year"]], [RDP_2001, RDP_2011, RDP_2011 + 9*5000, RDP_2011 + 9*5000 + 21 * param["future_rate_public_housing"]], 'linear')        
    number_RDP = spline_RDP(year_RDP)           
        
    year_short_term = np.argmin(np.abs(sum(construction_rdp.total_yield_DU_ST) - (number_RDP - number_RDP[0])))
    #year_long_term = np.argmin(np.abs(sum(construction_rdp.total_yield_DU_LT + construction_rdp.total_yield_DU_ST) - (number_RDP - number_RDP[0])))
    year_long_term = 30
    area_RDP_short_term = np.minimum(construction_rdp.area_ST, (param["backyard_size"] + param["RDP_size"]) * construction_rdp.total_yield_DU_ST)
    area_RDP_long_term = np.minimum(np.minimum(construction_rdp.area_ST + construction_rdp.area_LT, (param["backyard_size"] + param["RDP_size"]) * (construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT)), area_pixel)

    area_backyard_short_term = area_backyard + np.maximum(area_RDP_short_term - construction_rdp.total_yield_DU_ST * param["RDP_size"], 0) / area_pixel
    area_RDP_short_term = area_RDP + np.minimum(construction_rdp.total_yield_DU_ST * param["RDP_size"], construction_rdp.area_ST) / area_pixel
    area_backyard_short_term = np.minimum(area_backyard_short_term, param["max_land_use"] - area_RDP_short_term)
    area_backyard_long_term = area_backyard + np.maximum(area_RDP_long_term - (construction_rdp.total_yield_DU_LT + construction_rdp.total_yield_DU_ST) * param["RDP_size"], 0) / area_pixel
    area_RDP_long_term = area_RDP + np.minimum((construction_rdp.total_yield_DU_LT + construction_rdp.total_yield_DU_ST) * param["RDP_size"], area_RDP_long_term) / area_pixel
    area_backyard_long_term = np.minimum(area_backyard_long_term, param["max_land_use"] - area_RDP_long_term)

    year_data_informal = [2000 - param["baseline_year"], year_begin_RDP - param["baseline_year"], year_short_term, year_long_term]
    spline_land_backyard = interp1d(year_data_informal,  np.transpose([np.fmax(area_backyard, actual_backyards), np.fmax(area_backyard, actual_backyards), np.fmax(area_backyard_short_term, actual_backyards), np.fmax(area_backyard_long_term, actual_backyards)]), 'linear')
        
    spline_land_RDP = interp1d(year_data_informal,  np.transpose([area_RDP, area_RDP, area_RDP_short_term, area_RDP_long_term]), 'linear')
    spline_estimate_RDP = interp1d(year_data_informal, np.transpose([number_properties_2000, RDP_houses_estimates, RDP_houses_estimates + construction_rdp.total_yield_DU_ST, RDP_houses_estimates + construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT]), 'linear')

    #1. Informal
    
    informal_2011 = np.transpose(land_use_data_old.informal) / area_pixel
    
    informal_2020 = np.fmax(informal_risks_2020 / area_pixel, informal_2011)
    
    high_proba = informal_risks_VERYHIGH.area + informal_risks_HIGH.area
    
    informal_2023 = np.fmin(coeff_land_no_urban_edge, np.fmax(informal_2020, np.transpose(np.fmin(informal_risks_short.area, high_proba)) / area_pixel)) - spline_land_RDP(12)
    informal_2023[informal_2023 < 0] = 0

    informal_2025 = np.fmin(coeff_land_no_urban_edge, np.fmax(informal_2023, np.transpose(np.fmin(informal_risks_medium.area, high_proba)) / area_pixel)) - spline_land_RDP(14)
    informal_2025[informal_2025 < 0] = 0
    
    informal_2030 = np.fmin(coeff_land_no_urban_edge, np.fmax(informal_2025, np.transpose(np.fmin(informal_risks_long.area, high_proba)) / area_pixel)) - spline_land_RDP(19)
    informal_2030[informal_2030 < 0] = 0
    
    if options["informal_land_constrained"] == 0:
        spline_land_informal = interp1d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 19, 29],  np.transpose([informal_2011, informal_2011,informal_2011,informal_2011,informal_2011,informal_2011,informal_2011,informal_2011,informal_2011, informal_2020, informal_2023, informal_2025, informal_2030, informal_2030]), 'linear')
    elif options["informal_land_constrained"] == 1:
        spline_land_informal = interp1d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 19, 29],  np.transpose([informal_2011, informal_2011,informal_2011,informal_2011,informal_2011,informal_2011,informal_2011,informal_2011,informal_2011, informal_2020, informal_2020, informal_2020, informal_2020, informal_2020]), 'linear')
    
    # 4. Formal
       
    coeff_land_private_urban_edge = (coeff_land_urban_edge - informal_2011 - area_RDP - np.fmax(area_backyard, actual_backyards)) * param["max_land_use"]
    coeff_land_private_no_urban_edge = (coeff_land_no_urban_edge - informal_2011 - area_RDP - np.fmax(area_backyard, actual_backyards)) * param["max_land_use"]
    coeff_land_private_urban_edge[coeff_land_private_urban_edge < 0] = 0
    coeff_land_private_no_urban_edge[coeff_land_private_no_urban_edge < 0] = 0
        
    if options["urban_edge"] == 0:
        coeff_land_private = coeff_land_private_urban_edge
    else:
        coeff_land_private = coeff_land_private_no_urban_edge
                       
    # 5. Constraints
        
    if options["urban_edge"] == 0:
        year_constraints = np.array([1990, param["year_urban_edge"] - 1, param["year_urban_edge"], 2040]) - param["baseline_year"]
        spline_land_constraints = interp1d(year_constraints, np.transpose(np.array([coeff_land_urban_edge, coeff_land_urban_edge, coeff_land_no_urban_edge, coeff_land_no_urban_edge])), 'linear')
    else:
        year_constraints = np.array([1990, 2040]) - param["baseline_year"]
        spline_land_constraints = interp1d(year_constraints, np.transpose(np.array([coeff_land_urban_edge, coeff_land_urban_edge])))

    return spline_estimate_RDP, spline_land_backyard, spline_land_RDP, spline_RDP, spline_land_constraints, spline_land_informal, coeff_land_backyard

def import_floods_data(options, param, path_folder):
    
    #Import floods data
    fluvial_floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
    pluvial_floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr', 'P_100yr', 'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr']
    path_data = path_folder + "FATHOM/"
    
    d_pluvial = {}
    d_fluvial = {}
    for flood in fluvial_floods:
        type_flood = copy.deepcopy(flood)
        d_fluvial[flood] = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))
    for flood in pluvial_floods:
        type_flood = copy.deepcopy(flood)
        d_pluvial[flood] = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))
    
    #Depth-damage functions
    structural_damages_small_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.0479, 0.1312, 0.1795, 0.3591, 1, 1]) #de Villiers 2007
    structural_damages_medium_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.083, 0.2273, 0.3083, 0.62, 1, 1]) #de Villiers 2007
    structural_damages_large_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.0799, 0.2198, 0.2997, 0.5994, 1, 1]) #de Villiers 2007
    content_damages = interp1d([0, 0.1, 0.3, 0.6, 1.2, 1.5, 2.4, 10], [0, 0.06, 0.15, 0.35, 0.77, 0.95, 1, 1]) #de Villiers 2007
    structural_damages_type1 = interp1d([0, 0.5, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10], [0, 0.5, 0.9, 1, 1,1,1,1,1,1,1,1,1]) #Englhardt 2019
    structural_damages_type2 = interp1d([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10], [0, 0.45, 0.65, 0.82, 0.95, 1, 1, 1,1,1,1,1]) #Englhardt 2019
    structural_damages_type3a = interp1d([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10], [0, 0.4, 0.55, 0.7, 0.78,0.81,0.81,0.81,0.81,0.81,0.81,0.81]) #Englhardt 2019
    structural_damages_type3b = interp1d([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10], [0, 0.25, 0.4,0.48,0.58,0.62,0.65,0.75,0.78,0.8,0.81,0.81]) #Englhardt 2019
    structural_damages_type4a = interp1d([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10], [0, 0.31, 0.45,0.55,0.62,0.65,0.65,0.65,0.65,0.65,0.65,0.65]) #Englhardt 2019
    structural_damages_type4b = interp1d([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10], [0,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.62,0.64,0.65,0.65]) #Englhardt 2019
        
    #if options["WBUS2"] == 1:
    #    WBUS2_20yr = pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Flood plains - from Claus/WBUS2_20yr.xlsx")
    #    WBUS2_50yr = pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Flood plains - from Claus/WBUS2_50yr.xlsx")
    #    WBUS2_100yr = pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Flood plains - from Claus/WBUS2_100yr.xlsx")
    #    d['FD_20yr'].prop_flood_prone = np.fmax(d['FD_20yr'].prop_flood_prone, WBUS2_20yr.prop_flood_prone)
    #    d['FD_50yr'].prop_flood_prone = np.fmax(d['FD_50yr'].prop_flood_prone, WBUS2_50yr.prop_flood_prone)
    #    d['FD_100yr'].prop_flood_prone = np.fmax(d['FD_100yr'].prop_flood_prone, WBUS2_100yr.prop_flood_prone)
    #    d['FD_20yr'].flood_depth = np.maximum(d['FD_20yr'].prop_flood_prone, param["depth_WBUS2_20yr"])
    #    d['FD_50yr'].flood_depth = np.maximum(d['FD_50yr'].prop_flood_prone, param["depth_WBUS2_50yr"])
    #    d['FD_100yr'].flood_depth = np.maximum(d['FD_100yr'].prop_flood_prone, param["depth_WBUS2_100yr"])

    def compute_fraction_capital_destroyed(d, type_flood, damage_function, housing_type):
    
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
        elif (type_flood == 'P') & ((housing_type == 'subsidized') | (housing_type == 'backyard')):
            d[type_flood + '_5yr'].prop_flood_prone = np.zeros(24014)
            d[type_flood + '_10yr'].prop_flood_prone = np.zeros(24014)
            d[type_flood + '_5yr'].flood_depth = np.zeros(24014)
            d[type_flood + '_10yr'].flood_depth = np.zeros(24014)
        
        damages0 = (d[type_flood + '_5yr'].prop_flood_prone * damage_function(d[type_flood + '_5yr'].flood_depth)) + (d[type_flood + '_5yr'].prop_flood_prone * damage_function(d[type_flood + '_10yr'].flood_depth))
        damages1 = (d[type_flood + '_5yr'].prop_flood_prone * damage_function(d[type_flood + '_5yr'].flood_depth)) + (d[type_flood + '_10yr'].prop_flood_prone * damage_function(d[type_flood + '_10yr'].flood_depth))
        damages2 = (d[type_flood + '_10yr'].prop_flood_prone * damage_function(d[type_flood + '_10yr'].flood_depth)) + (d[type_flood + '_20yr'].prop_flood_prone * damage_function(d[type_flood + '_20yr'].flood_depth))
        damages3 = (d[type_flood + '_20yr'].prop_flood_prone * damage_function(d[type_flood + '_20yr'].flood_depth)) + (d[type_flood + '_50yr'].prop_flood_prone * damage_function(d[type_flood + '_50yr'].flood_depth))
        damages4 = (d[type_flood + '_50yr'].prop_flood_prone * damage_function(d[type_flood + '_50yr'].flood_depth)) + (d[type_flood + '_75yr'].prop_flood_prone * damage_function(d[type_flood + '_75yr'].flood_depth))
        damages5 = (d[type_flood + '_75yr'].prop_flood_prone * damage_function(d[type_flood + '_75yr'].flood_depth)) + (d[type_flood + '_100yr'].prop_flood_prone * damage_function(d[type_flood + '_100yr'].flood_depth))
        damages6 = (d[type_flood + '_100yr'].prop_flood_prone * damage_function(d[type_flood + '_100yr'].flood_depth)) + (d[type_flood + '_200yr'].prop_flood_prone * damage_function(d[type_flood + '_200yr'].flood_depth))
        damages7 = (d[type_flood + '_200yr'].prop_flood_prone * damage_function(d[type_flood + '_200yr'].flood_depth)) + (d[type_flood + '_250yr'].prop_flood_prone * damage_function(d[type_flood + '_250yr'].flood_depth))
        damages8 = (d[type_flood + '_250yr'].prop_flood_prone * damage_function(d[type_flood + '_250yr'].flood_depth)) + (d[type_flood + '_500yr'].prop_flood_prone * damage_function(d[type_flood + '_500yr'].flood_depth))
        damages9 = (d[type_flood + '_500yr'].prop_flood_prone * damage_function(d[type_flood + '_500yr'].flood_depth)) + (d[type_flood + '_1000yr'].prop_flood_prone * damage_function(d[type_flood + '_1000yr'].flood_depth))
        damages10 = (d[type_flood + '_1000yr'].prop_flood_prone * damage_function(d[type_flood + '_1000yr'].flood_depth)) + (d[type_flood + '_1000yr'].prop_flood_prone * damage_function(d[type_flood + '_1000yr'].flood_depth))
    
        return 0.5 * ((interval0 * damages0) + (interval1 * damages1) + (interval2 * damages2) + (interval3 * damages3) + (interval4 * damages4) + (interval5 * damages5) + (interval6 * damages6) + (interval7 * damages7) + (interval8 * damages8) + (interval9 * damages9) + (interval10 * damages10))

    fraction_capital_destroyed = pd.DataFrame()
    
    if options["pluvial"] == 0:
        fraction_capital_destroyed["contents_formal"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', content_damages, 'formal')
        fraction_capital_destroyed["contents_informal"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', content_damages, 'informal')
        fraction_capital_destroyed["contents_backyard"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', content_damages, 'backyard')
        fraction_capital_destroyed["contents_subsidized"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', content_damages, 'subsidized')
        fraction_capital_destroyed["structure_formal_1"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type4a, 'formal')
        fraction_capital_destroyed["structure_formal_2"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type4b, 'formal')
        fraction_capital_destroyed["structure_subsidized_1"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type4a, 'subsidized')
        fraction_capital_destroyed["structure_subsidized_2"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type4b, 'subsidized')
        fraction_capital_destroyed["structure_informal_settlements"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type2, 'informal')
        fraction_capital_destroyed["structure_informal_backyards"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type2, 'backyard')
        fraction_capital_destroyed["structure_formal_backyards"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type3a, 'backyard')
    elif options["pluvial"] == 1:
        fraction_capital_destroyed["contents_formal"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', content_damages, 'formal') + compute_fraction_capital_destroyed(d_pluvial, 'P', content_damages, 'formal')
        fraction_capital_destroyed["contents_informal"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', content_damages, 'informal') + compute_fraction_capital_destroyed(d_pluvial, 'P', content_damages, 'informal')
        fraction_capital_destroyed["contents_backyard"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', content_damages, 'backyard') + compute_fraction_capital_destroyed(d_pluvial, 'P', content_damages, 'backyard')
        fraction_capital_destroyed["contents_subsidized"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', content_damages, 'subsidized') + compute_fraction_capital_destroyed(d_pluvial, 'P', content_damages, 'subsidized')
        fraction_capital_destroyed["structure_formal_1"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type4a, 'formal') + compute_fraction_capital_destroyed(d_pluvial, 'P', structural_damages_type4a, 'formal')
        fraction_capital_destroyed["structure_formal_2"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type4b, 'formal') + compute_fraction_capital_destroyed(d_pluvial, 'P', structural_damages_type4b, 'formal')
        fraction_capital_destroyed["structure_subsidized_1"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type4a, 'subsidized') + compute_fraction_capital_destroyed(d_pluvial, 'P', structural_damages_type4a, 'subsidized')
        fraction_capital_destroyed["structure_subsidized_2"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type4b, 'subsidized') + compute_fraction_capital_destroyed(d_pluvial, 'P', structural_damages_type4b, 'subsidized')
        fraction_capital_destroyed["structure_informal_settlements"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type2, 'informal') + compute_fraction_capital_destroyed(d_pluvial, 'P', structural_damages_type2, 'informal')
        fraction_capital_destroyed["structure_informal_backyards"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type2, 'backyard') + compute_fraction_capital_destroyed(d_pluvial, 'P', structural_damages_type2, 'backyard')
        fraction_capital_destroyed["structure_formal_backyards"] = compute_fraction_capital_destroyed(d_fluvial, 'FD', structural_damages_type3a, 'backyard') + compute_fraction_capital_destroyed(d_pluvial, 'P', structural_damages_type3a, 'backyard')
        
    fraction_capital_destroyed["structure_backyards"] = ((16216 * fraction_capital_destroyed["structure_formal_backyards"]) + (74916 * fraction_capital_destroyed["structure_informal_backyards"])) / 91132
    
    return fraction_capital_destroyed, content_damages, structural_damages_type4b, structural_damages_type4a, structural_damages_type2, structural_damages_type3a

def import_macro_data(param, path_scenarios):
    
    #interest_rate  
    scenario_interest_rate = pd.read_csv(path_scenarios+ 'Scenario_interest_rate_1.csv', sep = ';')
    spline_interest_rate = interp1d(scenario_interest_rate.Year_interest_rate[~np.isnan(scenario_interest_rate.real_interest_rate)] - param["baseline_year"], scenario_interest_rate.real_interest_rate[~np.isnan(scenario_interest_rate.real_interest_rate)], 'linear') #Interest rate
    nb_years_interest_rate = 3
    interest_rate_n_years = spline_interest_rate(np.arange(0 - nb_years_interest_rate, 0))
    interest_rate_n_years[interest_rate_n_years < 0] = np.nan
    interest_rate = np.nanmean(interest_rate_n_years)/100

    #Pop
    #scenario_pop = pd.read_csv('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/data_Cape_Town/Scenarios/Scenario_pop_2.csv', sep = ';')        
    #population = scenario_pop.HH_total[scenario_pop.Year_pop == param["baseline_year"]].squeeze()
    population = 1055925
    return interest_rate, population

def import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, t):
    coeff_land_private = (spline_land_constraints(t) - spline_land_backyard(t) - spline_land_informal(t) - spline_land_RDP(t)) * param["max_land_use"]
    coeff_land_private[coeff_land_private < 0] = 0
    coeff_land_backyard = spline_land_backyard(t) * param["max_land_use_backyard"]
    coeff_land_RDP = spline_land_RDP(t)
    coeff_land_settlement = spline_land_informal(t) * param["max_land_use_settlement"]
    coeff_land = np.array([coeff_land_private, coeff_land_backyard, coeff_land_settlement, coeff_land_RDP])
    return coeff_land

def import_basile_simulation():
    mat1 = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations scenarios - 201908.mat')
    #mat1 = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations - 201907.mat')
    simul1 = mat1["simulation_noUE"]
    simul1_error = simul1["error"][0][0]
    simul1_utility = simul1["utility"][0][0]
    simul1_households_housing_type = simul1["householdsHousingType"][0][0]
    simul1_rent = simul1["rent"][0][0]
    simul1_dwelling_size = simul1["dwellingSize"][0][0]
    simul1_households_center = simul1["householdsCenter"][0][0]
    data = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/0. Precalculated inputs/data.mat')['data']
    SP_code = data["spCode"][0][0].squeeze()
    return simul1_error, simul1_utility, simul1_households_housing_type, simul1_rent, simul1_dwelling_size, simul1_households_center, SP_code

def SP_to_grid_2011_1(data_SP, grid): #we take SP_Code argument out 
    grid_intersect = pd.read_csv('C:/Users/monni/Documents/GitHub/cape_town_NEDUM/2. Data/data_Cape_Town/grid_SP_intersect.csv', sep = ';')   
    data_grid = np.zeros(len(grid.dist))   
    for index in range(0, len(grid.dist)):  
        intersect = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.id[index]])
        area_exclu = 0       
        for i in range(0, len(intersect)):     
            if len(data_SP[sp_code == intersect[i]]) == 0: #name of the column in data_sp, we'll se if this allows to take SP_code out of arguments                      
                area_exclu = area_exclu + sum(grid_intersect.Area[(grid_intersect.ID_grille == grid.id[index]) & (grid_intersect.SP_CODE == intersect[i])])
            else:
                data_grid[index] = data_grid[index] + sum(grid_intersect.Area[(grid_intersect.ID_grille == grid.id[index]) & (grid_intersect.SP_CODE == intersect[i])]) * data_SP[sp_code == intersect[i]]      #same as above  
        if area_exclu > 0.9 * sum(grid_intersect.Area[grid_intersect.ID_grille == grid.id[index]]):
            data_grid[index] = np.nan         
        else:
            if (sum(grid_intersect.Area[grid_intersect.ID_grille == grid.id[index]]) - area_exclu) > 0:
                data_grid[index] = data_grid[index] / (sum(grid_intersect.Area[grid_intersect.ID_grille == grid.id[index]]) - area_exclu)
            else:
               data_grid[index] = np.nan 
                
    return data_grid
