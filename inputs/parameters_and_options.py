# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:50:59 2020

@author: Charlotte Liotta
"""

import numpy as np
import scipy.io

def import_options():
    options = {"tax_out_urban_edge" : 0}
    options = {"urban_edge" : 0}
    options["adjust_housing_supply"] = 1
    options["import_precalculated_parameters"] = 1 #Do we do the calibration again?
    options["load_households_data"] = 0
    options["agents_anticipate_floods"] = 1
    options["WBUS2"] = 0
    return options

def import_param(options, precalculated_inputs):
    
    #Baseline Year
    param = {"baseline_year" : 2011}

    #Parameters
    if options["import_precalculated_parameters"] == 1:
        param["beta"] = scipy.io.loadmat(precalculated_inputs + 'calibratedUtility_beta.mat')["calibratedUtility_beta"].squeeze()
        param["q0"] = scipy.io.loadmat(precalculated_inputs + 'calibratedUtility_q0.mat')["calibratedUtility_q0"].squeeze()
        param["alpha"] = 1 - param["beta"]
        param["coeff_b"] = scipy.io.loadmat(precalculated_inputs + 'calibratedHousing_b.mat')["coeff_b"].squeeze()
        param["coeff_A"] = scipy.io.loadmat(precalculated_inputs + 'calibratedHousing_kappa.mat')["coeffKappa"].squeeze()
        param["coeff_a"] = 1 - param["coeff_b"]
        #param["amenity_backyard"] = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/calibratedParamAmenities.mat')["calibratedParamAmenities"][0].squeeze()
        #param["amenity_settlement"] = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/calibratedParamAmenities.mat')["calibratedParamAmenities"][1].squeeze()
        param["lambda"] = scipy.io.loadmat(precalculated_inputs + 'lambda.mat')["lambdaKeep"].squeeze()
    
    param["depreciation_rate"] = 0.025
    param["interest_rate"] = 0.025    
    param["shack_size"] = 14 #Size of a backyard shack (m2)
    param["RDP_size"] = 40 #Size of a RDP house (m2)
    param["backyard_size"] = 70 #size of the backyard of a RDP house (m2)
    param["future_rate_public_housing"] = 1000 #0 #1000
    param["informal_structure_value"] = 4000
    param["fraction_z_dwellings"] = 0.49
    param["subsidized_structure_value"] = 150000
        
    #Land Use
    param["max_land_use"] = 0.7
    param["max_land_use_backyard"] = 0.45
    param["max_land_use_settlement"] = 0.4

    #Density
    param["min_urban_density"] = 30000
    param["historic_radius"] = 100
    param["limit_height_center"] = 10 #very high => as if there were no limit
    param["limit_height_out"] = 10
    param["agricultural_rent_2011"] = 807.2
    param["agricultural_rent_2001"] = 70.7
    param["year_urban_edge"] = 2015 #in case option.urban_edge = 0, the year the constraint is removed

    #Incomes and income distribution
    param["nb_of_income_classes"] = 4
    param["income_distribution"] = np.array([0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4])
    param["threshold_jobs"] = 20000 #number of jobs above which we keep the employment center
    param["step"] = 2
    param["household_size"] = [1.14, 1.94, 1.94, 1.94] #Household size (accounting for unemployment rate)

    #Transportation
    param["waiting_time_metro"] = 10 #minutes
    param["walking_speed"] = 4 #km/h
    param["time_cost"] = 1
    
    #Solver
    param["max_iter"] = 5000
    param["precision"] = 0.01
    
    #Dynamic
    param["time_invest_housing"] = 3
    param["time_depreciation_buildings"] = 100
    param["iter_calc_lite"] = 1
    
    return param

def import_construction_parameters(param, grid, housing_types, dwelling_size_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land):
    
    param["housing_in"] = np.empty(len(grid_formal_density_HFA))
    param["housing_in"][:] = np.nan
    param["housing_in"][coeff_land[0,:] != 0] = grid_formal_density_HFA[coeff_land[0,:] != 0] / coeff_land[0,:][coeff_land[0,:] != 0] * 1.1
    param["housing_in"][(coeff_land[0,:] == 0) | np.isnan(grid_formal_density_HFA)] = 0
    
    param["housing_in"][param["housing_in"] > 2 * (10**6)] = 2 * (10**6)
    param["housing_in"][param["housing_in"] < 0] = 0
    
    #In Mitchells Plain, housing supply is given exogenously (planning), and household of group 2 live there (Coloured neighborhood). 
    param["minimum_housing_supply"] = np.zeros(len(grid.dist))
    param["minimum_housing_supply"][mitchells_plain_grid_2011] = mitchells_plain_grid_2011[mitchells_plain_grid_2011] / coeff_land[0, mitchells_plain_grid_2011]
    param["minimum_housing_supply"][(coeff_land[0,:] < 0.1) | (np.isnan(param["minimum_housing_supply"]))] = 0
    param["multi_proba_group"] = np.empty((param["nb_of_income_classes"], len(grid.dist)))
    param["multi_proba_group"][:] = np.nan
    
    #Define minimum lot-size 
    param["mini_lot_size"] = np.nanmin(dwelling_size_sp[housing_types.total_dwellings_SP_2011 != 0][(housing_types.informal_SP_2011[housing_types.total_dwellings_SP_2011 != 0] + housing_types.backyard_SP_2011[housing_types.total_dwellings_SP_2011 != 0]) / housing_types.total_dwellings_SP_2011[housing_types.total_dwellings_SP_2011 != 0] < 0.1])

    return param