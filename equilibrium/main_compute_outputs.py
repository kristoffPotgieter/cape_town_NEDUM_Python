# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:01:05 2020

@author: Charlotte Liotta
"""
import numpy as np
from equilibrium.functions_solver import *

def compute_outputs(housing_type, utility, amenities, param, income_net_of_commuting_costs, fraction_capital_destroyed, grid, income_class_by_housing_type, options, housing_limit, agricultural_rent, interest_rate, coeff_land, minimum_housing_supply, construction_param, housing_in, param_pockets, param_backyards_pockets):
    
    # %% Dwelling size
    
    if housing_type == 'formal':
        
        dwelling_size = compute_dwelling_size_formal(utility, amenities, param, income_net_of_commuting_costs, fraction_capital_destroyed)
    
        #Here we introduce the minimum lot-size 
        dwelling_size = np.maximum(dwelling_size, param["mini_lot_size"])
        dwelling_size[income_class_by_housing_type.formal == 0, :] = np.nan
    
    elif housing_type == 'backyard':
        dwelling_size = param["shack_size"] * np.ones((4, len(grid.dist)))
        dwelling_size[income_class_by_housing_type.backyard == 0, :] = np.nan
    
    elif housing_type == 'informal':
        dwelling_size = param["shack_size"] * np.ones((4, len(grid.dist)))
        dwelling_size[income_class_by_housing_type.settlement == 0, :] = np.nan
        
        #plt.hist(data[data> 0.05], bins=100)
        #plt.xlim(0,20)
        #plt.tick_params(top='off', bottom='ON', left='off', right='off', labelleft='on', labelbottom='on')

        #data = housing_types_grid.informal_grid_2011 * param["shack_size"] / (250000 * 0.4)
        
    # %% Bid rents

    if housing_type == 'formal':
        R_mat = param["beta"] * (income_net_of_commuting_costs) / (dwelling_size - (param["alpha"] * param["q0"])) 
        R_mat[income_net_of_commuting_costs < 0] = 0
        R_mat[income_class_by_housing_type.formal == 0, :] = 0
    
    elif housing_type == 'backyard':
        R_mat = 1 / param["shack_size"] * (income_net_of_commuting_costs - ((1 + fraction_capital_destroyed.contents_backyard[None, :] * param["fraction_z_dwellings"]) * ((utility[:, None] / (amenities[None, :] * param_backyards_pockets[None, :] * ((dwelling_size - param["q0"]) ** param["beta"]))) ** (1/ param["alpha"]))) - (param["informal_structure_value"] * (interest_rate + param["depreciation_rate"])) - (fraction_capital_destroyed.structure_backyards[None, :] * param["informal_structure_value"]))
        R_mat[income_class_by_housing_type.backyard == 0, :] = 0
    
    elif housing_type == 'informal':
        #vec_correc_amenities = np.ones((income_net_of_commuting_costs.shape[0], income_net_of_commuting_costs.shape[1]))
        #vec_correc_amenities[:, (grid.dist < 26) & (grid.dist > 22)] = vec_correc_amenities[:, (grid.dist < 26) & (grid.dist > 22)] * 1.15
        #vec_correc_amenities[:, (grid.dist < 30) & (grid.dist > 26)] = vec_correc_amenities[:, (grid.dist < 30) & (grid.dist > 26)] * 1.12
        #vec_correc_amenities[:, (grid.dist < 17) & (grid.dist > 15)] = vec_correc_amenities[:, (grid.dist < 17) & (grid.dist > 15)] * 1.05
        #vec_correc_amenities[:, (grid.dist < 22) & (grid.dist > 17)] = vec_correc_amenities[:, (grid.dist < 22) & (grid.dist > 17)] * 1.05
        #R_mat = (1 / param["shack_size"]) * (income_net_of_commuting_costs - ((1 + fraction_capital_destroyed.contents[None, :] * param["fraction_z_dwellings"]) * ((utility[:, None] / (amenities[None, :] * param["amenity_settlement"] * vec_correc_amenities * ((dwelling_size - param["q0"]) ** param["beta"]))) ** (1/ param["alpha"]))) - (param["informal_structure_value"] * (interest_rate + param["depreciation_rate"])) - (fraction_capital_destroyed.structure[None, :] * param["informal_structure_value"]))
        R_mat = (1 / param["shack_size"]) * (income_net_of_commuting_costs - ((1 + fraction_capital_destroyed.contents_informal[None, :] * param["fraction_z_dwellings"]) * ((utility[:, None] / (amenities[None, :] * param_pockets[None, :] * ((dwelling_size - param["q0"]) ** param["beta"]))) ** (1/ param["alpha"]))) - (param["informal_structure_value"] * (interest_rate + param["depreciation_rate"])) - (fraction_capital_destroyed.structure_informal_settlements[None, :] * param["informal_structure_value"]))        
        #R_mat = (1 / param["shack_size"]) * (income_net_of_commuting_costs - ((1 + fraction_capital_destroyed.contents[None, :] * param["fraction_z_dwellings"]) * ((utility[:, None] / (amenities[None, :] * param["amenity_settlement"] * ((dwelling_size - param["q0"]) ** param["beta"]))) ** (1/ param["alpha"]))) - (param["informal_structure_value"] * (interest_rate + param["depreciation_rate"])) - (fraction_capital_destroyed.structure[None, :] * param["informal_structure_value"]))        
        
        R_mat[income_class_by_housing_type.settlement == 0, :] = 0

    R_mat[R_mat < 0] = 0
    R_mat[np.isnan(R_mat)] = 0
    
    #Income group in each location
    proba = (R_mat == np.nanmax(R_mat, 0))
    #proba[~np.isnan(param["multi_proba_group"])] = param["multi_proba_group"][~np.isnan(param["multi_proba_group"])]
    limit = (income_net_of_commuting_costs > 0) & (proba > 0) & (~np.isnan(income_net_of_commuting_costs)) & (R_mat > 0)
    proba = proba * limit

    which_group = np.nanargmax(R_mat, 0)
    #which_group[~np.isnan(multiProbaGroup(1,:))] = sum(repmat([1:param.numberIncomeGroup]', 1, sum(~isnan(multiProbaGroup(1,:)))).*proba(:,~isnan(multiProbaGroup(1,:))));
    #temp = np.arange(0, income_net_of_commuting_costs.shape[1])) * size(transTemp.incomeNetOfCommuting,1)
    #which_group_temp = which_group + temp; 
    
    R = np.empty(len(which_group))
    R[:] = np.nan
    dwelling_size_temp = np.empty(len(which_group))
    dwelling_size_temp[:] = np.nan
    for i in range(0, len(which_group)):
        R[i] = R_mat[int(which_group[i]), i]
        dwelling_size_temp[i] = dwelling_size[int(which_group[i]), i]
        
    dwelling_size = dwelling_size_temp
    
    
    
    # %% Housing supply
    
    if housing_type == 'formal':
        housing_supply = compute_housing_supply_formal(R, options, housing_limit, param, agricultural_rent, interest_rate, fraction_capital_destroyed, minimum_housing_supply, construction_param, housing_in, dwelling_size)
        housing_supply[R == 0] = 0
    elif housing_type == 'backyard':
        housing_supply = compute_housing_supply_backyard(R, param, income_net_of_commuting_costs, fraction_capital_destroyed, dwelling_size)
        housing_supply[R == 0] = 0
    elif housing_type == 'informal':
        housing_supply = 1000000 * np.ones(len(which_group))
        housing_supply[R == 0] = 0
    

    
    # %% Outputs
    
    people_init = housing_supply / dwelling_size * (np.nansum(limit,0) > 0)
    people_init[np.isnan(people_init)] = 0
    people_init_land = people_init * coeff_land * 0.25
    
    
    people_center = people_init_land[None, :] * proba
    people_center[np.isnan(people_center)] = 0
    job_simul = np.nansum(people_center, 1)
    
    if housing_type == 'formal':
        R = np.maximum(R, agricultural_rent)
      
    return job_simul, R, people_init, people_center, housing_supply, dwelling_size, R_mat
    