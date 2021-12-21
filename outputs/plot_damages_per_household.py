# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:58:33 2020

@author: Charlotte Liotta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from flood_outputs import *
from plot_damages_per_household import *

#### pb 1: nb de pers pour chaque type d'inondations
#### pb 2: graphiques par classe

#0. Flood damages per household

floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
#floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr', 'P_100yr', 'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr']
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"
option = "percent" #"absolu"

spline_agricultural_rent, spline_interest_rate, spline_RDP, spline_population_income_distribution, spline_inflation, spline_income_distribution, spline_population, spline_interest_rate, spline_income, spline_minimum_housing_supply, spline_fuel = import_scenarios(income_2011, param, grid)
formal_structure_cost_2011 = compute_formal_structure_cost_method2(simulation_rent[0, :, :], param, interpolate_interest_rate(spline_interest_rate, 0), import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 0), simulation_households_housing_type[0, :, :], (spline_income(0) / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"])    
content_cost_2011 = compute_content_cost(simulation_households_center[0, :, :], np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/year_0.npy"), param, fraction_capital_destroyed, simulation_rent[0, :, :], simulation_dwelling_size[0, :, :], interpolate_interest_rate(spline_interest_rate, 0))
formal_structure_cost_2040 = compute_formal_structure_cost_method2(simulation_rent[28, :, :], param, interpolate_interest_rate(spline_interest_rate, 28), import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 28), simulation_households_housing_type[28, :, :], (spline_income(28) / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"])    
content_cost_2040 = compute_content_cost(simulation_households_center[28, :, :], np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/year_29.npy"), param, fraction_capital_destroyed, simulation_rent[28, :, :], simulation_dwelling_size[28, :, :], interpolate_interest_rate(spline_interest_rate, 28))


option = "percent" #"absolu"

for item in floods:

    param["subsidized_structure_value_ref"] = 150000
    param["informal_structure_value_ref"] = 4000
    df2011 = pd.DataFrame()
    df2040 = pd.DataFrame()
    type_flood = copy.deepcopy(item)
    data_flood = np.squeeze(pd.read_excel(path_data + item + ".xlsx"))
    
    formal_damages = structural_damages_type4a(data_flood['flood_depth'])
    formal_damages[simulation_dwelling_size[0, 0, :] > param["threshold"]] = structural_damages_type4b(data_flood.flood_depth[simulation_dwelling_size[0, 0, :] > param["threshold"]])
    subsidized_damages = structural_damages_type4a(data_flood['flood_depth'])
    subsidized_damages[simulation_dwelling_size[0, 3, :] > param["threshold"]] = structural_damages_type4b(data_flood.flood_depth[simulation_dwelling_size[0, 3, :] > param["threshold"]])
        
    df2011['formal_structure_damages'] = formal_structure_cost_2011 * formal_damages
    df2011['subsidized_structure_damages'] = param["subsidized_structure_value_ref"] * subsidized_damages
    df2011['informal_structure_damages'] = param["informal_structure_value_ref"] * structural_damages_type2(data_flood['flood_depth'])
    df2011['backyard_structure_damages'] = ((16216 * (param["informal_structure_value_ref"] * structural_damages_type2(data_flood['flood_depth']))) + (74916 * (param["informal_structure_value_ref"] * structural_damages_type3a(data_flood['flood_depth'])))) / (74916 + 16216)
            
    df2011['formal_content_damages'] =  content_cost_2011.formal * content_damages(data_flood['flood_depth'])
    df2011['subsidized_content_damages'] = content_cost_2011.subsidized * content_damages(data_flood['flood_depth'])
    df2011['informal_content_damages'] = content_cost_2011.informal * content_damages(data_flood['flood_depth'])
    df2011['backyard_content_damages'] = content_cost_2011.backyard * content_damages(data_flood['flood_depth'])
    
    formal_damages = structural_damages_type4a(data_flood['flood_depth'])
    formal_damages[simulation_dwelling_size[28, 0, :] > param["threshold"]] = structural_damages_type4b(data_flood.flood_depth[simulation_dwelling_size[28, 0, :] > param["threshold"]])
    subsidized_damages = structural_damages_type4a(data_flood['flood_depth'])
    subsidized_damages[simulation_dwelling_size[28, 3, :] > param["threshold"]] = structural_damages_type4b(data_flood.flood_depth[simulation_dwelling_size[28, 3, :] > param["threshold"]])
   
    df2040['formal_structure_damages'] = formal_structure_cost_2040 * formal_damages
    df2040['subsidized_structure_damages'] = param["subsidized_structure_value_ref"] * (spline_inflation(28) / spline_inflation(0)) * subsidized_damages
    df2040['informal_structure_damages'] = param["informal_structure_value_ref"] * (spline_inflation(28) / spline_inflation(0)) * structural_damages_type2(data_flood['flood_depth'])
    df2040['backyard_structure_damages'] = ((16216 * (param["informal_structure_value_ref"] * (spline_inflation(28) / spline_inflation(0)) * structural_damages_type2(data_flood['flood_depth']))) + (74916 * (param["informal_structure_value_ref"] * (spline_inflation(28) / spline_inflation(0)) * structural_damages_type3a(data_flood['flood_depth'])))) / (74916 + 16216)
    
    df2040['formal_content_damages'] =  content_cost_2040.formal * content_damages(data_flood['flood_depth'])
    df2040['subsidized_content_damages'] = content_cost_2040.subsidized * content_damages(data_flood['flood_depth'])
    df2040['informal_content_damages'] = content_cost_2040.informal * content_damages(data_flood['flood_depth'])
    df2040['backyard_content_damages'] = content_cost_2040.backyard * content_damages(data_flood['flood_depth'])

    df2011["formal_pop_flood_prone"] = simulation_households_housing_type[0, 0, :] * data_flood["prop_flood_prone"]
    df2011["backyard_pop_flood_prone"] = simulation_households_housing_type[0, 1, :] * data_flood["prop_flood_prone"]
    df2011["informal_pop_flood_prone"] = simulation_households_housing_type[0, 2, :] * data_flood["prop_flood_prone"]
    df2011["subsidized_pop_flood_prone"] = simulation_households_housing_type[0, 3, :] * data_flood["prop_flood_prone"]
    
    df2040["formal_pop_flood_prone"] = simulation_households_housing_type[28, 0, :] * data_flood["prop_flood_prone"]
    df2040["backyard_pop_flood_prone"] = simulation_households_housing_type[28, 1, :] * data_flood["prop_flood_prone"]
    df2040["informal_pop_flood_prone"] = simulation_households_housing_type[28, 2, :] * data_flood["prop_flood_prone"]
    df2040["subsidized_pop_flood_prone"] = simulation_households_housing_type[28, 3, :] * data_flood["prop_flood_prone"]
    
    df2011["formal_damages"] = df2011['formal_structure_damages'] + df2011['formal_content_damages']
    df2011["informal_damages"] = df2011['informal_structure_damages'] + df2011['informal_content_damages']
    df2011["subsidized_damages"] = df2011['subsidized_structure_damages'] + df2011['subsidized_content_damages']
    df2011["backyard_damages"] = df2011['backyard_structure_damages'] + df2011['backyard_content_damages']
    
    df2040["formal_damages"] = df2040['formal_structure_damages'] + df2040['formal_content_damages']    
    df2040["informal_damages"] = df2040['informal_structure_damages'] + df2040['informal_content_damages']
    df2040["subsidized_damages"] = df2040['subsidized_structure_damages'] + df2040['subsidized_content_damages']
    df2040["backyard_damages"] = df2040['backyard_structure_damages'] + df2040['backyard_content_damages']

    if item == "P_20yr":
        df2011["formal_damages"] = 0
        df2040["formal_damages"] = 0
        df2011["formal_pop_flood_prone"] = 0
        df2040["formal_pop_flood_prone"] = 0
    elif ((item == "P_5yr") |(item == "P_10yr")):
        df2011["formal_damages"] = 0
        df2040["formal_damages"] = 0
        df2011["subsidized_damages"] = 0
        df2040["subsidized_damages"] = 0
        df2011["backyard_damages"] = 0
        df2040["backyard_damages"] = 0
        df2011["formal_pop_flood_prone"] = 0
        df2040["formal_pop_flood_prone"] = 0
        df2011["backyard_pop_flood_prone"] = 0
        df2040["backyard_pop_flood_prone"] = 0
        df2011["subsidized_pop_flood_prone"] = 0
        df2040["subsidized_pop_flood_prone"] = 0
    writer = pd.ExcelWriter('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + str(item) + '_2011.xlsx')
    df2011.to_excel(excel_writer = writer)
    writer.save()
    writer = pd.ExcelWriter('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + str(item) + '_2040.xlsx')
    
    df2040.to_excel(excel_writer = writer)
    writer.save()
   
damages_5yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_5yr' + '_2011.xlsx')  
damages_10yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_10yr' + '_2011.xlsx')  
damages_20yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_20yr' + '_2011.xlsx')  
damages_50yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_50yr' + '_2011.xlsx')  
damages_75yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_75yr' + '_2011.xlsx')  
damages_100yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_100yr' + '_2011.xlsx')  
damages_200yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_200yr' + '_2011.xlsx')  
damages_250yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_250yr' + '_2011.xlsx')  
damages_500yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_500yr' + '_2011.xlsx')  
damages_1000yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_1000yr' + '_2011.xlsx')  

damages_5yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_5yr' + '_2040.xlsx')  
damages_10yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_10yr' + '_2040.xlsx')
damages_20yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_20yr' + '_2040.xlsx')  
damages_50yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_50yr' + '_2040.xlsx')  
damages_75yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_75yr' + '_2040.xlsx')  
damages_100yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_100yr' + '_2040.xlsx')  
damages_200yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_200yr' + '_2040.xlsx')  
damages_250yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_250yr' + '_2040.xlsx')  
damages_500yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_500yr' + '_2040.xlsx')  
damages_1000yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_1000yr' + '_2040.xlsx')  

#damages_5yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_5yr' + '_2011.xlsx')  
#damages_10yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_10yr' + '_2011.xlsx')  
#damages_20yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_20yr' + '_2011.xlsx')  
#damages_50yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_50yr' + '_2011.xlsx')  
#damages_75yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_75yr' + '_2011.xlsx')
#damages_100yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_100yr' + '_2011.xlsx')  
#damages_200yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_200yr' + '_2011.xlsx')  
#damages_250yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_250yr' + '_2011.xlsx')  
#damages_500yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_500yr' + '_2011.xlsx')  
#damages_1000yr_2011 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_1000yr' + '_2011.xlsx')  

#damages_5yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_5yr' + '_2040.xlsx')  
#damages_10yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_10yr' + '_2040.xlsx')  
#damages_20yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_20yr' + '_2040.xlsx')  
#damages_50yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_50yr' + '_2040.xlsx')  
#damages_75yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_75yr' + '_2040.xlsx')  
#damages_100yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_100yr' + '_2040.xlsx')  
#damages_200yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_200yr' + '_2040.xlsx')  
#damages_250yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_250yr' + '_2040.xlsx')  
#damages_500yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_500yr' + '_2040.xlsx')  
#damages_1000yr_2040 = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'P_1000yr' + '_2040.xlsx') 

damages_10yr_2011.iloc[:, 9:13] = damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13]
damages_20yr_2011.iloc[:, 9:13] = damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13]
damages_50yr_2011.iloc[:, 9:13] = damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13]
damages_75yr_2011.iloc[:, 9:13] = damages_75yr_2011.iloc[:, 9:13] - damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13]
damages_100yr_2011.iloc[:, 9:13] = damages_100yr_2011.iloc[:, 9:13] - damages_75yr_2011.iloc[:, 9:13] - damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13]
damages_200yr_2011.iloc[:, 9:13] = damages_200yr_2011.iloc[:, 9:13] - damages_100yr_2011.iloc[:, 9:13] - damages_75yr_2011.iloc[:, 9:13] - damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13]
damages_250yr_2011.iloc[:, 9:13] = damages_250yr_2011.iloc[:, 9:13] - damages_200yr_2011.iloc[:, 9:13] - damages_100yr_2011.iloc[:, 9:13] - damages_75yr_2011.iloc[:, 9:13] - damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13] - damages_5yr_2011.iloc[:, 9:13]
damages_500yr_2011.iloc[:, 9:13] = damages_500yr_2011.iloc[:, 9:13] - damages_250yr_2011.iloc[:, 9:13] - damages_200yr_2011.iloc[:, 9:13] - damages_100yr_2011.iloc[:, 9:13] - damages_75yr_2011.iloc[:, 9:13] - damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13] - damages_10yr_2011.iloc[:, 9:13]- damages_5yr_2011.iloc[:, 9:13]
damages_1000yr_2011.iloc[:, 9:13] = damages_1000yr_2011.iloc[:, 9:13] - damages_500yr_2011.iloc[:, 9:13] - damages_250yr_2011.iloc[:, 9:13] - damages_200yr_2011.iloc[:, 9:13] - damages_100yr_2011.iloc[:, 9:13] - damages_75yr_2011.iloc[:, 9:13] - damages_50yr_2011.iloc[:, 9:13] - damages_20yr_2011.iloc[:, 9:13]- damages_10yr_2011.iloc[:, 9:13]- damages_5yr_2011.iloc[:, 9:13]

damages_10yr_2040.iloc[:, 9:13] = damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13]
damages_20yr_2040.iloc[:, 9:13] = damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13]
damages_50yr_2040.iloc[:, 9:13] = damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13]
damages_75yr_2040.iloc[:, 9:13] = damages_75yr_2040.iloc[:, 9:13] - damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13]
damages_100yr_2040.iloc[:, 9:13] = damages_100yr_2040.iloc[:, 9:13] - damages_75yr_2040.iloc[:, 9:13] - damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13]
damages_200yr_2040.iloc[:, 9:13] = damages_200yr_2040.iloc[:, 9:13] - damages_100yr_2040.iloc[:, 9:13] - damages_75yr_2040.iloc[:, 9:13] - damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13]
damages_250yr_2040.iloc[:, 9:13] = damages_250yr_2040.iloc[:, 9:13] - damages_200yr_2040.iloc[:, 9:13] - damages_100yr_2040.iloc[:, 9:13] - damages_75yr_2040.iloc[:, 9:13] - damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13] - damages_5yr_2040.iloc[:, 9:13]
damages_500yr_2040.iloc[:, 9:13] = damages_500yr_2040.iloc[:, 9:13] - damages_250yr_2040.iloc[:, 9:13] - damages_200yr_2040.iloc[:, 9:13] - damages_100yr_2040.iloc[:, 9:13] - damages_75yr_2040.iloc[:, 9:13] - damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13] - damages_10yr_2040.iloc[:, 9:13]- damages_5yr_2040.iloc[:, 9:13]
damages_1000yr_2040.iloc[:, 9:13] = damages_1000yr_2040.iloc[:, 9:13] - damages_500yr_2040.iloc[:, 9:13] - damages_250yr_2040.iloc[:, 9:13] - damages_200yr_2040.iloc[:, 9:13] - damages_100yr_2040.iloc[:, 9:13] - damages_75yr_2040.iloc[:, 9:13] - damages_50yr_2040.iloc[:, 9:13] - damages_20yr_2040.iloc[:, 9:13]- damages_10yr_2040.iloc[:, 9:13]- damages_5yr_2040.iloc[:, 9:13]

damages_5yr_2011.iloc[:, 13:17] = annualize_damages([damages_5yr_2011.iloc[:, 13:17], damages_10yr_2011.iloc[:, 13:17], damages_20yr_2011.iloc[:, 13:17], damages_50yr_2011.iloc[:, 13:17], damages_75yr_2011.iloc[:, 13:17], damages_100yr_2011.iloc[:, 13:17], damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_10yr_2011.iloc[:, 13:17] = annualize_damages([0, damages_10yr_2011.iloc[:, 13:17], damages_20yr_2011.iloc[:, 13:17], damages_50yr_2011.iloc[:, 13:17], damages_75yr_2011.iloc[:, 13:17], damages_100yr_2011.iloc[:, 13:17], damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_20yr_2011.iloc[:, 13:17] = annualize_damages([0, 0, damages_20yr_2011.iloc[:, 13:17], damages_50yr_2011.iloc[:, 13:17], damages_75yr_2011.iloc[:, 13:17], damages_100yr_2011.iloc[:, 13:17], damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_50yr_2011.iloc[:, 13:17] = annualize_damages([0, 0, 0, damages_50yr_2011.iloc[:, 13:17], damages_75yr_2011.iloc[:, 13:17], damages_100yr_2011.iloc[:, 13:17], damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_75yr_2011.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, damages_75yr_2011.iloc[:, 13:17], damages_100yr_2011.iloc[:, 13:17], damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_100yr_2011.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, 0, damages_100yr_2011.iloc[:, 13:17], damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_200yr_2011.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, 0, 0, damages_200yr_2011.iloc[:, 13:17], damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_250yr_2011.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, 0, 0, 0, damages_250yr_2011.iloc[:, 13:17], damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_500yr_2011.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, 0, 0, 0, 0, damages_500yr_2011.iloc[:, 13:17], damages_1000yr_2011.iloc[:, 13:17]])
damages_1000yr_2011.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, 0, 0, 0, 0, 0, damages_1000yr_2011.iloc[:, 13:17]])

damages_5yr_2040.iloc[:, 13:17] = annualize_damages([damages_5yr_2040.iloc[:, 13:17], damages_10yr_2040.iloc[:, 13:17], damages_20yr_2040.iloc[:, 13:17], damages_50yr_2040.iloc[:, 13:17], damages_75yr_2040.iloc[:, 13:17], damages_100yr_2040.iloc[:, 13:17], damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_10yr_2040.iloc[:, 13:17] = annualize_damages([0, damages_10yr_2040.iloc[:, 13:17], damages_20yr_2040.iloc[:, 13:17], damages_50yr_2040.iloc[:, 13:17], damages_75yr_2040.iloc[:, 13:17], damages_100yr_2040.iloc[:, 13:17], damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_20yr_2040.iloc[:, 13:17] = annualize_damages([0, 0, damages_20yr_2040.iloc[:, 13:17], damages_50yr_2040.iloc[:, 13:17], damages_75yr_2040.iloc[:, 13:17], damages_100yr_2040.iloc[:, 13:17], damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_50yr_2040.iloc[:, 13:17] = annualize_damages([0, 0, 0, damages_50yr_2040.iloc[:, 13:17], damages_75yr_2040.iloc[:, 13:17], damages_100yr_2040.iloc[:, 13:17], damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_75yr_2040.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, damages_75yr_2040.iloc[:, 13:17], damages_100yr_2040.iloc[:, 13:17], damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_100yr_2040.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, 0, damages_100yr_2040.iloc[:, 13:17], damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_200yr_2040.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, 0, 0, damages_200yr_2040.iloc[:, 13:17], damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_250yr_2040.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, 0, 0, 0, damages_250yr_2040.iloc[:, 13:17], damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_500yr_2040.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, 0, 0, 0, 0, damages_500yr_2040.iloc[:, 13:17], damages_1000yr_2040.iloc[:, 13:17]])
damages_1000yr_2040.iloc[:, 13:17] = annualize_damages([0, 0, 0, 0, 0, 0, 0, 0, 0, damages_1000yr_2040.iloc[:, 13:17]])

damages_5yr_2011 = damages_5yr_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_10yr_2011 = damages_10yr_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_20yr_2011 = damages_20yr_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_50yr_2011 = damages_50yr_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_75yr_2011 = damages_75yr_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_100yr_2011 = damages_100yr_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_200yr_2011 = damages_200yr_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_250yr_2011 = damages_250yr_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_500yr_2011 = damages_500yr_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_1000yr_2011 = damages_1000yr_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']

damages_5yr_2040 = damages_5yr_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_10yr_2040 = damages_10yr_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_20yr_2040 = damages_20yr_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_50yr_2040 = damages_50yr_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_75yr_2040 = damages_75yr_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_100yr_2040 = damages_100yr_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_200yr_2040 = damages_200yr_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_250yr_2040 = damages_250yr_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_500yr_2040 = damages_500yr_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_1000yr_2040 = damages_1000yr_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']

income_class_2011 = np.argmax(simulation_households[0, :, :, :], 1)    
average_income_2011 = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/average_income_year_0.npy")

income_class_2040 = np.argmax(simulation_households[28, :, :, :], 1)     
average_income_2040 = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/average_income_year_28.npy")
    

real_income_2011 = np.empty((24014, 4))
for i in range(0, 24014):
    for j in range(0, 4):
        print(i)
        real_income_2011[i, j] = average_income_2011[np.array(income_class_2011)[j, i], i]
    
real_income_2040 = np.empty((24014, 4))
for i in range(0, 24014):
    for j in range(0, 4):
        print(i)
        real_income_2040[i, j] = average_income_2040[np.array(income_class_2040)[j, i], i]
    
        

real_income_2011 = np.matlib.repmat(real_income_2011, 10, 1).squeeze()
real_income_2040 = np.matlib.repmat(real_income_2040, 10, 1).squeeze()

total_2011 = np.vstack([damages_5yr_2011, damages_10yr_2011, damages_20yr_2011, damages_50yr_2011, damages_75yr_2011, damages_100yr_2011, damages_200yr_2011, damages_250yr_2011, damages_500yr_2011, damages_1000yr_2011])
total_2040 = np.vstack([damages_5yr_2040, damages_10yr_2040, damages_20yr_2040, damages_50yr_2040, damages_75yr_2040, damages_100yr_2040, damages_200yr_2040, damages_250yr_2040, damages_500yr_2040, damages_1000yr_2040])   

total_2011[:, 4] = (total_2011[:, 4] / real_income_2011[:,0]) * 100
total_2011[:, 5] = (total_2011[:, 5] / real_income_2011[:,2]) * 100
total_2011[:, 6] = (total_2011[:, 6] / real_income_2011[:,3]) * 100
total_2011[:, 7] = (total_2011[:, 7] / real_income_2011[:,1]) * 100

total_2040[:, 4] = (total_2040[:, 4] / real_income_2040[:,0]) * 100
total_2040[:, 5] = (total_2040[:, 5] / real_income_2040[:,2]) * 100
total_2040[:, 6] = (total_2040[:, 6] / real_income_2040[:,3]) * 100
total_2040[:, 7] = (total_2040[:, 7] / real_income_2040[:,1]) * 100

## Reshape
formal_2011 = total_2011[:, [0, 4]]
backyard_2011 = total_2011[:, [1, 7]]
informal_2011 = total_2011[:, [2, 5]]
subsidized_2011 = total_2011[:, [3, 6]]

formal_2040 = total_2040[:, [0, 4]]
backyard_2040 = total_2040[:, [1, 7]]
informal_2040 = total_2040[:, [2, 5]]
subsidized_2040 = total_2040[:, [3, 6]]

### C'est le moment de subsuet by income class

#Formal 2011:shape (240140, 2)
#income_class_2011:shape((4, 24014))

income_class_2011_reshape = np.matlib.repmat(income_class_2011, 1, 10).squeeze()
income_class_2040_reshape = np.matlib.repmat(income_class_2040, 1, 10).squeeze()

formal_2011_class1 = formal_2011[income_class_2011_reshape[0, :] == 0]
formal_2011_class2 = formal_2011[income_class_2011_reshape[0, :] == 1]
formal_2011_class3 = formal_2011[income_class_2011_reshape[0, :] == 2]
formal_2011_class4 = formal_2011[income_class_2011_reshape[0, :] == 3]

formal_2040_class1 = formal_2040[income_class_2040_reshape[0, :] == 0]
formal_2040_class2 = formal_2040[income_class_2040_reshape[0, :] == 1]
formal_2040_class3 = formal_2040[income_class_2040_reshape[0, :] == 2]
formal_2040_class4 = formal_2040[income_class_2040_reshape[0, :] == 3]

subsidized_2011_class1 = subsidized_2011[income_class_2011_reshape[3, :] == 0]
subsidized_2011_class2 = subsidized_2011[income_class_2011_reshape[3, :] == 1]
subsidized_2011_class3 = subsidized_2011[income_class_2011_reshape[3, :] == 2]
subsidized_2011_class4 = subsidized_2011[income_class_2011_reshape[3, :] == 3]

subsidized_2040_class1 = subsidized_2040[income_class_2040_reshape[3, :] == 0]
subsidized_2040_class2 = subsidized_2040[income_class_2040_reshape[3, :] == 1]
subsidized_2040_class3 = subsidized_2040[income_class_2040_reshape[3, :] == 2]
subsidized_2040_class4 = subsidized_2040[income_class_2040_reshape[3, :] == 3]

backyard_2011_class1 = backyard_2011[income_class_2011_reshape[1, :] == 0]
backyard_2011_class2 = backyard_2011[income_class_2011_reshape[1, :] == 1]
backyard_2011_class3 = backyard_2011[income_class_2011_reshape[1, :] == 2]
backyard_2011_class4 = backyard_2011[income_class_2011_reshape[1, :] == 3]

backyard_2040_class1 = backyard_2040[income_class_2040_reshape[1, :] == 0]
backyard_2040_class2 = backyard_2040[income_class_2040_reshape[1, :] == 1]
backyard_2040_class3 = backyard_2040[income_class_2040_reshape[1, :] == 2]
backyard_2040_class4 = backyard_2040[income_class_2040_reshape[1, :] == 3]

informal_2011_class1 = informal_2011[income_class_2011_reshape[2, :] == 0]
informal_2011_class2 = informal_2011[income_class_2011_reshape[2, :] == 1]
informal_2011_class3 = informal_2011[income_class_2011_reshape[2, :] == 2]
informal_2011_class4 = informal_2011[income_class_2011_reshape[2, :] == 3]

informal_2040_class1 = informal_2040[income_class_2040_reshape[2, :] == 0]
informal_2040_class2 = informal_2040[income_class_2040_reshape[2, :] == 1]
informal_2040_class3 = informal_2040[income_class_2040_reshape[2, :] == 2]
informal_2040_class4 = informal_2040[income_class_2040_reshape[2, :] == 3]

#Total
array_2011 = np.vstack([formal_2011, backyard_2011, informal_2011, subsidized_2011])
subset_2011 = array_2011[~np.isnan(array_2011[:,1])]
array_2040 = np.vstack([formal_2040, backyard_2040, informal_2040, subsidized_2040])
subset_2040 = array_2040[~np.isnan(array_2040[:,1])]
sns.distplot(subset_2011[:,1],
             bins=np.arange(0,0.7,0.01), hist = True, kde = False, hist_kws={'weights': subset_2011[:,0]}, color = 'black', label = "2011")
sns.distplot(subset_2040[:,1],
             bins=np.arange(0,0.7,0.01), hist = True, kde = False, hist_kws={'weights': subset_2040[:,0]}, label = "2040")
plt.legend()
#plt.ylim(0, 320000)
#plt.ylim(0, 50000)
plt.xlabel("Share of the annual income destroyed by floods - annualized (%)")
plt.ylabel("Number of households")

#Class 1
array_2011 = np.vstack([formal_2011_class1, backyard_2011_class1, informal_2011_class1, subsidized_2011_class1])
subset_2011 = array_2011[~np.isnan(array_2011[:,1])]
array_2040 = np.vstack([formal_2040_class1, backyard_2040_class1, informal_2040_class1, subsidized_2040_class1])
subset_2040 = array_2040[~np.isnan(array_2040[:,1])]
sns.distplot(subset_2011[:,1],
             bins=np.arange(0,0.7,0.01), hist = True, kde = False, hist_kws={'weights': subset_2011[:,0]}, color = 'black', label = "2011")
sns.distplot(subset_2040[:,1],
             bins=np.arange(0,0.7,0.01), hist = True, kde = False, hist_kws={'weights': subset_2040[:,0]}, label = "2040")
plt.legend()
plt.xlabel("Share of the annual income destroyed by floods - annualized (%)")
plt.ylabel("Number of households")

#Class 2
array_2011 = np.vstack([formal_2011_class2, backyard_2011_class2, informal_2011_class2, subsidized_2011_class2])
subset_2011 = array_2011[~np.isnan(array_2011[:,1])]
array_2040 = np.vstack([formal_2040_class2, backyard_2040_class2, informal_2040_class2, subsidized_2040_class2])
subset_2040 = array_2040[~np.isnan(array_2040[:,1])]
sns.distplot(subset_2011[:,1],
             bins=np.arange(0,0.7,0.01), hist = True, kde = False, hist_kws={'weights': subset_2011[:,0]}, color = 'black', label = "2011")
sns.distplot(subset_2040[:,1],
             bins=np.arange(0,0.7,0.01), hist = True, kde = False, hist_kws={'weights': subset_2040[:,0]}, label = "2040")
plt.legend()
plt.xlabel("Share of the annual income destroyed by floods - annualized (%)")
plt.ylabel("Number of households")

#Class 3
array_2011 = np.vstack([formal_2011_class3, backyard_2011_class3, informal_2011_class3, subsidized_2011_class3])
subset_2011 = array_2011[~np.isnan(array_2011[:,1])]
array_2040 = np.vstack([formal_2040_class3, backyard_2040_class3, informal_2040_class3, subsidized_2040_class3])
subset_2040 = array_2040[~np.isnan(array_2040[:,1])]
sns.distplot(subset_2011[:,1],
             bins=np.arange(0,0.7,0.01), hist = True, kde = False, hist_kws={'weights': subset_2011[:,0]}, color = 'black', label = "2011")
sns.distplot(subset_2040[:,1],
             bins=np.arange(0,0.7,0.01), hist = True, kde = False, hist_kws={'weights': subset_2040[:,0]}, label = "2040")
plt.legend()
plt.xlabel("Share of the annual income destroyed by floods - annualized (%)")
plt.ylabel("Number of households")

#Class 4
array_2011 = np.vstack([formal_2011_class4, backyard_2011_class4, informal_2011_class4, subsidized_2011_class4])
subset_2011 = array_2011[~np.isnan(array_2011[:,1])]
array_2040 = np.vstack([formal_2040_class4, backyard_2040_class4, informal_2040_class4, subsidized_2040_class4])
subset_2040 = array_2040[~np.isnan(array_2040[:,1])]
sns.distplot(subset_2011[:,1],
             bins=np.arange(0,0.7,0.01), hist = True, kde = False, hist_kws={'weights': subset_2011[:,0]}, color = 'black', label = "2011")
sns.distplot(subset_2040[:,1],
             bins=np.arange(0,0.7,0.01), hist = True, kde = False, hist_kws={'weights': subset_2040[:,0]}, label = "2040")
plt.legend()
plt.xlabel("Share of the annual income destroyed by floods - annualized (%)")
plt.ylabel("Number of households")





   