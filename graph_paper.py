# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:09:18 2021

@author: charl
"""

import numpy as np
import scipy.io
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import matplotlib.patches as mpatches

from inputs.data import *
from inputs.parameters_and_options import *
from equilibrium.compute_equilibrium import *
from outputs.export_outputs import *
from outputs.export_outputs_floods import *
from outputs.flood_outputs import *
from equilibrium.functions_dynamic import *
from equilibrium.run_simulations import *
from inputs.WBUS2_depth import *

path_simulation_sc1 = 'C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/20210804_1_1'
path_simulation_sc2 = 'C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/20210804_1_0'

no_policy_sc1 = 'C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/no_floods_scenario_1_1'
no_policy_sc2 = 'C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/no_floods_scenario_1_0'

households_center_sc1 = np.load(path_simulation_sc1 + '/simulation_households_center.npy')
simulation_dwelling_size_sc1 = np.load(path_simulation_sc1 + '/simulation_dwelling_size.npy')
simulation_rent_sc1 = np.load(path_simulation_sc1 + '/simulation_rent.npy')
simulation_households_housing_type_sc1 = np.load(path_simulation_sc1 + '/simulation_households_housing_type.npy')
simulation_households_sc1 = np.load(path_simulation_sc1 + '/simulation_households.npy')
simulation_utility_sc1 = np.load(path_simulation_sc1 + '/simulation_utility.npy')

households_center_sc2 = np.load(path_simulation_sc2 + '/simulation_households_center.npy')
simulation_dwelling_size_sc2 = np.load(path_simulation_sc2 + '/simulation_dwelling_size.npy')
simulation_rent_sc2 = np.load(path_simulation_sc2 + '/simulation_rent.npy')
simulation_households_housing_type_sc2 = np.load(path_simulation_sc2 + '/simulation_households_housing_type.npy')
simulation_households_sc2 = np.load(path_simulation_sc2 + '/simulation_households.npy')
simulation_utility_sc2 = np.load(path_simulation_sc2 + '/simulation_utility.npy')

fluvial_floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
pluvial_floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr', 'P_100yr', 'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr']
path_flood_data = "C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/FATHOM/"


### FIGURE 1: HOUSEHOLDS PER HOUSING TYPES
plt.figure(figsize=(10,7))
plt.rcParams.update({'font.size': 10})
order = ['Formal private', 'Formal subsidized', 'Informal in \n backyards', 'Informal \n settlements']
data_sc1 = pd.DataFrame({'2011': np.nansum(simulation_households_housing_type_sc1[0, :, :], 1), '2020': np.nansum(simulation_households_housing_type_sc1[9, :, :], 1),'2030': np.nansum(simulation_households_housing_type_sc1[19, :, :], 1),'2040': np.nansum(simulation_households_housing_type_sc1[28, :, :], 1),}, index = ["Formal private", "Informal in \n backyards", "Informal \n settlements", "Formal subsidized"])
data_sc2 = pd.DataFrame({'2011': np.nansum(simulation_households_housing_type_sc2[0, :, :], 1), '2020': np.nansum(simulation_households_housing_type_sc2[9, :, :], 1),'2030': np.nansum(simulation_households_housing_type_sc2[19, :, :], 1),'2040': np.nansum(simulation_households_housing_type_sc2[28, :, :], 1),}, index = ["Formal private", "Informal in \n backyards", "Informal \n settlements", "Formal subsidized"])
data_mean = pd.DataFrame({'2011': (np.nansum(simulation_households_housing_type_sc1[0, :, :], 1) + np.nansum(simulation_households_housing_type_sc2[0, :, :], 1)) / 2, '2020': (np.nansum(simulation_households_housing_type_sc1[9, :, :], 1) + np.nansum(simulation_households_housing_type_sc2[9, :, :], 1)) / 2,'2030': (np.nansum(simulation_households_housing_type_sc1[19, :, :], 1) + np.nansum(simulation_households_housing_type_sc2[19, :, :], 1))/2,'2040': (np.nansum(simulation_households_housing_type_sc1[28, :, :], 1) + np.nansum(simulation_households_housing_type_sc2[28, :, :], 1)) / 2,}, index = ["Formal private", "Informal in \n backyards", "Informal \n settlements", "Formal subsidized"])
data_mean.loc[order].plot(kind="bar", cmap = plt.get_cmap('summer', 4))
rects1 = plt.bar(np.arange(4) - 0.2, data_mean.loc[order]['2011'], color = plt.get_cmap('summer')(0.), width = 0.13)
rects2 = plt.bar(np.arange(4) - 0.067, data_mean.loc[order]['2020'], color = plt.get_cmap('summer')(0.33), width = 0.13)
rects3 = plt.bar(np.arange(4) +0.067, data_mean.loc[order]['2030'], color = plt.get_cmap('summer')(0.66), width = 0.13)
rects4 = plt.bar(np.arange(4) + 0.2, data_mean.loc[order]['2040'], color = plt.get_cmap('summer')(1.), width = 0.13)
p1, = plt.plot(np.arange(len(data_sc1)) -0.2 , data_sc1.loc[order]["2011"], marker="D", linestyle="", alpha=0.8, color="r", label = "SC1")
p2, = plt.plot(np.arange(len(data_sc1)) -0.2, data_sc2.loc[order]["2011"], marker="D", linestyle="", alpha=0.8, color="b", label = "SC2")
plt.plot(np.arange(len(data_sc1)) -0.067 , data_sc1.loc[order]["2020"], marker="D", linestyle="", alpha=0.8, color="r")
plt.plot(np.arange(len(data_sc1)) -0.067, data_sc2.loc[order]["2020"], marker="D", linestyle="", alpha=0.8, color="b")
plt.plot(np.arange(len(data_sc1)) +0.067 , data_sc1.loc[order]["2030"], marker="D", linestyle="", alpha=0.8, color="r")
plt.plot(np.arange(len(data_sc1)) +0.067, data_sc2.loc[order]["2030"], marker="D", linestyle="", alpha=0.8, color="b")
plt.plot(np.arange(len(data_sc1)) +0.2 , data_sc1.loc[order]["2040"], marker="D", linestyle="", alpha=0.8, color="r")
plt.plot(np.arange(len(data_sc1)) +0.2, data_sc2.loc[order]["2040"], marker="D", linestyle="", alpha=0.8, color="b")
plt.tick_params(labelbottom=True)
plt.xticks(rotation='horizontal')
plt.ylabel("Number of households")
plt.ylim(0, 880000)
l1 = plt.legend([p1, p2], ["SC1", "SC2"], bbox_to_anchor=(0.6, 1))
l2 = plt.legend([rects1, rects2, rects3, rects4], ["2011", "2020", "2030", "2040"], loc = 1)
plt.gca().add_artist(l1)

### FIGURE 2: TABLE UTILITY

table_utility_sc1 = pd.DataFrame(columns=['Income class 1', 'Income class 2', 'Income class 3', 'Income class 4'], index=['2011','2020', '2030', '2040', 'gain'])
table_utility_sc1.loc['2011'] = simulation_utility_sc1[0, :]
table_utility_sc1.loc['2020'] = simulation_utility_sc1[9, :]
table_utility_sc1.loc['2030'] = simulation_utility_sc1[19, :]
table_utility_sc1.loc['2040'] = simulation_utility_sc1[29, :]
table_utility_sc1.loc['gain'] = (simulation_utility_sc1[29, :] -  simulation_utility_sc1[0, :]) /simulation_utility_sc1[0, :]


table_utility_sc2 = pd.DataFrame(columns=['Income class 1', 'Income class 2', 'Income class 3', 'Income class 4'], index=['2011','2020', '2030', '2040', 'gain'])
table_utility_sc2.loc['2011'] = simulation_utility_sc2[0, :]
table_utility_sc2.loc['2020'] = simulation_utility_sc2[9, :]
table_utility_sc2.loc['2030'] = simulation_utility_sc2[19, :]
table_utility_sc2.loc['2040'] = simulation_utility_sc2[29, :]
table_utility_sc2.loc['gain'] = (simulation_utility_sc2[29, :] -  simulation_utility_sc2[0, :]) /simulation_utility_sc2[0, :]

ratio_utility = simulation_utility_sc2 /  simulation_utility_sc1
plt.plot(np.arange(2011, 2041),ratio_utility[:, 0], label = "Income class 1")
plt.plot(np.arange(2011, 2041),ratio_utility[:, 1], label = "Income class 2")
plt.plot(np.arange(2011, 2041),ratio_utility[:, 2], label = "Income class 3")
plt.plot(np.arange(2011, 2041),ratio_utility[:, 3], label = "Income class 4")
plt.legend()
plt.ylabel('Utility ratio')
plt.show()

no_policy_utility_sc1 = np.load(no_policy_sc1 + '/simulation_utility.npy')
no_policy_utility_sc2 = np.load(no_policy_sc2 + '/simulation_utility.npy')

((simulation_utility_sc1[29,:] /  no_policy_utility_sc1[29,:]) - 1) * 100
((simulation_utility_sc2[29,:]  /  no_policy_utility_sc2[29,:]) - 1) * 100 

welfare_sc1 = simulation_utility_sc1 * np.nansum(np.nansum(simulation_households_sc1, 3), 1)
welfare_sc2 = simulation_utility_sc2 * np.nansum(np.nansum(simulation_households_sc2, 3), 1)
welfare_sc1 = np.nansum(welfare_sc1, 1)
welfare_sc2 = np.nansum(welfare_sc2, 1)

plt.plot(np.arange(2011, 2041), welfare_sc2 / welfare_sc1, label = "Scenario 1")
###In 2011
#utility_2011 = np.log(simulation_utility_sc1[0, :])
#utility_2040 = np.log(simulation_utility_sc1[0, :])

#housing_2011 = np.log(simulation_dwelling_size_sc1[0, :, :] - np.transpose(np.matlib.repmat(np.array([param["q0"], 0, 0, 0]), 24014, 1))) * param["beta"]
#amenities_2011 = np.log(amenities)
#disamenity_parameter_is_2011 = np.log(param["pockets"])
#disamenity_parameter_ib_2011 = np.log(param["backyard_pockets"])
#housing_2040_sc1 = np.log(simulation_dwelling_size_sc1[29, :, :] - np.transpose(np.matlib.repmat(np.array([param["q0"], 0, 0, 0]), 24014, 1))) * param["beta"]

#z_2011 = np.transpose(np.matlib.repmat(utility_2011, 24014, 1)) - housing_2011 - np.matlib.repmat(amenities_2011, 4, 1) - np.array([np.zeros(24014), disamenity_parameter_ib_2011, disamenity_parameter_is_2011, np.zeros(24014)])
#z_2040 = np.transpose(np.matlib.repmat(utility_2040, 24014, 1)) - housing_2040_sc1 - np.matlib.repmat(amenities_2011, 4, 1) - np.array([np.zeros(24014), disamenity_parameter_ib_2011, disamenity_parameter_is_2011, np.zeros(24014)])

#housing_type = 0
#sns.distplot(z_2011[housing_type, :])
#sns.distplot(housing_2011[housing_type, :])
#sns.distplot(np.matlib.repmat(amenities_2011, 4, 1) + np.array([np.zeros(24014), disamenity_parameter_ib_2011, disamenity_parameter_is_2011, np.zeros(24014)]))
#sns.distplot(utility_2011[housing_type, :])

#Composite good consumption

param["pockets"] = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/fluvial_and_pluvial/param_pockets.npy')
param["backyard_pockets"] = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/fluvial_and_pluvial/param_backyards.npy')
param["pockets"][(spline_land_informal(29) > 0) & (spline_land_informal(0) == 0)] = 0.79

### FIGURE 3: WEIGHTED AVERAGE RENTS

def export_map(value, grid):
    map = plt.scatter(grid.x, 
            grid.y, 
            s=None,
            c=value,
            cmap = 'Reds',
            marker='.')
    plt.colorbar(map)
    plt.axis('off')
    #plt.clim(-1,-4)
    #plt.clim(0,150)
    #plt.clim(0,5000)

variation_rents_formal = (simulation_rent_sc1[29, 0, :] - simulation_rent_sc2[29, 0, :]) * 100 / simulation_rent_sc2[29, 0, :]
variation_rents_backyard = (simulation_rent_sc1[29, 1, :] - simulation_rent_sc2[29, 1, :]) * 100 / simulation_rent_sc2[29, 1, :]
variation_rents_informal = (simulation_rent_sc1[29, 2, :] - simulation_rent_sc2[29, 2, :]) * 100 / simulation_rent_sc2[29, 2, :]

variation_rents_formal[(simulation_households_housing_type_sc1[29, 0, :] < 1) & (simulation_households_housing_type_sc2[29, 0, :] < 1)] = np.nan
variation_rents_backyard[(simulation_households_housing_type_sc1[29, 1, :] < 1) & (simulation_households_housing_type_sc2[29, 1, :] < 1)] = np.nan
variation_rents_informal[(simulation_households_housing_type_sc1[29, 2, :] < 1) & (simulation_households_housing_type_sc2[29, 2, :] < 1)] = np.nan

variation_rents = pd.DataFrame()
variation_rents["formal"] = variation_rents_formal
variation_rents["backyard"] = variation_rents_backyard
variation_rents["informal"] = variation_rents_informal
variation_rents.to_excel('C:/Users/charl/OneDrive/Bureau/cape_town/results_20210330/variation_rents.xlsx')
export_map(variation_rents_formal, grid)
export_map(variation_rents_backyard, grid)
export_map(variation_rents_informal, grid)
### FIGURE 4: EXPOSURE TO FLOODS IN 2011

#Choisir fluvial ou pluvial

stats_per_housing_type_2011 = compute_stats_per_housing_type(fluvial_floods, path_data, simulation_households_housing_type_sc1[0, 0, :], simulation_households_housing_type_sc1[0, 3, :], simulation_households_housing_type_sc1[0, 2, :], simulation_households_housing_type_sc1[0, 1, :], "X", "X", 0.01)
#stats_per_housing_type_2011 = compute_stats_per_housing_type(pluvial_floods, path_data, simulation_households_housing_type_sc1[0, 0, :], simulation_households_housing_type_sc1[0, 3, :], simulation_households_housing_type_sc1[0, 2, :], simulation_households_housing_type_sc1[0, 1, :], "X", "X", 0.01)

plt.rcParams.update({'font.size': 21})
label = ["Formal \n private", "Formal \n subsidized", "Informal \n in backyards", "Informal \n settlements"]
stats_2011_1 = [stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[2], stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[2], stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[2], stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[2]]
stats_2011_2 = [stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[3], stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[3], stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[3], stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[3]]
stats_2011_3 = [stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[5]]
colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
r = np.arange(4)
barWidth = 0.25
plt.figure(figsize=(10,7))
plt.bar(r, stats_2011_1, color=colors[0], edgecolor='white', label="20 years")
plt.bar(r, np.array(stats_2011_2) - np.array(stats_2011_1), bottom=np.array(stats_2011_1), color=colors[1], edgecolor='white', label='50 years')
plt.bar(r, np.array(stats_2011_3) - (np.array(stats_2011_2)), bottom=(np.array(stats_2011_2)), color=colors[2], edgecolor='white', label='100 years')
plt.legend(loc = 'upper right')
plt.xticks(r, label)
#plt.ylim(0, 55000)
plt.tick_params(labelbottom=True)
plt.ylabel("Dwellings in flood-prone areas")
plt.show()

stats_per_housing_type_2011["tot"] = stats_per_housing_type_2011.fraction_formal_in_flood_prone_area + np.array(stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area) + stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area + stats_per_housing_type_2011.fraction_informal_in_flood_prone_area

plt.figure(figsize=(10,7))
barWidth = 0.25
vec_2011_formal = stats_per_housing_type_2011.fraction_formal_in_flood_prone_area / stats_per_housing_type_2011["tot"]
vec_2011_formal = [np.nansum(simulation_households_housing_type_sc1[0, :, :], 1)[0] / sum(np.nansum(simulation_households_housing_type_sc1[0, :, :], 1)), vec_2011_formal[2], vec_2011_formal[3],vec_2011_formal[5]]
vec_2011_subsidized = stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area / stats_per_housing_type_2011["tot"]
vec_2011_subsidized = [np.nansum(simulation_households_housing_type_sc1[0, :, :], 1)[3]/ sum(np.nansum(simulation_households_housing_type_sc1[0, :, :], 1)), vec_2011_subsidized[2], vec_2011_subsidized[3],vec_2011_subsidized[5]]
vec_2011_informal = stats_per_housing_type_2011.fraction_informal_in_flood_prone_area / stats_per_housing_type_2011["tot"]
vec_2011_informal = [np.nansum(simulation_households_housing_type_sc1[0, :, :], 1)[2]/ sum(np.nansum(simulation_households_housing_type_sc1[0, :, :], 1)),vec_2011_informal[2], vec_2011_informal[3],vec_2011_informal[5]]
vec_2011_backyard = stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area / stats_per_housing_type_2011["tot"]
vec_2011_backyard = [np.nansum(simulation_households_housing_type_sc1[0, :, :], 1)[1]/ sum(np.nansum(simulation_households_housing_type_sc1[0, :, :], 1)),vec_2011_backyard[2], vec_2011_backyard[3],vec_2011_backyard[5]]
plt.ylim(0, 1.3)
label = ["Over the city", "In 20-year \n return period \n flood zones","In 50-year \n return period \n flood zones","In 100-year \n return period \n flood zones"]
plt.bar(np.arange(4), vec_2011_formal, color=colors[0], edgecolor='white', label="Formal private")
plt.bar(np.arange(4), vec_2011_subsidized, bottom=vec_2011_formal, color=colors[1], edgecolor='white', label="Formal subsidized")
plt.bar(np.arange(4), vec_2011_informal, bottom=np.array(vec_2011_formal) + np.array(vec_2011_subsidized), color=colors[2], edgecolor='white', label="Informal settlements")
plt.bar(np.arange(4), vec_2011_backyard, bottom=np.array(vec_2011_formal) + np.array(vec_2011_subsidized) + np.array(vec_2011_informal), color=colors[3], edgecolor='white', label="Informal in backyards")
plt.legend(loc = 'upper left')
plt.ylabel("Fraction of dwellings of each housing type")
plt.xticks(np.arange(4), label)

### FIGURE 5: FLOOD DAMAGES WITH A PDF OR A CDF - 100 year return period (2011)

simulation_dwelling_size = simulation_dwelling_size_sc2
simulation_rent = simulation_rent_sc2
simulation_households_housing_type = simulation_households_housing_type_sc2
simulation_households_center = households_center_sc2
simulation_households = simulation_households_sc2
name = '20213003_79_1_0'
floods = pluvial_floods


spline_agricultural_rent, spline_interest_rate, spline_RDP, spline_population_income_distribution, spline_inflation, spline_income_distribution, spline_population, spline_interest_rate, spline_income, spline_minimum_housing_supply, spline_fuel = import_scenarios(income_2011, param, grid, path_scenarios)
formal_structure_cost_2011 = compute_formal_structure_cost_method2(simulation_rent[0, :, :], param, interpolate_interest_rate(spline_interest_rate, 0), import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 0), simulation_households_housing_type[0, :, :], (spline_income(0) / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"])    
content_cost_2011 = compute_content_cost(simulation_households_center[0, :, :], np.load(precalculated_transport + "year_0.npy"), param, fraction_capital_destroyed, simulation_rent[0, :, :], simulation_dwelling_size[0, :, :], interpolate_interest_rate(spline_interest_rate, 0))
formal_structure_cost_2040 = compute_formal_structure_cost_method2(simulation_rent[28, :, :], param, interpolate_interest_rate(spline_interest_rate, 28), import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 28), simulation_households_housing_type[28, :, :], (spline_income(28) / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"])    
content_cost_2040 = compute_content_cost(simulation_households_center[28, :, :], np.load(precalculated_transport + "year_29.npy"), param, fraction_capital_destroyed, simulation_rent[28, :, :], simulation_dwelling_size[28, :, :], interpolate_interest_rate(spline_interest_rate, 28))


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
    writer = pd.ExcelWriter(path_outputs + name + '/damages_' + str(item) + '_2011.xlsx')
    df2011.to_excel(excel_writer = writer)
    writer.save()
    writer = pd.ExcelWriter(path_outputs + name + '/damages_' + str(item) + '_2040.xlsx')
    
    df2040.to_excel(excel_writer = writer)
    writer.save()
   
damages_5yr_2011 = pd.read_excel(path_outputs + name + '/damages_' + floods[0] + '_2011.xlsx')  
damages_10yr_2011 = pd.read_excel(path_outputs + name + '/damages_' + floods[1] + '_2011.xlsx')  
damages_20yr_2011 = pd.read_excel(path_outputs + name + '/damages_' + floods[2] + '_2011.xlsx')  
damages_50yr_2011 = pd.read_excel(path_outputs + name + '/damages_' + floods[3] + '_2011.xlsx')  
damages_75yr_2011 = pd.read_excel(path_outputs + name + '/damages_' + floods[4] + '_2011.xlsx')  
damages_100yr_2011 = pd.read_excel(path_outputs + name + '/damages_' + floods[5] + '_2011.xlsx')  
damages_200yr_2011 = pd.read_excel(path_outputs + name + '/damages_' + floods[6] + '_2011.xlsx')  
damages_250yr_2011 = pd.read_excel(path_outputs + name + '/damages_' + floods[7] + '_2011.xlsx')  
damages_500yr_2011 = pd.read_excel(path_outputs + name + '/damages_' + floods[8] + '_2011.xlsx')  
damages_1000yr_2011 = pd.read_excel(path_outputs + name + '/damages_' + floods[9] + '_2011.xlsx')  

damages_5yr_2040 = pd.read_excel(path_outputs + name + '/damages_' + floods[0] + '_2040.xlsx')  
damages_10yr_2040 = pd.read_excel(path_outputs + name + '/damages_' + floods[1] + '_2040.xlsx')
damages_20yr_2040 = pd.read_excel(path_outputs + name + '/damages_' + floods[2] + '_2040.xlsx')  
damages_50yr_2040 = pd.read_excel(path_outputs + name + '/damages_' + floods[3] + '_2040.xlsx')  
damages_75yr_2040 = pd.read_excel(path_outputs + name + '/damages_' + floods[4] + '_2040.xlsx')  
damages_100yr_2040 = pd.read_excel(path_outputs + name + '/damages_' + floods[5] + '_2040.xlsx')  
damages_200yr_2040 = pd.read_excel(path_outputs + name + '/damages_' + floods[6] + '_2040.xlsx')  
damages_250yr_2040 = pd.read_excel(path_outputs + name + '/damages_' + floods[7] + '_2040.xlsx')  
damages_500yr_2040 = pd.read_excel(path_outputs + name + '/damages_' + floods[8] + '_2040.xlsx')  
damages_1000yr_2040 = pd.read_excel(path_outputs + name + '/damages_' + floods[9] + '_2040.xlsx')  

#Flood prone population: on veut la population qui est affectée par les 10yr return period mais pas par les 5 yr return period,...
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
average_income_2011 = np.load(precalculated_transport + "average_income_year_0.npy")

income_class_2040 = np.argmax(simulation_households[28, :, :, :], 1)     
average_income_2040 = np.load(precalculated_transport + "average_income_year_28.npy")
    

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

### FIGURE 6: SEVERITY OF FLOODS IN MM BY HH TYPE OR INCOME GROUP - 100 year return period (2011)


simulation_dwelling_size = simulation_dwelling_size_sc2
simulation_rent = simulation_rent_sc2
simulation_households_housing_type = simulation_households_housing_type_sc2
simulation_households_center = households_center_sc2
simulation_households = simulation_households_sc2
name = '20213003_79_1_0'

floods = fluvial_floods
type_flood = 'FD_100yr'

data_flood = np.squeeze(pd.read_excel(path_data + type_flood + ".xlsx"))

data_flood["formal_pop_flood_prone"] = simulation_households_housing_type[0, 0, :] * data_flood["prop_flood_prone"]
data_flood["backyard_pop_flood_prone"] = simulation_households_housing_type[0, 1, :] * data_flood["prop_flood_prone"]
data_flood["informal_pop_flood_prone"] = simulation_households_housing_type[0, 2, :] * data_flood["prop_flood_prone"]
data_flood["subsidized_pop_flood_prone"] = simulation_households_housing_type[0, 3, :] * data_flood["prop_flood_prone"]
    
if item == "P_20yr":
    data_flood["formal_pop_flood_prone"] = 0
elif ((item == "P_5yr") |(item == "P_10yr")):
    data_flood["formal_pop_flood_prone"] = 0
    data_flood["backyard_pop_flood_prone"] = 0
    data_flood["subsidized_pop_flood_prone"] = 0
    

income_class_2011 = np.argmax(simulation_households[0, :, :, :], 1)    
average_income_2011 = np.load(precalculated_transport + "average_income_year_0.npy")


real_income_2011 = np.empty((24014, 4))
for i in range(0, 24014):
    for j in range(0, 4):
        print(i)
        real_income_2011[i, j] = average_income_2011[np.array(income_class_2011)[j, i], i]


## Reshape
formal_2011 = data_flood.iloc[:, [0, 2]]
backyard_2011 = data_flood.iloc[:, [0, 3]]
informal_2011 = data_flood.iloc[:, [0, 4]]
subsidized_2011 = data_flood.iloc[:, [0, 5]]


### C'est le moment de subsuet by income class

formal_2011_class1 = formal_2011[income_class_2011[0, :] == 0]
formal_2011_class2 = formal_2011[income_class_2011[0, :] == 1]
formal_2011_class3 = formal_2011[income_class_2011[0, :] == 2]
formal_2011_class4 = formal_2011[income_class_2011[0, :] == 3]

subsidized_2011_class1 = subsidized_2011[income_class_2011[3, :] == 0]
subsidized_2011_class2 = subsidized_2011[income_class_2011[3, :] == 1]
subsidized_2011_class3 = subsidized_2011[income_class_2011[3, :] == 2]
subsidized_2011_class4 = subsidized_2011[income_class_2011[3, :] == 3]

backyard_2011_class1 = backyard_2011[income_class_2011[1, :] == 0]
backyard_2011_class2 = backyard_2011[income_class_2011[1, :] == 1]
backyard_2011_class3 = backyard_2011[income_class_2011[1, :] == 2]
backyard_2011_class4 = backyard_2011[income_class_2011[1, :] == 3]


informal_2011_class1 = informal_2011[income_class_2011[2, :] == 0]
informal_2011_class2 = informal_2011[income_class_2011[2, :] == 1]
informal_2011_class3 = informal_2011[income_class_2011[2, :] == 2]
informal_2011_class4 = informal_2011[income_class_2011[2, :] == 3]


#Total
array_2011_class1 = np.vstack([formal_2011_class1, backyard_2011_class1, informal_2011_class1, subsidized_2011_class1])
array_2011_class2 = np.vstack([formal_2011_class2, backyard_2011_class2, informal_2011_class2, subsidized_2011_class2])
array_2011_class3 = np.vstack([formal_2011_class3, backyard_2011_class3, informal_2011_class3, subsidized_2011_class3])
array_2011_class4 = np.vstack([formal_2011_class4, backyard_2011_class4, informal_2011_class4, subsidized_2011_class4])


#sns.distplot(array_2011_class1[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': array_2011_class1[:,1]}, color = 'black', label = "2011", kde = False)
#sns.distplot(array_2011_class2[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': array_2011_class2[:,1]}, color = 'black', label = "2011", kde = False)
#sns.distplot(array_2011_class3[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': array_2011_class3[:,1]}, color = 'black', label = "2011", kde = False)
sns.distplot(array_2011_class4[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': array_2011_class4[:,1]}, color = 'black', label = "2011", kde = False)
plt.ylim(0, 13000)
plt.xlabel("Severity of floods (m)")
plt.ylabel("Number of households")

g = sns.FacetGrid(tips, col="size", height=2, col_wrap=3)
g.map(sns.histplot, "total_bill")

### FIGURE 7: FLOOD DAMAGES WITH A PDF OR A CDF FOR 4 GROUPS ON 1 CHART - 100 year return period (2011)

### FIGURE 8: EXPOSURE TO FLOOD (FIG 8 - 11)

#Choisir fluvial ou pluvial et sc1 ou sc2

simulation_households_housing_type = simulation_households_housing_type_sc1

stats_per_housing_type_2011 = compute_stats_per_housing_type(fluvial_floods, path_data, simulation_households_housing_type[0, 0, :], simulation_households_housing_type[0, 3, :], simulation_households_housing_type[0, 2, :], simulation_households_housing_type[0, 1, :], "X", "X", 0.01)
stats_per_housing_type_2040 = compute_stats_per_housing_type(fluvial_floods, path_data, simulation_households_housing_type[28, 0, :], simulation_households_housing_type[28, 3, :], simulation_households_housing_type[28, 2, :], simulation_households_housing_type[28, 1, :], "X", "X", 0.01)

plt.rcParams.update({'font.size': 21})
label = ["Formal \n private", "Formal \n subsidized", "Informal \n in backyards", "Informal \n settlements"]
stats_2011_1 = [stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[2], stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[2], stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[2], stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[2]]
stats_2011_2 = [stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[3], stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[3], stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[3], stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[3]]
stats_2011_3 = [stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[5]]
stats_2040_1 = [stats_per_housing_type_2040.fraction_formal_in_flood_prone_area[2], stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area[2], stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area[2], stats_per_housing_type_2040.fraction_informal_in_flood_prone_area[2]]
stats_2040_2 = [stats_per_housing_type_2040.fraction_formal_in_flood_prone_area[3], stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area[3], stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area[3], stats_per_housing_type_2040.fraction_informal_in_flood_prone_area[3]]
stats_2040_3 = [stats_per_housing_type_2040.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area[5], stats_per_housing_type_2040.fraction_informal_in_flood_prone_area[5]]
colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
r = np.arange(4)
barWidth = 0.25
plt.figure(figsize=(10,7))
plt.bar(r, stats_2011_1, color=colors[0], edgecolor='white', width=barWidth, label="20 years")
plt.bar(r, np.array(stats_2011_2) - np.array(stats_2011_1), bottom=np.array(stats_2011_1), color=colors[1], edgecolor='white', width=barWidth, label='50 years')
plt.bar(r, np.array(stats_2011_3) - (np.array(stats_2011_2)), bottom=(np.array(stats_2011_2)), color=colors[2], edgecolor='white', width=barWidth, label='100 years')
plt.bar(r + 0.25, np.array(stats_2040_1), color=colors[0], edgecolor='white', width=barWidth)
plt.bar(r + 0.25, np.array(stats_2040_2) - np.array(stats_2040_1), bottom=np.array(stats_2040_1), color=colors[1], edgecolor='white', width=barWidth)
plt.bar(r + 0.25, np.array(stats_2040_3) - np.array(stats_2040_2), bottom=np.array(stats_2040_2), color=colors[2], edgecolor='white', width=barWidth)
plt.legend(bbox_to_anchor=(0.3, 0.9))
plt.xticks(r, label)
plt.ylim(0, 76000)
#plt.ylim(0, 300000)
plt.text(r[0] - 0.125, stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[5] + 3000, "2011",fontsize =  15)
plt.text(r[1] - 0.125, stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[5] + 3000, "2011",fontsize =  15) 
plt.text(r[3] - 0.125, stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[5] +3000, "2011",fontsize =  15) 
plt.text(r[2] - 0.125, stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[5] + 3000, "2011",fontsize =  15)
plt.text(r[0] + 0.125, stats_per_housing_type_2040.fraction_formal_in_flood_prone_area[5] + 3000, '2040',fontsize =  15)
plt.text(r[1] + 0.125, stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area[5] +3000, '2040',fontsize =  15) 
plt.text(r[3] + 0.125, stats_per_housing_type_2040.fraction_informal_in_flood_prone_area[5] + 3000, '2040',fontsize =  15) 
plt.text(r[2] + 0.125, stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area[5] + 3000, '2040',fontsize =  15) 
plt.tick_params(labelbottom=True)
plt.ylabel("Dwellings in flood-prone areas")
plt.show()

floods_2011 = stats_per_housing_type_2011.loc[stats_per_housing_type_2011.flood == 'FD_100yr'].iloc[:, 1:5]
population_2011 = np.nansum(simulation_households_housing_type[0, :, :], 1)
risk_formal_2011 = floods_2011.fraction_formal_in_flood_prone_area / population_2011[0]
risk_subsidized_2011 = floods_2011.fraction_subsidized_in_flood_prone_area / population_2011[3]
risk_informal_2011 = floods_2011.fraction_informal_in_flood_prone_area / population_2011[2]
risk_backyard_2011 = floods_2011.fraction_backyard_in_flood_prone_area / population_2011[1]

floods_2040 = stats_per_housing_type_2040.loc[stats_per_housing_type_2040.flood == 'FD_100yr'].iloc[:, 1:5]
population_2040 = np.nansum(simulation_households_housing_type[29, :, :], 1)
risk_formal_2040 = floods_2040.fraction_formal_in_flood_prone_area / population_2040[0]
risk_subsidized_2040 = floods_2040.fraction_subsidized_in_flood_prone_area / population_2040[3]
risk_informal_2040 = floods_2040.fraction_informal_in_flood_prone_area / population_2040[2]
risk_backyard_2040 = floods_2040.fraction_backyard_in_flood_prone_area / population_2040[1]

stats_per_housing_type_2011["tot"] = stats_per_housing_type_2011.fraction_formal_in_flood_prone_area + np.array(stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area) + stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area + stats_per_housing_type_2011.fraction_informal_in_flood_prone_area
stats_per_housing_type_2040["tot"] = stats_per_housing_type_2040.fraction_formal_in_flood_prone_area + np.array(stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area) + stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area + stats_per_housing_type_2040.fraction_informal_in_flood_prone_area

plt.figure(figsize=(10,7))
barWidth = 0.25
vec_2011_formal = stats_per_housing_type_2011.fraction_formal_in_flood_prone_area / stats_per_housing_type_2011["tot"]
vec_2011_formal = [np.nansum(simulation_households_housing_type[0, :, :], 1)[0] / sum(np.nansum(simulation_households_housing_type[0, :, :], 1)), vec_2011_formal[2], vec_2011_formal[3],vec_2011_formal[5]]
vec_2011_subsidized = stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area / stats_per_housing_type_2011["tot"]
vec_2011_subsidized = [np.nansum(simulation_households_housing_type[0, :, :], 1)[3]/ sum(np.nansum(simulation_households_housing_type[0, :, :], 1)), vec_2011_subsidized[2], vec_2011_subsidized[3],vec_2011_subsidized[5]]
vec_2011_informal = stats_per_housing_type_2011.fraction_informal_in_flood_prone_area / stats_per_housing_type_2011["tot"]
vec_2011_informal = [np.nansum(simulation_households_housing_type[0, :, :], 1)[2]/ sum(np.nansum(simulation_households_housing_type[0, :, :], 1)),vec_2011_informal[2], vec_2011_informal[3],vec_2011_informal[5]]
vec_2011_backyard = stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area / stats_per_housing_type_2011["tot"]
vec_2011_backyard = [np.nansum(simulation_households_housing_type[0, :, :], 1)[1]/ sum(np.nansum(simulation_households_housing_type[0, :, :], 1)),vec_2011_backyard[2], vec_2011_backyard[3],vec_2011_backyard[5]]
vec_2040_formal = stats_per_housing_type_2040.fraction_formal_in_flood_prone_area / stats_per_housing_type_2040["tot"]
vec_2040_formal = [np.nansum(simulation_households_housing_type[28, :, :], 1)[0]/ sum(np.nansum(simulation_households_housing_type[28, :, :], 1)), vec_2040_formal[2], vec_2040_formal[3],vec_2040_formal[5]]
vec_2040_subsidized = stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area / stats_per_housing_type_2040["tot"]
vec_2040_subsidized = [np.nansum(simulation_households_housing_type[28, :, :], 1)[3]/ sum(np.nansum(simulation_households_housing_type[28, :, :], 1)), vec_2040_subsidized[2], vec_2040_subsidized[3],vec_2040_subsidized[5]]
vec_2040_informal = stats_per_housing_type_2040.fraction_informal_in_flood_prone_area / stats_per_housing_type_2040["tot"]
vec_2040_informal = [np.nansum(simulation_households_housing_type[28, :, :], 1)[2]/ sum(np.nansum(simulation_households_housing_type[28, :, :], 1)),vec_2040_informal[2], vec_2040_informal[3],vec_2040_informal[5]]
vec_2040_backyard = stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area / stats_per_housing_type_2040["tot"]
vec_2040_backyard = [np.nansum(simulation_households_housing_type[28, :, :], 1)[1]/ sum(np.nansum(simulation_households_housing_type[28, :, :], 1)),vec_2040_backyard[2], vec_2040_backyard[3],vec_2040_backyard[5]]
plt.ylim(0, 1.3)
label = ["Over the city", "In 20-year \n return period \n flood zones","In 50-year \n return period \n flood zones","In 100-year \n return period \n flood zones"]
plt.bar(np.arange(4), vec_2011_formal, color=colors[0], edgecolor='white', width=barWidth, label="Formal private")
plt.bar(np.arange(4), vec_2011_subsidized, bottom=vec_2011_formal, color=colors[1], edgecolor='white', width=barWidth, label="Formal subsidized")
plt.bar(np.arange(4), vec_2011_informal, bottom=np.array(vec_2011_formal) + np.array(vec_2011_subsidized), color=colors[2], edgecolor='white', width=barWidth, label="Informal settlements")
plt.bar(np.arange(4), vec_2011_backyard, bottom=np.array(vec_2011_formal) + np.array(vec_2011_subsidized) + np.array(vec_2011_informal), color=colors[3], edgecolor='white', width=barWidth, label="Informal in backyards")
plt.bar(np.arange(4) + 0.25, vec_2040_formal, color=colors[0], edgecolor='white', width=barWidth)
plt.bar(np.arange(4) + 0.25, vec_2040_subsidized, bottom=vec_2040_formal, color=colors[1], edgecolor='white', width=barWidth)
plt.bar(np.arange(4) + 0.25, vec_2040_informal, bottom=np.array(vec_2040_formal) + np.array(vec_2040_subsidized), color=colors[2], edgecolor='white', width=barWidth)
plt.bar(np.arange(4) + 0.25, vec_2040_backyard, bottom=np.array(vec_2040_formal) + np.array(vec_2040_subsidized) + np.array(vec_2040_informal), color=colors[3], edgecolor='white', width=barWidth)
plt.legend(loc = 'upper left')
plt.ylabel("Fraction of dwellings of each housing type")
plt.xticks(np.arange(4), label)
plt.text(r[0] - 0.1, 1.005, "2011")
plt.text(r[1] - 0.1, 1.005, "2011") 
plt.text(r[2] - 0.1, 1.005, "2011") 
plt.text(r[3] - 0.1, 1.005, "2011")
plt.text(r[0] + 0.15, 1.005, '2040')
plt.text(r[1] + 0.15, 1.005, '2040') 
plt.text(r[2] + 0.15, 1.005, '2040') 
plt.text(r[3] + 0.15, 1.005, '2040') 


### FIGURE 9: FLOOD DAMAGES WITH A PDF OR A CDF - 100 year return period

simulation_dwelling_size = simulation_dwelling_size_sc2
simulation_rent = simulation_rent_sc2
simulation_households_housing_type = simulation_households_housing_type_sc2
simulation_households_center = households_center_sc2
simulation_households = simulation_households_sc2
name = '20213003_79_1_0'

floods = fluvial_floods
type_flood = 'FD_100yr'


spline_agricultural_rent, spline_interest_rate, spline_RDP, spline_population_income_distribution, spline_inflation, spline_income_distribution, spline_population, spline_interest_rate, spline_income, spline_minimum_housing_supply, spline_fuel = import_scenarios(income_2011, param, grid, path_scenarios)
formal_structure_cost_2011 = compute_formal_structure_cost_method2(simulation_rent[0, :, :], param, interpolate_interest_rate(spline_interest_rate, 0), import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 0), simulation_households_housing_type[0, :, :], (spline_income(0) / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"])    
content_cost_2011 = compute_content_cost(simulation_households_center[0, :, :], np.load(precalculated_transport + "year_0.npy"), param, fraction_capital_destroyed, simulation_rent[0, :, :], simulation_dwelling_size[0, :, :], interpolate_interest_rate(spline_interest_rate, 0))
formal_structure_cost_2040 = compute_formal_structure_cost_method2(simulation_rent[28, :, :], param, interpolate_interest_rate(spline_interest_rate, 28), import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 28), simulation_households_housing_type[28, :, :], (spline_income(28) / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"])    
content_cost_2040 = compute_content_cost(simulation_households_center[28, :, :], np.load(precalculated_transport + "year_29.npy"), param, fraction_capital_destroyed, simulation_rent[28, :, :], simulation_dwelling_size[28, :, :], interpolate_interest_rate(spline_interest_rate, 28))


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
    writer = pd.ExcelWriter(path_outputs + name + '/damages_' + str(item) + '_2011.xlsx')
    df2011.to_excel(excel_writer = writer)
    writer.save()
    writer = pd.ExcelWriter(path_outputs + name + '/damages_' + str(item) + '_2040.xlsx')
    
    df2040.to_excel(excel_writer = writer)
    writer.save()
   
damages_2011 = pd.read_excel(path_outputs + name + '/damages_' + type_flood + '_2011.xlsx')  
damages_2040 = pd.read_excel(path_outputs + name + '/damages_' + type_flood + '_2040.xlsx')  



damages_2011 = damages_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
damages_2040 = damages_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']

income_class_2011 = np.argmax(simulation_households[0, :, :, :], 1)    
average_income_2011 = np.load(precalculated_transport + "average_income_year_0.npy")

income_class_2040 = np.argmax(simulation_households[28, :, :, :], 1)     
average_income_2040 = np.load(precalculated_transport + "average_income_year_28.npy")
    

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
    

damages_2011.iloc[:, 4] = (damages_2011.iloc[:, 4] / real_income_2011[:,0]) * 100
damages_2011.iloc[:, 5] = (damages_2011.iloc[:, 5] / real_income_2011[:,2]) * 100
damages_2011.iloc[:, 6] = (damages_2011.iloc[:, 6] / real_income_2011[:,3]) * 100
damages_2011.iloc[:, 7] = (damages_2011.iloc[:, 7] / real_income_2011[:,1]) * 100

damages_2040.iloc[:, 4] = (damages_2040.iloc[:, 4] / real_income_2040[:,0]) * 100
damages_2040.iloc[:, 5] = (damages_2040.iloc[:, 5] / real_income_2040[:,2]) * 100
damages_2040.iloc[:, 6] = (damages_2040.iloc[:, 6] / real_income_2040[:,3]) * 100
damages_2040.iloc[:, 7] = (damages_2040.iloc[:, 7] / real_income_2040[:,1]) * 100

## Reshape
formal_2011 = damages_2011.iloc[:, [0, 4]]
backyard_2011 = damages_2011.iloc[:, [1, 7]]
informal_2011 = damages_2011.iloc[:, [2, 5]]
subsidized_2011 = damages_2011.iloc[:, [3, 6]]

formal_2040 = damages_2040.iloc[:, [0, 4]]
backyard_2040 = damages_2040.iloc[:, [1, 7]]
informal_2040 = damages_2040.iloc[:, [2, 5]]
subsidized_2040 = damages_2040.iloc[:, [3, 6]]

### C'est le moment de subsuet by income class

formal_2011_class1 = formal_2011[income_class_2011[0, :] == 0]
formal_2011_class2 = formal_2011[income_class_2011[0, :] == 1]
formal_2011_class3 = formal_2011[income_class_2011[0, :] == 2]
formal_2011_class4 = formal_2011[income_class_2011[0, :] == 3]

formal_2040_class1 = formal_2040[income_class_2040[0, :] == 0]
formal_2040_class2 = formal_2040[income_class_2040[0, :] == 1]
formal_2040_class3 = formal_2040[income_class_2040[0, :] == 2]
formal_2040_class4 = formal_2040[income_class_2040[0, :] == 3]

subsidized_2011_class1 = subsidized_2011[income_class_2011[3, :] == 0]
subsidized_2011_class2 = subsidized_2011[income_class_2011[3, :] == 1]
subsidized_2011_class3 = subsidized_2011[income_class_2011[3, :] == 2]
subsidized_2011_class4 = subsidized_2011[income_class_2011[3, :] == 3]

subsidized_2040_class1 = subsidized_2040[income_class_2040[3, :] == 0]
subsidized_2040_class2 = subsidized_2040[income_class_2040[3, :] == 1]
subsidized_2040_class3 = subsidized_2040[income_class_2040[3, :] == 2]
subsidized_2040_class4 = subsidized_2040[income_class_2040[3, :] == 3]

backyard_2011_class1 = backyard_2011[income_class_2011[1, :] == 0]
backyard_2011_class2 = backyard_2011[income_class_2011[1, :] == 1]
backyard_2011_class3 = backyard_2011[income_class_2011[1, :] == 2]
backyard_2011_class4 = backyard_2011[income_class_2011[1, :] == 3]

backyard_2040_class1 = backyard_2040[income_class_2040[1, :] == 0]
backyard_2040_class2 = backyard_2040[income_class_2040[1, :] == 1]
backyard_2040_class3 = backyard_2040[income_class_2040[1, :] == 2]
backyard_2040_class4 = backyard_2040[income_class_2040[1, :] == 3]

informal_2011_class1 = informal_2011[income_class_2011[2, :] == 0]
informal_2011_class2 = informal_2011[income_class_2011[2, :] == 1]
informal_2011_class3 = informal_2011[income_class_2011[2, :] == 2]
informal_2011_class4 = informal_2011[income_class_2011[2, :] == 3]

informal_2040_class1 = informal_2040[income_class_2040[2, :] == 0]
informal_2040_class2 = informal_2040[income_class_2040[2, :] == 1]
informal_2040_class3 = informal_2040[income_class_2040[2, :] == 2]
informal_2040_class4 = informal_2040[income_class_2040[2, :] == 3]

#Total
array_2011 = np.vstack([formal_2011, backyard_2011, informal_2011, subsidized_2011])
subset_2011 = array_2011[~np.isnan(array_2011[:,1])]
array_2040 = np.vstack([formal_2040, backyard_2040, informal_2040, subsidized_2040])
subset_2040 = array_2040[~np.isnan(array_2040[:,1])]

sns.distplot(subset_2011[:,1],
             bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2011[:,0]}, color = 'black', label = "2011")
sns.distplot(subset_2040[:,1],
             bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2040[:,0]}, label = "2040")

plt.legend()
plt.xlabel("Share of the annual income destroyed by 100-year return period floods (%)")
plt.ylabel("Number of households")
plt.ylim(0, 35000)
#plt.ylim(0, 380000)


### FIGURE 10: FLOOD DAMAGES WITH A PDF OR A CDF - 100 year return period - FOCUS ON FLUVIAL NO EVICTION


array_2011_class4 = np.vstack([formal_2011_class4, backyard_2011_class4, informal_2011_class4, subsidized_2011_class4])
subset_2011_class4 = array_2011_class4[~np.isnan(array_2011_class4[:,1])]

array_2011_class3 = np.vstack([formal_2011_class3, backyard_2011_class3, informal_2011_class3, subsidized_2011_class3])
subset_2011_class3 = array_2011_class3[~np.isnan(array_2011_class3[:,1])]

array_2011_class2 = np.vstack([formal_2011_class2, backyard_2011_class2, informal_2011_class2, subsidized_2011_class2])
subset_2011_class2 = array_2011_class2[~np.isnan(array_2011_class2[:,1])]

array_2011_class1 = np.vstack([formal_2011_class1, backyard_2011_class1, informal_2011_class1, subsidized_2011_class1])
subset_2011_class1 = array_2011_class1[~np.isnan(array_2011_class1[:,1])]



sns.distplot(subset_2011_class1[:,1],
             bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2011_class1[:,0]}, color = 'blue', label = "Income class 1")
sns.distplot(subset_2011_class2[:,1],
             bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2011_class2[:,0]}, color = 'red', label = "Income class 2")
sns.distplot(subset_2011_class3[:,1],
             bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2011_class3[:,0]}, color = 'yellow', label = "Income class 3")
sns.distplot(subset_2011_class4[:,1],
             bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2011_class4[:,0]}, color = 'green', label = "Income class 4")
plt.legend()
plt.xlabel("Share of the annual income destroyed by 100-year return period floods (%)")
plt.ylabel("Number of households")
plt.xlim(0, 250)