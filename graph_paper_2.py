# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:25:50 2021

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
path_outputs = "C:/Users/charl/OneDrive/Bureau/graphs_20210426/"
os.mkdir(path_outputs)

#### WELFARE

welfare_sc1 = simulation_utility_sc1 * np.nansum(np.nansum(simulation_households_sc1, 3), 1)
welfare_sc2 = simulation_utility_sc2 * np.nansum(np.nansum(simulation_households_sc2, 3), 1)
welfare_sc1 = np.nansum(welfare_sc1, 1)
welfare_sc2 = np.nansum(welfare_sc2, 1)

plt.figure(figsize = (7, 5))
plt.rcParams.update({'font.size': 15})
plt.plot(np.arange(2011, 2041), welfare_sc2 / welfare_sc1, label = "Scenario 1")
plt.savefig(path_outputs + 'welfare_ratio_sc2_sc1.png')
plt.close()

### FLOOD EXPOSURE

#Options: 2011 or 2011_2040, by income class or by housing type, percent or absolute, fluvial or pluvial, sc1 or sc2

def plot_flood_exposure(year, group, scale, type_flood, scenario, path_outputs):
    
    if scenario == "sc1":
        simulation_households_housing_type = simulation_households_housing_type_sc1
        simulation_households = simulation_households_sc1
    elif scenario == "sc2":
        simulation_households_housing_type = simulation_households_housing_type_sc2
        simulation_households = simulation_households_sc2
        
    if (type_flood == fluvial_floods) & (scale == 'absolute'):
        y_limit = 75000
    elif (type_flood == pluvial_floods) & (scale == 'absolute'):
        y_limit = 280000
    elif (type_flood == fluvial_floods) & (scale == 'percent'):
        y_limit = 0.12
    elif (type_flood == pluvial_floods) & (scale == 'percent'):
        y_limit = 0.40
        
    if group == "housing_types":
        stats_per_housing_type_2011 = compute_stats_per_housing_type(type_flood, path_flood_data, simulation_households_housing_type[0, 0, :], simulation_households_housing_type[0, 3, :], simulation_households_housing_type[0, 2, :], simulation_households_housing_type[0, 1, :], "X", "X", 0.01)
        if year == "2011_2040":
            stats_per_housing_type_2040 = compute_stats_per_housing_type(type_flood, path_flood_data, simulation_households_housing_type[29, 0, :], simulation_households_housing_type[29, 3, :], simulation_households_housing_type[29, 2, :], simulation_households_housing_type[29, 1, :], "X", "X", 0.01)
            
            if scale == 'percent':
                stats_per_housing_type_2040.fraction_formal_in_flood_prone_area = stats_per_housing_type_2040.fraction_formal_in_flood_prone_area / np.nansum(simulation_households_housing_type[29, 0, :])
                stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area = stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area / np.nansum(simulation_households_housing_type[29, 1, :])
                stats_per_housing_type_2040.fraction_informal_in_flood_prone_area = stats_per_housing_type_2040.fraction_informal_in_flood_prone_area / np.nansum(simulation_households_housing_type[29, 2, :])
                stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area = stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area / np.nansum(simulation_households_housing_type[29, 3, :])

        if scale == 'percent':
            stats_per_housing_type_2011.fraction_formal_in_flood_prone_area = stats_per_housing_type_2011.fraction_formal_in_flood_prone_area / np.nansum(simulation_households_housing_type[0, 0, :])
            stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area = stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area / np.nansum(simulation_households_housing_type[0, 1, :])
            stats_per_housing_type_2011.fraction_informal_in_flood_prone_area = stats_per_housing_type_2011.fraction_informal_in_flood_prone_area / np.nansum(simulation_households_housing_type[0, 2, :])
            stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area = stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area / np.nansum(simulation_households_housing_type[0, 3, :])

        plt.rcParams.update({'font.size': 21})
        label = ["Formal \n private", "Formal \n subsidized", "Informal \n in backyards", "Informal \n settlements"]
        stats_2011_1 = [stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[2], stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[2], stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[2], stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[2]]
        stats_2011_2 = [stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[3], stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[3], stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[3], stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[3]]
        stats_2011_3 = [stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[5]]
        if year == "2011_2040":
            stats_2040_1 = [stats_per_housing_type_2040.fraction_formal_in_flood_prone_area[2], stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area[2], stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area[2], stats_per_housing_type_2040.fraction_informal_in_flood_prone_area[2]]
            stats_2040_2 = [stats_per_housing_type_2040.fraction_formal_in_flood_prone_area[3], stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area[3], stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area[3], stats_per_housing_type_2040.fraction_informal_in_flood_prone_area[3]]
            stats_2040_3 = [stats_per_housing_type_2040.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area[5], stats_per_housing_type_2040.fraction_informal_in_flood_prone_area[5]]        
        colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
        r = np.arange(4)
        if year == "2011_2040":
            barWidth = 0.25
        elif year == "2011":
            barWidth = 0.5
        plt.figure(figsize=(14,7))
        plt.bar(r, stats_2011_1, color=colors[0], edgecolor='white', label="20 years", width=barWidth)
        plt.bar(r, np.array(stats_2011_2) - np.array(stats_2011_1), bottom=np.array(stats_2011_1), color=colors[1], edgecolor='white', label='50 years', width=barWidth)
        plt.bar(r, np.array(stats_2011_3) - (np.array(stats_2011_2)), bottom=(np.array(stats_2011_2)), color=colors[2], edgecolor='white', label='100 years', width=barWidth)
        if year == "2011_2040":
            plt.bar(r + 0.25, np.array(stats_2040_1), color=colors[0], edgecolor='white', width=barWidth)
            plt.bar(r + 0.25, np.array(stats_2040_2) - np.array(stats_2040_1), bottom=np.array(stats_2040_1), color=colors[1], edgecolor='white', width=barWidth)
            plt.bar(r + 0.25, np.array(stats_2040_3) - np.array(stats_2040_2), bottom=np.array(stats_2040_2), color=colors[2], edgecolor='white', width=barWidth)
            plt.text(r[0] - 0.125, stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[5], "2011",fontsize =  15)
            plt.text(r[1] - 0.125, stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[5], "2011",fontsize =  15) 
            plt.text(r[3] - 0.125, stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[5], "2011",fontsize =  15) 
            plt.text(r[2] - 0.125, stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[5], "2011",fontsize =  15)
            plt.text(r[0] + 0.125, stats_per_housing_type_2040.fraction_formal_in_flood_prone_area[5], '2040',fontsize =  15)
            plt.text(r[1] + 0.125, stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area[5], '2040',fontsize =  15) 
            plt.text(r[3] + 0.125, stats_per_housing_type_2040.fraction_informal_in_flood_prone_area[5], '2040',fontsize =  15) 
            plt.text(r[2] + 0.125, stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area[5], '2040',fontsize =  15) 
        plt.legend()
        plt.xticks(r, label)
        if year == "2011_2040":
            plt.ylim(0, y_limit)
        plt.tick_params(labelbottom=True)
        if scale == "absolute":
            plt.ylabel("Dwellings in flood-prone areas")
        elif scale == "percent":
            plt.ylabel("Dwellings in flood-prone areas (%)")
            
    elif group == "income_class":
        
        if type_flood == fluvial_floods:
            data_flood_100 = np.squeeze(pd.read_excel(path_flood_data + 'FD_100yr' + ".xlsx"))
            data_flood_50 = np.squeeze(pd.read_excel(path_flood_data + 'FD_50yr' + ".xlsx"))
            data_flood_20 = np.squeeze(pd.read_excel(path_flood_data + 'FD_20yr' + ".xlsx"))
        elif type_flood == pluvial_floods:
            data_flood_100 = np.squeeze(pd.read_excel(path_flood_data + 'P_100yr' + ".xlsx"))
            data_flood_50 = np.squeeze(pd.read_excel(path_flood_data + 'P_50yr' + ".xlsx"))
            data_flood_20 = np.squeeze(pd.read_excel(path_flood_data + 'P_20yr' + ".xlsx"))
            
        data_flood_100["formal_pop_flood_prone"] = simulation_households_housing_type[0, 0, :] * data_flood_100["prop_flood_prone"]
        data_flood_100["backyard_pop_flood_prone"] = simulation_households_housing_type[0, 1, :] * data_flood_100["prop_flood_prone"]
        data_flood_100["informal_pop_flood_prone"] = simulation_households_housing_type[0, 2, :] * data_flood_100["prop_flood_prone"]
        data_flood_100["subsidized_pop_flood_prone"] = simulation_households_housing_type[0, 3, :] * data_flood_100["prop_flood_prone"]
    
        data_flood_50["formal_pop_flood_prone"] = simulation_households_housing_type[0, 0, :] * data_flood_50["prop_flood_prone"]
        data_flood_50["backyard_pop_flood_prone"] = simulation_households_housing_type[0, 1, :] * data_flood_50["prop_flood_prone"]
        data_flood_50["informal_pop_flood_prone"] = simulation_households_housing_type[0, 2, :] * data_flood_50["prop_flood_prone"]
        data_flood_50["subsidized_pop_flood_prone"] = simulation_households_housing_type[0, 3, :] * data_flood_50["prop_flood_prone"]
    
        data_flood_20["formal_pop_flood_prone"] = simulation_households_housing_type[0, 0, :] * data_flood_20["prop_flood_prone"]
        data_flood_20["backyard_pop_flood_prone"] = simulation_households_housing_type[0, 1, :] * data_flood_20["prop_flood_prone"]
        data_flood_20["informal_pop_flood_prone"] = simulation_households_housing_type[0, 2, :] * data_flood_20["prop_flood_prone"]
        data_flood_20["subsidized_pop_flood_prone"] = simulation_households_housing_type[0, 3, :] * data_flood_20["prop_flood_prone"]
    
        if type_flood == pluvial_floods:
            data_flood_20["formal_pop_flood_prone"] = 0
    
        ## Reshape
        formal_2011_100 = data_flood_100.iloc[:, [0, 2]]
        backyard_2011_100 = data_flood_100.iloc[:, [0, 3]]
        informal_2011_100 = data_flood_100.iloc[:, [0, 4]]
        subsidized_2011_100 = data_flood_100.iloc[:, [0, 5]]
        
        formal_2011_50 = data_flood_50.iloc[:, [0, 2]]
        backyard_2011_50 = data_flood_50.iloc[:, [0, 3]]
        informal_2011_50 = data_flood_50.iloc[:, [0, 4]]
        subsidized_2011_50 = data_flood_50.iloc[:, [0, 5]]
        
        formal_2011_20 = data_flood_20.iloc[:, [0, 2]]
        backyard_2011_20 = data_flood_20.iloc[:, [0, 3]]
        informal_2011_20 = data_flood_20.iloc[:, [0, 4]]
        subsidized_2011_20 = data_flood_20.iloc[:, [0, 5]]
    
        income_class_2011 = np.argmax(simulation_households[0, :, :, :], 1)    
    
        def split_by_class(data, income_class_location):
            data_class_1 = data[income_class_location == 0]
            data_class_2 = data[income_class_location == 1]
            data_class_3 = data[income_class_location == 2]
            data_class_4 = data[income_class_location == 3]
            return data_class_1, data_class_2, data_class_3, data_class_4
            
        formal_2011_100_class1, formal_2011_100_class2, formal_2011_100_class3, formal_2011_100_class4 = split_by_class(formal_2011_100, income_class_2011[0, :])
        formal_2011_50_class1, formal_2011_50_class2, formal_2011_50_class3, formal_2011_50_class4 = split_by_class(formal_2011_50, income_class_2011[0, :])
        formal_2011_20_class1, formal_2011_20_class2, formal_2011_20_class3, formal_2011_20_class4 = split_by_class(formal_2011_20, income_class_2011[0, :])
        
        subsidized_2011_100_class1, subsidized_2011_100_class2, subsidized_2011_100_class3, subsidized_2011_100_class4 = split_by_class(subsidized_2011_100, income_class_2011[3, :])
        subsidized_2011_50_class1, subsidized_2011_50_class2, subsidized_2011_50_class3, subsidized_2011_50_class4 = split_by_class(subsidized_2011_50, income_class_2011[3, :])
        subsidized_2011_20_class1, subsidized_2011_20_class2, subsidized_2011_20_class3, subsidized_2011_20_class4 = split_by_class(subsidized_2011_20, income_class_2011[3, :])
       
        backyard_2011_100_class1, backyard_2011_100_class2, backyard_2011_100_class3, backyard_2011_100_class4 = split_by_class(backyard_2011_100, income_class_2011[1, :])
        backyard_2011_50_class1, backyard_2011_50_class2, backyard_2011_50_class3, backyard_2011_50_class4 = split_by_class(backyard_2011_50, income_class_2011[1, :])
        backyard_2011_20_class1, backyard_2011_20_class2, backyard_2011_20_class3, backyard_2011_20_class4 = split_by_class(backyard_2011_20, income_class_2011[1, :])
       
        informal_2011_100_class1, informal_2011_100_class2, informal_2011_100_class3, informal_2011_100_class4 = split_by_class(informal_2011_100, income_class_2011[2, :])
        informal_2011_50_class1, informal_2011_50_class2, informal_2011_50_class3, informal_2011_50_class4 = split_by_class(informal_2011_50, income_class_2011[2, :])
        informal_2011_20_class1, informal_2011_20_class2, informal_2011_20_class3, informal_2011_20_class4 = split_by_class(informal_2011_20, income_class_2011[2, :])
        
        #Total
        array_2011_100 = np.array([sum(formal_2011_100_class1.iloc[:,1]) + sum(subsidized_2011_100_class1.iloc[:,1]) + sum(backyard_2011_100_class1.iloc[:,1]) + sum(informal_2011_100_class1.iloc[:,1]), sum(formal_2011_100_class2.iloc[:,1]) + sum(subsidized_2011_100_class2.iloc[:,1]) + sum(backyard_2011_100_class2.iloc[:,1]) + sum(informal_2011_100_class2.iloc[:,1]), sum(formal_2011_100_class3.iloc[:,1]) + sum(subsidized_2011_100_class3.iloc[:,1]) + sum(backyard_2011_100_class3.iloc[:,1]) + sum(informal_2011_100_class3.iloc[:,1]), sum(formal_2011_100_class4.iloc[:,1]) + sum(subsidized_2011_100_class4.iloc[:,1]) + sum(backyard_2011_100_class4.iloc[:,1]) + sum(informal_2011_100_class4.iloc[:,1])])
        array_2011_50 = np.array([sum(formal_2011_50_class1.iloc[:,1]) + sum(subsidized_2011_50_class1.iloc[:,1]) + sum(backyard_2011_50_class1.iloc[:,1]) + sum(informal_2011_50_class1.iloc[:,1]), sum(formal_2011_50_class2.iloc[:,1]) + sum(subsidized_2011_50_class2.iloc[:,1]) + sum(backyard_2011_50_class2.iloc[:,1]) + sum(informal_2011_50_class2.iloc[:,1]), sum(formal_2011_50_class3.iloc[:,1]) + sum(subsidized_2011_50_class3.iloc[:,1]) + sum(backyard_2011_50_class3.iloc[:,1]) + sum(informal_2011_50_class3.iloc[:,1]), sum(formal_2011_50_class4.iloc[:,1]) + sum(subsidized_2011_50_class4.iloc[:,1]) + sum(backyard_2011_50_class4.iloc[:,1]) + sum(informal_2011_50_class4.iloc[:,1])])
        array_2011_20 = np.array([sum(formal_2011_20_class1.iloc[:,1]) + sum(subsidized_2011_20_class1.iloc[:,1]) + sum(backyard_2011_20_class1.iloc[:,1]) + sum(informal_2011_20_class1.iloc[:,1]), sum(formal_2011_20_class2.iloc[:,1]) + sum(subsidized_2011_20_class2.iloc[:,1]) + sum(backyard_2011_20_class2.iloc[:,1]) + sum(informal_2011_20_class2.iloc[:,1]), sum(formal_2011_20_class3.iloc[:,1]) + sum(subsidized_2011_20_class3.iloc[:,1]) + sum(backyard_2011_20_class3.iloc[:,1]) + sum(informal_2011_20_class3.iloc[:,1]), sum(formal_2011_20_class4.iloc[:,1]) + sum(subsidized_2011_20_class4.iloc[:,1]) + sum(backyard_2011_20_class4.iloc[:,1]) + sum(informal_2011_20_class4.iloc[:,1])])
        
        if year == '2011_2040':
            if type_flood == fluvial_floods:
                data_flood_100_2040 = np.squeeze(pd.read_excel(path_flood_data + 'FD_100yr' + ".xlsx"))
                data_flood_50_2040 = np.squeeze(pd.read_excel(path_flood_data + 'FD_50yr' + ".xlsx"))
                data_flood_20_2040 = np.squeeze(pd.read_excel(path_flood_data + 'FD_20yr' + ".xlsx"))
            elif type_flood == pluvial_floods:
                data_flood_100_2040 = np.squeeze(pd.read_excel(path_flood_data + 'P_100yr' + ".xlsx"))
                data_flood_50_2040 = np.squeeze(pd.read_excel(path_flood_data + 'P_50yr' + ".xlsx"))
                data_flood_20_2040 = np.squeeze(pd.read_excel(path_flood_data + 'P_20yr' + ".xlsx"))
        
            
            data_flood_100_2040["formal_pop_flood_prone"] = simulation_households_housing_type[29, 0, :] * data_flood_100["prop_flood_prone"]
            data_flood_100_2040["backyard_pop_flood_prone"] = simulation_households_housing_type[29, 1, :] * data_flood_100["prop_flood_prone"]
            data_flood_100_2040["informal_pop_flood_prone"] = simulation_households_housing_type[29, 2, :] * data_flood_100["prop_flood_prone"]
            data_flood_100_2040["subsidized_pop_flood_prone"] = simulation_households_housing_type[29, 3, :] * data_flood_100["prop_flood_prone"]
            
            data_flood_50_2040["formal_pop_flood_prone"] = simulation_households_housing_type[29, 0, :] * data_flood_50["prop_flood_prone"]
            data_flood_50_2040["backyard_pop_flood_prone"] = simulation_households_housing_type[29, 1, :] * data_flood_50["prop_flood_prone"]
            data_flood_50_2040["informal_pop_flood_prone"] = simulation_households_housing_type[29, 2, :] * data_flood_50["prop_flood_prone"]
            data_flood_50_2040["subsidized_pop_flood_prone"] = simulation_households_housing_type[29, 3, :] * data_flood_50["prop_flood_prone"]
            
            data_flood_20_2040["formal_pop_flood_prone"] = simulation_households_housing_type[29, 0, :] * data_flood_20["prop_flood_prone"]
            data_flood_20_2040["backyard_pop_flood_prone"] = simulation_households_housing_type[29, 1, :] * data_flood_20["prop_flood_prone"]
            data_flood_20_2040["informal_pop_flood_prone"] = simulation_households_housing_type[29, 2, :] * data_flood_20["prop_flood_prone"]
            data_flood_20_2040["subsidized_pop_flood_prone"] = simulation_households_housing_type[29, 3, :] * data_flood_20["prop_flood_prone"]
    
            if type_flood == pluvial_floods:
                data_flood_20_2040["formal_pop_flood_prone"] = 0
    
            ## Reshape
            formal_2040_100 = data_flood_100_2040.iloc[:, [0, 2]]
            backyard_2040_100 = data_flood_100_2040.iloc[:, [0, 3]]
            informal_2040_100 = data_flood_100_2040.iloc[:, [0, 4]]
            subsidized_2040_100 = data_flood_100_2040.iloc[:, [0, 5]]
            
            formal_2040_50 = data_flood_50_2040.iloc[:, [0, 2]]
            backyard_2040_50 = data_flood_50_2040.iloc[:, [0, 3]]
            informal_2040_50 = data_flood_50_2040.iloc[:, [0, 4]]
            subsidized_2040_50 = data_flood_50_2040.iloc[:, [0, 5]]
        
            formal_2040_20 = data_flood_20_2040.iloc[:, [0, 2]]
            backyard_2040_20 = data_flood_20_2040.iloc[:, [0, 3]]
            informal_2040_20 = data_flood_20_2040.iloc[:, [0, 4]]
            subsidized_2040_20 = data_flood_20_2040.iloc[:, [0, 5]]
    
            income_class_2040 = np.argmax(simulation_households[29, :, :, :], 1)    
    
            def split_by_class(data, income_class_location):
                data_class_1 = data[income_class_location == 0]
                data_class_2 = data[income_class_location == 1]
                data_class_3 = data[income_class_location == 2]
                data_class_4 = data[income_class_location == 3]
                return data_class_1, data_class_2, data_class_3, data_class_4
            
            formal_2040_100_class1, formal_2040_100_class2, formal_2040_100_class3, formal_2040_100_class4 = split_by_class(formal_2040_100, income_class_2040[0, :])
            formal_2040_50_class1, formal_2040_50_class2, formal_2040_50_class3, formal_2040_50_class4 = split_by_class(formal_2040_50, income_class_2040[0, :])
            formal_2040_20_class1, formal_2040_20_class2, formal_2040_20_class3, formal_2040_20_class4 = split_by_class(formal_2040_20, income_class_2040[0, :])
            
            subsidized_2040_100_class1, subsidized_2040_100_class2, subsidized_2040_100_class3, subsidized_2040_100_class4 = split_by_class(subsidized_2040_100, income_class_2040[3, :])
            subsidized_2040_50_class1, subsidized_2040_50_class2, subsidized_2040_50_class3, subsidized_2040_50_class4 = split_by_class(subsidized_2040_50, income_class_2040[3, :])
            subsidized_2040_20_class1, subsidized_2040_20_class2, subsidized_2040_20_class3, subsidized_2040_20_class4 = split_by_class(subsidized_2040_20, income_class_2040[3, :])
            
            backyard_2040_100_class1, backyard_2040_100_class2, backyard_2040_100_class3, backyard_2040_100_class4 = split_by_class(backyard_2040_100, income_class_2040[1, :])
            backyard_2040_50_class1, backyard_2040_50_class2, backyard_2040_50_class3, backyard_2040_50_class4 = split_by_class(backyard_2040_50, income_class_2040[1, :])
            backyard_2040_20_class1, backyard_2040_20_class2, backyard_2040_20_class3, backyard_2040_20_class4 = split_by_class(backyard_2040_20, income_class_2040[1, :])
       
            informal_2040_100_class1, informal_2040_100_class2, informal_2040_100_class3, informal_2040_100_class4 = split_by_class(informal_2040_100, income_class_2040[2, :])
            informal_2040_50_class1, informal_2040_50_class2, informal_2040_50_class3, informal_2040_50_class4 = split_by_class(informal_2040_50, income_class_2040[2, :])
            informal_2040_20_class1, informal_2040_20_class2, informal_2040_20_class3, informal_2040_20_class4 = split_by_class(informal_2040_20, income_class_2040[2, :])
        
            #Total
            array_2040_100 = np.array([sum(formal_2040_100_class1.iloc[:,1]) + sum(subsidized_2040_100_class1.iloc[:,1]) + sum(backyard_2040_100_class1.iloc[:,1]) + sum(informal_2040_100_class1.iloc[:,1]), sum(formal_2040_100_class2.iloc[:,1]) + sum(subsidized_2040_100_class2.iloc[:,1]) + sum(backyard_2040_100_class2.iloc[:,1]) + sum(informal_2040_100_class2.iloc[:,1]), sum(formal_2040_100_class3.iloc[:,1]) + sum(subsidized_2040_100_class3.iloc[:,1]) + sum(backyard_2040_100_class3.iloc[:,1]) + sum(informal_2040_100_class3.iloc[:,1]), sum(formal_2040_100_class4.iloc[:,1]) + sum(subsidized_2040_100_class4.iloc[:,1]) + sum(backyard_2040_100_class4.iloc[:,1]) + sum(informal_2040_100_class4.iloc[:,1])])
            array_2040_50 = np.array([sum(formal_2040_50_class1.iloc[:,1]) + sum(subsidized_2040_50_class1.iloc[:,1]) + sum(backyard_2040_50_class1.iloc[:,1]) + sum(informal_2040_50_class1.iloc[:,1]), sum(formal_2040_50_class2.iloc[:,1]) + sum(subsidized_2040_50_class2.iloc[:,1]) + sum(backyard_2040_50_class2.iloc[:,1]) + sum(informal_2040_50_class2.iloc[:,1]), sum(formal_2040_50_class3.iloc[:,1]) + sum(subsidized_2040_50_class3.iloc[:,1]) + sum(backyard_2040_50_class3.iloc[:,1]) + sum(informal_2040_50_class3.iloc[:,1]), sum(formal_2040_50_class4.iloc[:,1]) + sum(subsidized_2040_50_class4.iloc[:,1]) + sum(backyard_2040_50_class4.iloc[:,1]) + sum(informal_2040_50_class4.iloc[:,1])])
            array_2040_20 = np.array([sum(formal_2040_20_class1.iloc[:,1]) + sum(subsidized_2040_20_class1.iloc[:,1]) + sum(backyard_2040_20_class1.iloc[:,1]) + sum(informal_2040_20_class1.iloc[:,1]), sum(formal_2040_20_class2.iloc[:,1]) + sum(subsidized_2040_20_class2.iloc[:,1]) + sum(backyard_2040_20_class2.iloc[:,1]) + sum(informal_2040_20_class2.iloc[:,1]), sum(formal_2040_20_class3.iloc[:,1]) + sum(subsidized_2040_20_class3.iloc[:,1]) + sum(backyard_2040_20_class3.iloc[:,1]) + sum(informal_2040_20_class3.iloc[:,1]), sum(formal_2040_20_class4.iloc[:,1]) + sum(subsidized_2040_20_class4.iloc[:,1]) + sum(backyard_2040_20_class4.iloc[:,1]) + sum(informal_2040_20_class4.iloc[:,1])])
        
        if scale == 'percent':
            array_2011_100 = array_2011_100 / np.array(np.nansum(np.nansum(simulation_households[0,:,:,:], 2), 0))
            array_2011_50 = array_2011_50 / np.array(np.nansum(np.nansum(simulation_households[0,:,:,:], 2), 0))
            array_2011_20 = array_2011_20 / np.array(np.nansum(np.nansum(simulation_households[0,:,:,:], 2), 0))
            if year == '2011_2040':
                array_2040_100 = array_2040_100 / np.array(np.nansum(np.nansum(simulation_households[29,:,:,:], 2), 0))
                array_2040_50 = array_2040_50 / np.array(np.nansum(np.nansum(simulation_households[29,:,:,:], 2), 0))
                array_2040_20 = array_2040_20 / np.array(np.nansum(np.nansum(simulation_households[29,:,:,:], 2), 0))
           
                 
        plt.rcParams.update({'font.size': 21})
        label = ["Class 1", "Class 2", "Class 3", "Class 4"]
        colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
        r = np.arange(4)
        if year == "2011_2040":
            barWidth = 0.25
        elif year == "2011":
            barWidth = 0.5
        plt.figure(figsize=(14,7))
        plt.bar(r, array_2011_20, color=colors[0], edgecolor='white', label="20 years", width=barWidth)
        plt.bar(r, np.array(array_2011_50) - np.array(array_2011_20), bottom=np.array(array_2011_20), color=colors[1], edgecolor='white', label='50 years', width=barWidth)
        plt.bar(r, np.array(array_2011_100) - (np.array(array_2011_50)), bottom=(np.array(array_2011_50)), color=colors[2], edgecolor='white', label='100 years', width=barWidth)
        if year == "2011_2040":
            plt.bar(r + 0.25, np.array(array_2040_20), color=colors[0], edgecolor='white', width=barWidth)
            plt.bar(r + 0.25, np.array(array_2040_50) - np.array(array_2040_20), bottom=np.array(array_2040_20), color=colors[1], edgecolor='white', width=barWidth)
            plt.bar(r + 0.25, np.array(array_2040_100) - np.array(array_2040_50), bottom=np.array(array_2040_50), color=colors[2], edgecolor='white', width=barWidth)
            plt.text(r[0] - 0.125, array_2011_100[0], "2011",fontsize =  15)
            plt.text(r[1] - 0.125, array_2011_100[1], "2011",fontsize =  15) 
            plt.text(r[2] - 0.125, array_2011_100[2], "2011",fontsize =  15) 
            plt.text(r[3] - 0.125, array_2011_100[3], "2011",fontsize =  15)
            plt.text(r[0] + 0.125, array_2040_100[0], '2040',fontsize =  15)
            plt.text(r[1] + 0.125, array_2040_100[1], '2040',fontsize =  15) 
            plt.text(r[2] + 0.125, array_2040_100[2], '2040',fontsize =  15) 
            plt.text(r[3] + 0.125, array_2040_100[3], '2040',fontsize =  15) 
        plt.legend()
        plt.xticks(r, label)
        if year == "2011_2040":
            plt.ylim(0, y_limit)
        plt.tick_params(labelbottom=True)
        if scale == "absolute":
            plt.ylabel("Dwellings in flood-prone areas")
        elif scale == "percent":
            plt.ylabel("Dwellings in flood-prone areas (%)")
        
            
            
            
    if type_flood[0] == 'FD_5yr':
        name_flood = "fluvial"
    elif type_flood[0] == "P_5yr":
        name_flood = 'pluvial'
    plt.savefig(path_outputs + 'flood_exposure_by_' + group + '_' + year + '_' + scale + '_' + name_flood + '_' + scenario + '.png')

    if group == 'housing_types':
        df = pd.DataFrame()
        df["flood_exposure_2011"] = [stats_per_housing_type_2011.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_backyard_in_flood_prone_area[5], stats_per_housing_type_2011.fraction_informal_in_flood_prone_area[5]]
        if year == "2011_2040":
            df["flood_exposure_2040"] = [stats_per_housing_type_2040.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_2040.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_2040.fraction_backyard_in_flood_prone_area[5], stats_per_housing_type_2040.fraction_informal_in_flood_prone_area[5]]   
        df.index = ['FP', 'FS', 'IB', 'IS']
        df.to_excel(path_outputs + 'flood_exposure_by_' + group + '_' + year + '_' + scale + '_' + name_flood + '_' + scenario + '_100yr.xlsx')
    elif group == "income_class":
        df = pd.DataFrame()
        df["flood_exposure_2011"] = array_2011_100
        if year == "2011_2040":
            df["flood_exposure_2040"] = array_2040_100
        df.index = label
        df.to_excel(path_outputs + 'flood_exposure_by_' + group + '_' + year + '_' + scale + '_' + name_flood + '_' + scenario + '_100yr.xlsx')
   
## HOUSING TYPES

#2011
plot_flood_exposure('2011', 'housing_types', 'percent', fluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011', 'housing_types', 'absolute', fluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011', 'housing_types', 'percent', pluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011', 'housing_types', 'absolute', pluvial_floods, 'sc1', path_outputs)

#2040 - SC1
plot_flood_exposure('2011_2040', 'housing_types', 'percent', fluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011_2040', 'housing_types', 'absolute', fluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011_2040', 'housing_types', 'percent', pluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011_2040', 'housing_types', 'absolute', pluvial_floods, 'sc1', path_outputs)

#2040 - SC2
plot_flood_exposure('2011_2040', 'housing_types', 'percent', fluvial_floods, 'sc2', path_outputs)
plot_flood_exposure('2011_2040', 'housing_types', 'absolute', fluvial_floods, 'sc2', path_outputs)
plot_flood_exposure('2011_2040', 'housing_types', 'percent', pluvial_floods, 'sc2', path_outputs)
plot_flood_exposure('2011_2040', 'housing_types', 'absolute', pluvial_floods, 'sc2', path_outputs)

## INCOME CLASS

#2011
plot_flood_exposure('2011', 'income_class', 'percent', fluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011', 'income_class', 'absolute', fluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011', 'income_class', 'percent', pluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011', 'income_class', 'absolute', pluvial_floods, 'sc1', path_outputs)

#2040 - SC1
plot_flood_exposure('2011_2040', 'income_class', 'percent', fluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011_2040', 'income_class', 'absolute', fluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011_2040', 'income_class', 'percent', pluvial_floods, 'sc1', path_outputs)
plot_flood_exposure('2011_2040', 'income_class', 'absolute', pluvial_floods, 'sc1', path_outputs)

#2040 - SC2
plot_flood_exposure('2011_2040', 'income_class', 'percent', fluvial_floods, 'sc2', path_outputs)
plot_flood_exposure('2011_2040', 'income_class', 'absolute', fluvial_floods, 'sc2', path_outputs)
plot_flood_exposure('2011_2040', 'income_class', 'percent', pluvial_floods, 'sc2', path_outputs)
plot_flood_exposure('2011_2040', 'income_class', 'absolute', pluvial_floods, 'sc2', path_outputs)

### SEVERITY OF FLOODS

#Options: by income class or by housing type, type flood

def plot_flood_severity(group, type_flood, path_outputs):
    
    simulation_households_housing_type = simulation_households_housing_type_sc2
    simulation_households = simulation_households_sc2
    data_flood = np.squeeze(pd.read_excel(path_flood_data + type_flood + ".xlsx"))

    data_flood["formal_pop_flood_prone"] = simulation_households_housing_type[0, 0, :] * data_flood["prop_flood_prone"]
    data_flood["backyard_pop_flood_prone"] = simulation_households_housing_type[0, 1, :] * data_flood["prop_flood_prone"]
    data_flood["informal_pop_flood_prone"] = simulation_households_housing_type[0, 2, :] * data_flood["prop_flood_prone"]
    data_flood["subsidized_pop_flood_prone"] = simulation_households_housing_type[0, 3, :] * data_flood["prop_flood_prone"]
    
    if type_flood == "P_20yr":
        data_flood["formal_pop_flood_prone"] = 0
    elif ((type_flood == "P_5yr") |(type_flood == "P_10yr")):
        data_flood["formal_pop_flood_prone"] = 0
        data_flood["backyard_pop_flood_prone"] = 0
        data_flood["subsidized_pop_flood_prone"] = 0
    
    ## Reshape
    formal_2011 = data_flood.iloc[:, [0, 2]]
    backyard_2011 = data_flood.iloc[:, [0, 3]]
    informal_2011 = data_flood.iloc[:, [0, 4]]
    subsidized_2011 = data_flood.iloc[:, [0, 5]]
    
    if type_flood == "FD_100yr":
        y_limit_class = 13000
        y_limit_housing = 9000
    elif type_flood == "P_100yr":
        y_limit_housing = 65000
        y_limit_class = 50000
    
    if group == "housing_types":
        plt.figure(figsize=(12,7))
        sns.distplot(formal_2011.iloc[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': formal_2011.iloc[:,1]}, color = 'black', label = "2011", kde = False)
        plt.ylim(0, y_limit_housing)
        plt.xlabel("Severity of floods (m)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'severity_floods_by_' + group + '_' + type_flood + '_FP')
        plt.close()
    
        plt.figure(figsize=(12,7))
        sns.distplot(backyard_2011.iloc[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': backyard_2011.iloc[:,1]}, color = 'black', label = "2011", kde = False)
        plt.ylim(0, y_limit_housing)
        plt.xlabel("Severity of floods (m)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'severity_floods_by_' + group + '_' + type_flood + '_IB')
        plt.close()
    
        plt.figure(figsize=(12,7))
        sns.distplot(informal_2011.iloc[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': informal_2011.iloc[:,1]}, color = 'black', label = "2011", kde = False)
        plt.ylim(0, y_limit_housing)
        plt.xlabel("Severity of floods (m)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'severity_floods_by_' + group + '_' + type_flood + '_IS')
        plt.close()
    
        plt.figure(figsize=(12,7))
        sns.distplot(subsidized_2011.iloc[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': subsidized_2011.iloc[:,1]}, color = 'black', label = "2011", kde = False)
        plt.ylim(0, y_limit_housing)
        plt.xlabel("Severity of floods (m)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'severity_floods_by_' + group + '_' + type_flood + '_FS')
        plt.close()
        
        df = pd.DataFrame()
        df["nb_of_persons"] = [sum(formal_2011.iloc[:,1]), sum(subsidized_2011.iloc[:,1]), sum(backyard_2011.iloc[:,1]), sum(informal_2011.iloc[:,1])]
        df["average_flood_depth"] = [np.average(formal_2011.iloc[:,0], weights = formal_2011.iloc[:,1]), np.average(subsidized_2011.iloc[:,0], weights = subsidized_2011.iloc[:,1]), np.average(backyard_2011.iloc[:,0], weights = backyard_2011.iloc[:,1]), np.average(informal_2011.iloc[:,0], weights = informal_2011.iloc[:,1])]
        df.index = ['FP', 'FS', 'IB', 'IS']
        df.to_excel(path_outputs + 'severity_floods_by_' + group + '_' + type_flood + '.xlsx')
       

    if group == "income_class":
    
        income_class_2011 = np.argmax(simulation_households[0, :, :, :], 1)    
    
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

        plt.figure(figsize=(12,7))
        sns.distplot(array_2011_class1[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': array_2011_class1[:,1]}, color = 'black', label = "2011", kde = False)
        plt.ylim(0, y_limit_class)
        plt.xlabel("Severity of floods (m)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'severity_floods_by_' + group + '_' + type_flood + '_class1.png')
        plt.close()
    
        plt.figure(figsize=(12,7))
        sns.distplot(array_2011_class2[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': array_2011_class2[:,1]}, color = 'black', label = "2011", kde = False)
        plt.ylim(0, y_limit_class)
        plt.xlabel("Severity of floods (m)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'severity_floods_by_' + group + '_' + type_flood + '_class2')
        plt.close()
    
        plt.figure(figsize=(12,7))
        sns.distplot(array_2011_class3[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': array_2011_class3[:,1]}, color = 'black', label = "2011", kde = False)
        plt.ylim(0, y_limit_class)
        plt.xlabel("Severity of floods (m)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'severity_floods_by_' + group + '_' + type_flood + '_class3')
        plt.close()
        
        plt.figure(figsize=(12,7))
        sns.distplot(array_2011_class4[:,0], bins=np.arange(0,3.5,0.05), hist_kws={'weights': array_2011_class4[:,1]}, color = 'black', label = "2011", kde = False)
        plt.ylim(0, y_limit_class)
        plt.xlabel("Severity of floods (m)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'severity_floods_by_' + group + '_' + type_flood + '_class4')
        plt.close()
        
        df = pd.DataFrame()
        df["nb_of_persons"] = [sum(array_2011_class1[:,1]), sum(array_2011_class2[:,1]), sum(array_2011_class3[:,1]), sum(array_2011_class4[:,1])]
        df["average_flood_depth"] = [np.average(array_2011_class1[:,0], weights = array_2011_class1[:,1]), np.average(array_2011_class2[:,0], weights = array_2011_class2[:,1]), np.average(array_2011_class3[:,0], weights = array_2011_class3[:,1]), np.average(array_2011_class4[:,0], weights = array_2011_class4[:,1])]
        df.index = ['Income class 1', 'Income class 2', 'Income class 3', 'Income class 4']
        df.to_excel(path_outputs + 'severity_floods_by_' + group + '_' + type_flood + '.xlsx')
        
plot_flood_severity('income_class', 'FD_100yr', path_outputs)
plot_flood_severity('housing_types', 'FD_100yr', path_outputs)
plot_flood_severity('income_class', 'P_100yr', path_outputs)
plot_flood_severity('housing_types', 'P_100yr', path_outputs)

### FLOOD DAMAGES 

#options: fluvial or pluvial, sc1 or sc2, 100yr or annualized (+ table)
#option: detail by income class

def plot_flood_damages(floods, scenario, path_outputs, type_flood, detail_by_income_class):
    
    if scenario == 'sc1':
        simulation_dwelling_size = simulation_dwelling_size_sc1
        simulation_rent = simulation_rent_sc1
        simulation_households_housing_type = simulation_households_housing_type_sc1
        simulation_households_center = households_center_sc1
        simulation_households = simulation_households_sc1
    elif scenario == 'sc2':
        simulation_dwelling_size = simulation_dwelling_size_sc2
        simulation_rent = simulation_rent_sc2
        simulation_households_housing_type = simulation_households_housing_type_sc2
        simulation_households_center = households_center_sc2
        simulation_households = simulation_households_sc2


    spline_agricultural_rent, spline_interest_rate, spline_RDP, spline_population_income_distribution, spline_inflation, spline_income_distribution, spline_population, spline_interest_rate, spline_income, spline_minimum_housing_supply, spline_fuel = import_scenarios(income_2011, param, grid, path_scenarios)
    formal_structure_cost_2011 = compute_formal_structure_cost_method2(simulation_rent[0, :, :], param, interpolate_interest_rate(spline_interest_rate, 0), import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 0), simulation_households_housing_type[0, :, :], (spline_income(0) / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"])    
    content_cost_2011 = compute_content_cost(simulation_households_center[0, :, :], np.load(precalculated_transport + "year_0.npy"), param, fraction_capital_destroyed, simulation_rent[0, :, :], simulation_dwelling_size[0, :, :], interpolate_interest_rate(spline_interest_rate, 0))
    formal_structure_cost_2040 = compute_formal_structure_cost_method2(simulation_rent[28, :, :], param, interpolate_interest_rate(spline_interest_rate, 28), import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 28), simulation_households_housing_type[28, :, :], (spline_income(28) / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"])    
    content_cost_2040 = compute_content_cost(simulation_households_center[28, :, :], np.load(precalculated_transport + "year_29.npy"), param, fraction_capital_destroyed, simulation_rent[28, :, :], simulation_dwelling_size[28, :, :], interpolate_interest_rate(spline_interest_rate, 28))

    for item in floods:
        
        param["subsidized_structure_value_ref"] = 150000
        param["informal_structure_value_ref"] = 4000
        df2011 = pd.DataFrame()
        df2040 = pd.DataFrame()
        type_of_flood = copy.deepcopy(item)
        data_flood = np.squeeze(pd.read_excel(path_flood_data + item + ".xlsx"))
    
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
        writer = pd.ExcelWriter(path_outputs + 'damages_' + str(item) + '_2011.xlsx')
        df2011.to_excel(excel_writer = writer)
        writer.save()
        writer = pd.ExcelWriter(path_outputs + 'damages_' + str(item) + '_2040.xlsx')
        
        df2040.to_excel(excel_writer = writer)
        writer.save()
   
    if (type_flood != 'annualized'):
        damages_2011 = pd.read_excel(path_outputs + '/damages_' + type_flood + '_2011.xlsx')  
        damages_2040 = pd.read_excel(path_outputs + '/damages_' + type_flood + '_2040.xlsx')  

        damages_2011 = damages_2011.loc[:, 'formal_pop_flood_prone':'backyard_damages']
        damages_2040 = damages_2040.loc[:, 'formal_pop_flood_prone':'backyard_damages']
    else:
           
        damages_5yr_2011 = pd.read_excel(path_outputs + '/damages_' + floods[0] + '_2011.xlsx')  
        damages_10yr_2011 = pd.read_excel(path_outputs + '/damages_' + floods[1] + '_2011.xlsx')  
        damages_20yr_2011 = pd.read_excel(path_outputs + '/damages_' + floods[2] + '_2011.xlsx')  
        damages_50yr_2011 = pd.read_excel(path_outputs + '/damages_' + floods[3] + '_2011.xlsx')  
        damages_75yr_2011 = pd.read_excel(path_outputs + '/damages_' + floods[4] + '_2011.xlsx')  
        damages_100yr_2011 = pd.read_excel(path_outputs + '/damages_' + floods[5] + '_2011.xlsx')  
        damages_200yr_2011 = pd.read_excel(path_outputs + '/damages_' + floods[6] + '_2011.xlsx')  
        damages_250yr_2011 = pd.read_excel(path_outputs + '/damages_' + floods[7] + '_2011.xlsx')  
        damages_500yr_2011 = pd.read_excel(path_outputs + '/damages_' + floods[8] + '_2011.xlsx')  
        damages_1000yr_2011 = pd.read_excel(path_outputs + '/damages_' + floods[9] + '_2011.xlsx')  

        damages_5yr_2040 = pd.read_excel(path_outputs + '/damages_' + floods[0] + '_2040.xlsx')  
        damages_10yr_2040 = pd.read_excel(path_outputs + '/damages_' + floods[1] + '_2040.xlsx')
        damages_20yr_2040 = pd.read_excel(path_outputs + '/damages_' + floods[2] + '_2040.xlsx')  
        damages_50yr_2040 = pd.read_excel(path_outputs + '/damages_' + floods[3] + '_2040.xlsx')  
        damages_75yr_2040 = pd.read_excel(path_outputs + '/damages_' + floods[4] + '_2040.xlsx')  
        damages_100yr_2040 = pd.read_excel(path_outputs + '/damages_' + floods[5] + '_2040.xlsx')  
        damages_200yr_2040 = pd.read_excel(path_outputs + '/damages_' + floods[6] + '_2040.xlsx')  
        damages_250yr_2040 = pd.read_excel(path_outputs + '/damages_' + floods[7] + '_2040.xlsx')  
        damages_500yr_2040 = pd.read_excel(path_outputs + '/damages_' + floods[8] + '_2040.xlsx')  
        damages_1000yr_2040 = pd.read_excel(path_outputs + '/damages_' + floods[9] + '_2040.xlsx')  

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

        damages_2011 = np.vstack([damages_5yr_2011, damages_10yr_2011, damages_20yr_2011, damages_50yr_2011, damages_75yr_2011, damages_100yr_2011, damages_200yr_2011, damages_250yr_2011, damages_500yr_2011, damages_1000yr_2011])
        damages_2040 = np.vstack([damages_5yr_2040, damages_10yr_2040, damages_20yr_2040, damages_50yr_2040, damages_75yr_2040, damages_100yr_2040, damages_200yr_2040, damages_250yr_2040, damages_500yr_2040, damages_1000yr_2040])   
        damages_2011 = pd.DataFrame(damages_2011)
        damages_2040 = pd.DataFrame(damages_2040)
        
    income_class_2011 = np.argmax(simulation_households[0, :, :, :], 1)    
    average_income_2011 = np.load(precalculated_transport + "average_income_year_0.npy")

    income_class_2040 = np.argmax(simulation_households[28, :, :, :], 1)     
    average_income_2040 = np.load(precalculated_transport + "average_income_year_28.npy")
    
    real_income_2011 = np.empty((24014, 4))
    for i in range(0, 24014):
        for j in range(0, 4):
            real_income_2011[i, j] = average_income_2011[np.array(income_class_2011)[j, i], i]
    
    real_income_2040 = np.empty((24014, 4))
    for i in range(0, 24014):
        for j in range(0, 4):
            real_income_2040[i, j] = average_income_2040[np.array(income_class_2040)[j, i], i]
    
    if type_flood == "annualized":
        real_income_2011 = np.matlib.repmat(real_income_2011, 10, 1).squeeze()
        real_income_2040 = np.matlib.repmat(real_income_2040, 10, 1).squeeze()
        
        income_class_2011 = np.matlib.repmat(income_class_2011, 1, 10).squeeze()
        income_class_2040 = np.matlib.repmat(income_class_2040, 1, 10).squeeze()

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
    
    print(floods[0])
    if floods[0] == 'FD_5yr':
        name_flood = "fluvial"
    elif floods[0] == "P_5yr":
        name_flood = 'pluvial'

    plt.figure(figsize=(13, 7))
    if type_flood != 'annualized':
        sns.distplot(subset_2011[:,1], bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2011[:,0]}, color = 'black', label = "2011")
        sns.distplot(subset_2040[:,1], bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2040[:,0]}, label = "2040")
        plt.xlim(0, 250)
    else:
        sns.distplot(subset_2011[:,1], bins=np.arange(0,2.50,0.10), hist = True, kde = False, hist_kws={'weights': subset_2011[:,0]}, color = 'black', label = "2011")
        sns.distplot(subset_2040[:,1], bins=np.arange(0,2.50,0.10), hist = True, kde = False, hist_kws={'weights': subset_2040[:,0]}, label = "2040")
    plt.legend()
    plt.xlabel("Share of the annual income destroyed (%)")
    plt.ylabel("Number of households")
    plt.savefig(path_outputs + 'flood_damages_' + type_flood + '_' + name_flood + '_' + scenario + '.png')
    plt.close()
    
    df = pd.DataFrame()
    df["average_share_income_destroyed"] = [np.average(subset_2011[:,1], weights = subset_2011[:,0]), np.average(subset_2040[:,1], weights = subset_2040[:,0])]
    df["nb_of_households"] = [np.nansum(subset_2011[:,0]), np.nansum(subset_2040[:,0])]
    df.index = ['2011', '2040']
    df.to_excel(path_outputs + 'flood_damages_' + type_flood + '_' + name_flood + '_' + scenario + '.xlsx')
    
    if detail_by_income_class == True:
        
        array_2011_class4 = np.vstack([formal_2011_class4, backyard_2011_class4, informal_2011_class4, subsidized_2011_class4])
        subset_2011_class4 = array_2011_class4[~np.isnan(array_2011_class4[:,1])]

        array_2011_class3 = np.vstack([formal_2011_class3, backyard_2011_class3, informal_2011_class3, subsidized_2011_class3])
        subset_2011_class3 = array_2011_class3[~np.isnan(array_2011_class3[:,1])]

        array_2011_class2 = np.vstack([formal_2011_class2, backyard_2011_class2, informal_2011_class2, subsidized_2011_class2])
        subset_2011_class2 = array_2011_class2[~np.isnan(array_2011_class2[:,1])]

        array_2011_class1 = np.vstack([formal_2011_class1, backyard_2011_class1, informal_2011_class1, subsidized_2011_class1])
        subset_2011_class1 = array_2011_class1[~np.isnan(array_2011_class1[:,1])]

        array_2040_class4 = np.vstack([formal_2040_class4, backyard_2040_class4, informal_2040_class4, subsidized_2040_class4])
        subset_2040_class4 = array_2040_class4[~np.isnan(array_2040_class4[:,1])]

        array_2040_class3 = np.vstack([formal_2040_class3, backyard_2040_class3, informal_2040_class3, subsidized_2040_class3])
        subset_2040_class3 = array_2040_class3[~np.isnan(array_2040_class3[:,1])]

        array_2040_class2 = np.vstack([formal_2040_class2, backyard_2040_class2, informal_2040_class2, subsidized_2040_class2])
        subset_2040_class2 = array_2040_class2[~np.isnan(array_2040_class2[:,1])]

        array_2040_class1 = np.vstack([formal_2040_class1, backyard_2040_class1, informal_2040_class1, subsidized_2040_class1])
        subset_2040_class1 = array_2040_class1[~np.isnan(array_2040_class1[:,1])]

        plt.figure(figsize = (13,7))
        if type_flood != 'annualized':
            sns.distplot(subset_2011_class1[:,1], bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2011_class1[:,0]}, color = 'black', label = "2011")
            sns.distplot(subset_2040_class1[:,1], bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2040_class1[:,0]}, label = "2040")
            plt.xlim(0, 250)
        else:
            sns.distplot(subset_2011_class1[:,1], bins=np.arange(0,2.50,0.10), hist = True, kde = False, hist_kws={'weights': subset_2011_class1[:,0]}, color = 'black', label = "2011")
            sns.distplot(subset_2040_class1[:,1], bins=np.arange(0,2.50,0.10), hist = True, kde = False, hist_kws={'weights': subset_2040_class1[:,0]}, label = "2040")
            plt.xlim(0, 2.50)
        plt.legend()
        plt.xlabel("Share of the annual income destroyed (%)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'flood_damages_' + type_flood + '_' + name_flood + '_' + scenario + '_income_class1.png')
        plt.close()
        
        plt.figure(figsize = (13,7))
        if type_flood != 'annualized':
            sns.distplot(subset_2011_class2[:,1], bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2011_class2[:,0]}, color = 'black', label = "2011")
            sns.distplot(subset_2040_class2[:,1], bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2040_class2[:,0]}, label = "2040")
            plt.xlim(0, 250)
        else:
            sns.distplot(subset_2011_class2[:,1], bins=np.arange(0,2.50,0.10), hist = True, kde = False, hist_kws={'weights': subset_2011_class2[:,0]}, color = 'black', label = "2011")
            sns.distplot(subset_2040_class2[:,1], bins=np.arange(0,2.50,0.10), hist = True, kde = False, hist_kws={'weights': subset_2040_class2[:,0]}, label = "2040")
            plt.xlim(0, 2.50)
        plt.legend()
        plt.xlabel("Share of the annual income destroyed (%)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'flood_damages_' + type_flood + '_' + name_flood + '_' + scenario + '_income_class2.png')
        plt.close()
        
        plt.figure(figsize = (13,7))
        if type_flood != 'annualized':
            sns.distplot(subset_2011_class3[:,1], bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2011_class3[:,0]}, color = 'black', label = "2011")
            sns.distplot(subset_2040_class3[:,1], bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2040_class3[:,0]}, label = "2040")
            plt.xlim(0, 250)
        else:
            sns.distplot(subset_2011_class3[:,1], bins=np.arange(0,2.50,0.10), hist = True, kde = False, hist_kws={'weights': subset_2011_class3[:,0]}, color = 'black', label = "2011")
            sns.distplot(subset_2040_class3[:,1], bins=np.arange(0,2.50,0.10), hist = True, kde = False, hist_kws={'weights': subset_2040_class3[:,0]}, label = "2040")
            plt.xlim(0, 2.50)
        plt.legend()
        plt.xlabel("Share of the annual income destroyed (%)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'flood_damages_' + type_flood + '_' + name_flood + '_' + scenario + '_income_class3.png')
        plt.close()
        
        plt.figure(figsize = (13,7))
        if type_flood != 'annualized':
            sns.distplot(subset_2011_class4[:,1], bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2011_class4[:,0]}, color = 'black', label = "2011")
            sns.distplot(subset_2040_class4[:,1], bins=np.arange(0,250,10), hist = True, kde = False, hist_kws={'weights': subset_2040_class4[:,0]}, label = "2040")
            plt.xlim(0, 250)
        else:
            sns.distplot(subset_2011_class4[:,1], bins=np.arange(0,2.50,0.10), hist = True, kde = False, hist_kws={'weights': subset_2011_class4[:,0]}, color = 'black', label = "2011")
            sns.distplot(subset_2040_class4[:,1], bins=np.arange(0,2.50,0.10), hist = True, kde = False, hist_kws={'weights': subset_2040_class4[:,0]}, label = "2040")
            plt.xlim(0, 2.50)
        plt.legend()
        plt.xlabel("Share of the annual income destroyed (%)")
        plt.ylabel("Number of households")
        plt.savefig(path_outputs + 'flood_damages_' + type_flood + '_' + name_flood + '_' + scenario + '_income_class4.png')
        plt.close()
        
        df = pd.DataFrame()
        df["average_share_income_destroyed"] = [np.average(subset_2011_class1[:,1], weights = subset_2011_class1[:,0]), np.average(subset_2040_class1[:,1], weights = subset_2040_class1[:,0]), np.average(subset_2011_class2[:,1], weights = subset_2011_class2[:,0]), np.average(subset_2040_class2[:,1], weights = subset_2040_class2[:,0]), np.average(subset_2011_class3[:,1], weights = subset_2011_class3[:,0]), np.average(subset_2040_class3[:,1], weights = subset_2040_class3[:,0]), np.average(subset_2011_class4[:,1], weights = subset_2011_class4[:,0]), np.average(subset_2040_class4[:,1], weights = subset_2040_class4[:,0])]
        df["nb_of_households"] = [np.nansum(subset_2011_class1[:,0]), np.nansum(subset_2040_class1[:,0]), np.nansum(subset_2011_class2[:,0]), np.nansum(subset_2040_class2[:,0]), np.nansum(subset_2011_class3[:,0]), np.nansum(subset_2040_class3[:,0]), np.nansum(subset_2011_class4[:,0]), np.nansum(subset_2040_class4[:,0])]
        df.index = ['class1_2011', 'class1_2040', 'class2_2011', 'class2_2040', 'class3_2011', 'class3_2040', 'class4_2011', 'class4_2040']
        df.to_excel(path_outputs + 'flood_damages_' + type_flood + '_' + name_flood + '_' + scenario + '.xlsx')
    
        
#Return period 100 year
plot_flood_damages(fluvial_floods, 'sc1', path_outputs, 'FD_100yr', True)
plot_flood_damages(pluvial_floods, 'sc1', path_outputs, 'P_100yr', True)
plot_flood_damages(fluvial_floods, 'sc2', path_outputs, 'FD_100yr', True)
plot_flood_damages(pluvial_floods, 'sc2', path_outputs, 'P_100yr', True)

#Annualized
plot_flood_damages(fluvial_floods, 'sc1', path_outputs, 'annualized', True)
plot_flood_damages(pluvial_floods, 'sc1', path_outputs, 'annualized', True)
plot_flood_damages(fluvial_floods, 'sc2', path_outputs, 'annualized', True)
plot_flood_damages(pluvial_floods, 'sc2', path_outputs, 'annualized', True)


#What remains to do:
#utility changes no dynamics