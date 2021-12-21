# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:38:21 2020

@author: Charlotte Liotta
"""

import scipy
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import copy

from inputs.data import *
from outputs.flood_outputs import *

# %% Floods 
    
def plot_damages(damages, name):
    
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    
    data = [[annualize_damages(damages.formal_structure_damages)/1000000, annualize_damages(damages.subsidized_structure_damages)/1000000, annualize_damages(damages.informal_structure_damages)/1000000, annualize_damages(damages.backyard_structure_damages)/1000000],
            [annualize_damages(damages.formal_content_damages)/1000000, annualize_damages(damages.subsidized_content_damages)/1000000, annualize_damages(damages.informal_content_damages)/1000000, annualize_damages(damages.backyard_content_damages)/1000000]]
    X = np.arange(4)
    plt.bar(X + 0.00, data[0], color = 'b', width = 0.25, label = "Structures")
    plt.bar(X + 0.25, data[1], color = 'g', width = 0.25, label = "Contents")
    plt.legend()
    plt.ylim(0, 25)
    quarter = ["Formal private", "Formal subsidized", "Informal \n settlements", "Informal \n in backyards"]
    plt.xticks(X, quarter)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year")
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages.png')  
    plt.close()
    
    data = [[annualize_damages(damages.subsidized_structure_damages)/1000000, annualize_damages(damages.informal_structure_damages)/1000000, annualize_damages(damages.backyard_structure_damages)/1000000],
            [annualize_damages(damages.subsidized_content_damages)/1000000, annualize_damages(damages.informal_content_damages)/1000000, annualize_damages(damages.backyard_content_damages)/1000000]]
    X = np.arange(3)
    plt.bar(X + 0.00, data[0], color = 'b', width = 0.25, label = "Structures")
    plt.bar(X + 0.25, data[1], color = 'g', width = 0.25, label = "Contents")
    plt.legend()
    plt.ylim(0, 4)
    quarter = ["Formal subsidized", "Informal \n settlements", "Informal \n in backyards"]
    plt.xticks(X, quarter)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year")
    plt.show()
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages_zoom.png')  
    plt.close()
     
def compare_damages(damages1, damages2, label1, label2, name):
    
    data = [[annualize_damages(damages1.formal_structure_damages), annualize_damages(damages1.formal_content_damages)],
            [annualize_damages(damages2.formal_structure_damages), annualize_damages(damages2.formal_content_damages)]]
    X = np.arange(2)
    plt.bar(X + 0.00, data[0], color = 'b', width = 0.25, label = label1)
    plt.bar(X + 0.25, data[1], color = 'g', width = 0.25, label = "Data")
    plt.legend()
    plt.ylim(0, 28000000)
    plt.title("Formal")
    plt.text(0.125, 26000000, "Structures")
    plt.text(1.125, 26000000, "Contents")
    plt.show()
    plt.savefig('C:/Users/Coupain/Desktop/cape_town/4. Sorties/' + name + '/flood_damages_formal.png')  
    plt.close()

    data = [[annualize_damages(damages1.subsidized_structure_damages), annualize_damages(damages1.subsidized_content_damages)],
            [annualize_damages(damages2.subsidized_structure_damages), annualize_damages(damages2.subsidized_content_damages), ]]
    plt.bar(X + 0.00, data[0], color = 'b', width = 0.25, label = label1)
    plt.bar(X + 0.25, data[1], color = 'g', width = 0.25, label = label2)
    plt.ylim(0, 1800000)
    plt.title("Subsidized")
    plt.text(0.125, 1600000, "Structures")
    plt.text(1.125, 1600000, "Contents")
    plt.show()
    plt.savefig('C:/Users/Coupain/Desktop/cape_town/4. Sorties/' + name + '/flood_damages_subsidized.png')  
    plt.close()

    data = [[annualize_damages(damages1.informal_structure_damages), annualize_damages(damages1.informal_content_damages)],
            [annualize_damages(damages2.informal_structure_damages), annualize_damages(damages2.informal_content_damages), ]]
    plt.bar(X + 0.00, data[0], color = 'b', width = 0.25, label = label1)
    plt.bar(X + 0.25, data[1], color = 'g', width = 0.25, label = label2)
    plt.ylim(0, 200000)
    plt.title("Informal")
    plt.text(0.125, 180000, "Structures")
    plt.text(1.125, 180000, "Contents")
    plt.show()
    plt.savefig('C:/Users/Coupain/Desktop/cape_town/4. Sorties/' + name + '/flood_damages_informal.png')  
    plt.close()
    
    data = [[annualize_damages(damages1.backyard_structure_damages), annualize_damages(damages1.backyard_content_damages)],
            [annualize_damages(damages2.backyard_structure_damages), annualize_damages(damages2.backyard_content_damages), ]]
    plt.bar(X + 0.00, data[0], color = 'b', width = 0.25, label = label1)
    plt.bar(X + 0.25, data[1], color = 'g', width = 0.25, label = label2)
    plt.ylim(0, 800000)
    plt.title("Backyard")
    plt.text(0.125, 750000, "Structures")
    plt.text(1.125, 750000, "Contents")
    plt.show()
    plt.savefig('C:/Users/Coupain/Desktop/cape_town/4. Sorties/' + name + '/flood_damages_backyard.png')  
    plt.close()

def validation_flood(name, stats1, stats2, legend1, legend2, type_flood):
   
    label = ["Formal private", "Formal subsidized", "Informal \n settlements", "Informal \n in backyards"]
    tshirt = [stats1.flood_depth_formal[2], stats1.flood_depth_subsidized[2], stats1.flood_depth_informal[2], stats1.flood_depth_backyard[2]]
    tshirtb = [stats1.flood_depth_formal[3], stats1.flood_depth_subsidized[3], stats1.flood_depth_informal[3], stats1.flood_depth_backyard[3]]
    formal_shirt = [stats1.flood_depth_formal[5], stats1.flood_depth_subsidized[5], stats1.flood_depth_informal[5], stats1.flood_depth_backyard[5]]
    tshirt2 = [stats2.flood_depth_formal[2], stats2.flood_depth_subsidized[2], stats2.flood_depth_informal[2], stats2.flood_depth_backyard[2]]
    tshirtb2 = [stats2.flood_depth_formal[3], stats2.flood_depth_subsidized[3], stats2.flood_depth_informal[3], stats2.flood_depth_backyard[3]]
    formal_shirt2 = [stats2.flood_depth_formal[5], stats2.flood_depth_subsidized[5], stats2.flood_depth_informal[5], stats2.flood_depth_backyard[5]]
    colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
    r = np.arange(len(label))
    barWidth = 0.25
    plt.figure(figsize=(10,7))
    plt.bar(r, np.array(tshirt), color=colors[1], edgecolor='white', width=barWidth, label='20 years')
    plt.bar(r, np.array(tshirtb) - (np.array(tshirt)), bottom=(np.array(tshirt)), color=colors[2], edgecolor='white', width=barWidth, label='50 years')
    plt.bar(r, np.array(formal_shirt) - (np.array(tshirtb)), bottom=(np.array(tshirtb)), color=colors[3], edgecolor='white', width=barWidth, label='100 years')
    plt.bar(r + 0.25, np.array(tshirt2), color=colors[1], edgecolor='white', width=barWidth)
    plt.bar(r + 0.25, np.array(tshirtb2) - np.array(tshirt2), bottom=(np.array(tshirt2)), color=colors[2], edgecolor='white', width=barWidth)
    plt.bar(r + 0.25, np.array(formal_shirt2) - np.array(tshirtb2), bottom=np.array(tshirtb2), color=colors[3], edgecolor='white', width=barWidth)
    plt.legend()
    plt.xticks(r, label)
    #plt.ylim(0, 1)
    plt.text(r[0] - 0.1, stats1.flood_depth_formal[5] + 0.01, legend1)
    plt.text(r[1] - 0.1, stats1.flood_depth_subsidized[5] + 0.01, legend1) 
    plt.text(r[2] - 0.1, stats1.flood_depth_informal[5] + 0.01, legend1) 
    plt.text(r[3] - 0.1, stats1.flood_depth_backyard[5] + 0.01, legend1)
    plt.text(r[0] + 0.15, stats2.flood_depth_formal[5] + 0.01, legend2)
    plt.text(r[1] + 0.15, stats2.flood_depth_subsidized[5] + 0.01, legend2) 
    plt.text(r[2] + 0.15, stats2.flood_depth_informal[5] + 0.01, legend2) 
    plt.text(r[3] + 0.15, max(stats2.flood_depth_backyard[2], stats2.flood_depth_backyard[3], stats2.flood_depth_backyard[5]) + 0.01, legend2) 
    plt.ylabel("Average flood depth (m)")
    plt.tick_params(labelbottom=True)
    plt.show()
    plt.savefig('C:/Users/Coupain/Desktop/cape_town/4. Sorties/' + name + '/validation_flood_depth_' + type_flood + '.png')  
    plt.close()


    jeans = [stats1.fraction_formal_in_flood_prone_area[2], stats1.fraction_subsidized_in_flood_prone_area[2], stats1.fraction_informal_in_flood_prone_area[2], stats1.fraction_backyard_in_flood_prone_area[2]]
    tshirt = [stats1.fraction_formal_in_flood_prone_area[3], stats1.fraction_subsidized_in_flood_prone_area[3], stats1.fraction_informal_in_flood_prone_area[3], stats1.fraction_backyard_in_flood_prone_area[3]]
    formal_shirt = [stats1.fraction_formal_in_flood_prone_area[5], stats1.fraction_subsidized_in_flood_prone_area[5], stats1.fraction_informal_in_flood_prone_area[5], stats1.fraction_backyard_in_flood_prone_area[5]]
    jeans2 = [stats2.fraction_formal_in_flood_prone_area[2], stats2.fraction_subsidized_in_flood_prone_area[2], stats2.fraction_informal_in_flood_prone_area[2], stats2.fraction_backyard_in_flood_prone_area[2]]
    tshirt2 = [stats2.fraction_formal_in_flood_prone_area[3], stats2.fraction_subsidized_in_flood_prone_area[3], stats2.fraction_informal_in_flood_prone_area[3], stats2.fraction_backyard_in_flood_prone_area[3]]
    formal_shirt2 = [stats2.fraction_formal_in_flood_prone_area[5], stats2.fraction_subsidized_in_flood_prone_area[5], stats2.fraction_informal_in_flood_prone_area[5], stats2.fraction_backyard_in_flood_prone_area[5]]
    colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
    r = np.arange(len(label))
    barWidth = 0.25
    plt.figure(figsize=(10,7))
    plt.bar(r, jeans, color=colors[0], edgecolor='white', width=barWidth, label="20 years")
    plt.bar(r, np.array(tshirt) - np.array(jeans), bottom=np.array(jeans), color=colors[1], edgecolor='white', width=barWidth, label='50 years')
    plt.bar(r, np.array(formal_shirt) - (np.array(tshirt)), bottom=(np.array(tshirt)), color=colors[2], edgecolor='white', width=barWidth, label='100 years')
    plt.bar(r + 0.25, np.array(jeans2), color=colors[0], edgecolor='white', width=barWidth)
    plt.bar(r + 0.25, np.array(tshirt2) - np.array(jeans2), bottom=np.array(jeans2), color=colors[1], edgecolor='white', width=barWidth)
    plt.bar(r + 0.25, np.array(formal_shirt2) - np.array(tshirt2), bottom=np.array(tshirt2), color=colors[2], edgecolor='white', width=barWidth)
    plt.legend(loc = 'upper right')
    plt.xticks(r, label)
    plt.text(r[0] - 0.1, stats1.fraction_formal_in_flood_prone_area[5] + 0.005, legend1)
    plt.text(r[1] - 0.1, stats1.fraction_subsidized_in_flood_prone_area[5] + 0.005, legend1) 
    plt.text(r[2] - 0.1, stats1.fraction_informal_in_flood_prone_area[5] + 0.005, legend1) 
    plt.text(r[3] - 0.1, stats1.fraction_backyard_in_flood_prone_area[5] + 0.005, legend1)
    plt.text(r[0] + 0.15, stats2.fraction_formal_in_flood_prone_area[5] + 0.005, legend2)
    plt.text(r[1] + 0.15, stats2.fraction_subsidized_in_flood_prone_area[5] + 0.005, legend2) 
    plt.text(r[2] + 0.15, stats2.fraction_informal_in_flood_prone_area[5] + 0.005, legend2) 
    plt.text(r[3] + 0.15, stats2.fraction_backyard_in_flood_prone_area[5] + 0.005, legend2) 
    plt.tick_params(labelbottom=True)
    plt.ylabel("Dwellings in flood-prone areas (%)")
    plt.show()
    plt.savefig('C:/Users/Coupain/Desktop/cape_town/4. Sorties/' + name + '/validation_flood_proportion_' + type_flood +'.png')  
    plt.close()
    
"""

is_pockets = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/IS_Pockets_w_Count/pocket_per_cell.xlsx').x
is_pockets[np.isnan(is_pockets)] = 0


housing_types_grid_sal[np.isnan(housing_types_grid_sal)] = 0

#housing_types_grid_fromsubplace.formal_grid_2011 = housing_types_grid_fromsubplace.formal_grid_2011 * (sum(housing_types_grid_sal.formal_grid) / sum(housing_types_grid_fromsubplace.formal_grid_2011))

xData = grid.dist
formal_subplace = (housing_types_grid_fromsubplace.formal_grid_2011) / 0.25
backyard_subplace = (housing_types_grid_fromsubplace.backyard_grid_2011) / 0.25
informal_subplace = (housing_types_grid_fromsubplace.informal_grid_2011) / 0.25
informal_smallarea = (housing_types_grid_sal.informal_grid) / 0.25
backyard_smallarea = (housing_types_grid_sal.backyard_formal_grid + housing_types_grid_sal.backyard_informal_grid) / 0.25
formal_smallarea = (housing_types_grid_sal.formal_grid) / 0.25
remote_sensing = is_pockets / 0.25

df = pd.DataFrame(data = np.transpose(np.array([xData, formal_subplace, backyard_subplace, informal_subplace, formal_smallarea, backyard_smallarea, informal_smallarea, remote_sensing])), columns = ["xData", "formal_subplace", "backyard_subplace", "informal_subplace", "formal_smallarea", "backyard_smallarea", "informal_smallarea", "remote_sensing"])
df["round"] = round(df.xData)
new_df = df.groupby(['round']).mean()
    
plt.figure(figsize=(10, 7))
plt.plot(np.arange(max(df["round"] + 1)), new_df.formal_subplace, color = "black", label = "Subplace")
plt.plot(np.arange(max(df["round"] + 1)), new_df.formal_smallarea, color = "green", label = "Small area")
axes = plt.axes()
axes.set_ylim([0, 1600])
axes.set_xlim([0, 40])
#plt.title("Formal")
plt.legend()
plt.tick_params(labelbottom=True)
plt.xlabel("Distance to the city center (km)")
plt.ylabel("Households density (per km2)")

plt.figure(figsize=(10, 7))
plt.plot(np.arange(max(df["round"] + 1)), new_df.informal_subplace, color = "black", label = "Subplace")
plt.plot(np.arange(max(df["round"] + 1)), new_df.informal_smallarea, color = "green", label = "Small area")
plt.plot(np.arange(max(df["round"] + 1)), new_df.remote_sensing, color = "red", label = "Remote sensing")
axes = plt.axes()
axes.set_ylim([0, 350])
axes.set_xlim([0, 40])
plt.xlabel("Distance to the city center (km)")
plt.ylabel("Households density (per km2)")
plt.legend()
plt.tick_params(labelbottom=True)

    
plt.figure(figsize=(10, 7))
plt.plot(np.arange(max(df["round"] + 1)), new_df.backyard_subplace, color = "black", label = "Subplace")
plt.plot(np.arange(max(df["round"] + 1)), new_df.backyard_smallarea, color = "green", label = "Small area")
axes = plt.axes()
axes.set_ylim([0, 450])
axes.set_xlim([0, 40])
#plt.title("Backyard")
plt.legend()
plt.tick_params(labelbottom=True)
plt.xlabel("Distance to the city center (km)")
plt.ylabel("Households density (per km2)")




housing_types_grid_sal[np.isnan(housing_types_grid_sal)] = 0

xData = grid.dist
formal_backyard = (housing_types_grid_sal.backyard_formal_grid) / 0.25
informal_backyard = (housing_types_grid_sal.backyard_informal_grid) / 0.25


df = pd.DataFrame(data = np.transpose(np.array([xData, formal_backyard, informal_backyard])), columns = ["xData", "formal_backyard", "informal_backyard"])
df["round"] = round(df.xData)
new_df = df.groupby(['round']).mean()

plt.figure(figsize=(10, 7))
plt.plot(np.arange(max(df["round"] + 1)), new_df.formal_backyard, color = "black", label = "Backyard (bricks)")
plt.plot(np.arange(max(df["round"] + 1)), new_df.informal_backyard, color = "green", label = "Backyard (others)")
axes = plt.axes()
#axes.set_ylim([0, 450])
axes.set_xlim([0, 40])
#plt.title("Backyard")
plt.legend()
plt.tick_params(labelbottom=True)
plt.xlabel("Distance to the city center (km)")
plt.ylabel("Households density (per km2)")

floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"


stats_per_housing_type = pd.DataFrame(columns = ['flood',
                                                     'formal_backyard_prop', 'informal_backyard_prop',
                                                     'flood_depth_formal_backyard', 'flood_depth_informal_backyard'])
for flood in floods:
    type_flood = copy.deepcopy(flood)
    flood = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))
       
    stats_per_housing_type = stats_per_housing_type.append({'flood': type_flood, 
                                                            'formal_backyard_prop': np.sum(flood['prop_flood_prone'] * housing_types_grid_sal.backyard_formal_grid) / sum(housing_types_grid_sal.backyard_formal_grid), 
                                                            'informal_backyard_prop': np.sum(flood['prop_flood_prone'] * housing_types_grid_sal.backyard_informal_grid) / sum(housing_types_grid_sal.backyard_informal_grid),
                                                            'flood_depth_formal_backyard': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * housing_types_grid_sal.backyard_formal_grid)  / sum(flood['prop_flood_prone'] * housing_types_grid_sal.backyard_formal_grid)),
                                                            'flood_depth_informal_backyard': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * housing_types_grid_sal.backyard_informal_grid)  / sum(flood['prop_flood_prone'] * housing_types_grid_sal.backyard_informal_grid))}, ignore_index = True) 
    
label = ["Backyard (bricks)", "Backyard (others)"]
tshirt = [stats_per_housing_type.flood_depth_formal_backyard[2], stats_per_housing_type.flood_depth_informal_backyard[2]]
tshirtb = [stats_per_housing_type.flood_depth_formal_backyard[3], stats_per_housing_type.flood_depth_informal_backyard[3]]
formal_shirt = [stats_per_housing_type.flood_depth_formal_backyard[5], stats_per_housing_type.flood_depth_informal_backyard[5]]
colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
r = np.array([0, 0.5])
barWidth = 0.25
plt.figure(figsize=(10,7))
plt.bar(r, np.array(tshirt), color=colors[1], edgecolor='white', width=barWidth, label='20 years')
plt.bar(r, np.array(tshirtb) - (np.array(tshirt)), bottom=(np.array(tshirt)), color=colors[2], edgecolor='white', width=barWidth, label='50 years')
plt.bar(r, np.array(formal_shirt) - (np.array(tshirtb)), bottom=(np.array(tshirtb)), color=colors[3], edgecolor='white', width=barWidth, label='100 years')
plt.legend()
plt.xticks(r, label)
plt.ylim(0, 0.2)
plt.ylabel("Average flood depth (m)")
plt.tick_params(labelbottom=True)
plt.show()


jeans = [stats_per_housing_type.formal_backyard_prop[2], stats_per_housing_type.informal_backyard_prop[2]]
tshirt = [stats_per_housing_type.formal_backyard_prop[3], stats_per_housing_type.informal_backyard_prop[3]]
formal_shirt = [stats_per_housing_type.formal_backyard_prop[5], stats_per_housing_type.informal_backyard_prop[5]]
colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
r = np.array([0, 0.5])
barWidth = 0.25
plt.figure(figsize=(10,7))
plt.bar(r, jeans, color=colors[0], edgecolor='white', width=barWidth, label="20 years")
plt.bar(r, np.array(tshirt) - np.array(jeans), bottom=np.array(jeans), color=colors[1], edgecolor='white', width=barWidth, label='50 years')
plt.bar(r, np.array(formal_shirt) - (np.array(tshirt)), bottom=(np.array(tshirt)), color=colors[2], edgecolor='white', width=barWidth, label='100 years')
plt.legend()
plt.xticks(r, label)
plt.tick_params(labelbottom=True)
plt.ylabel("Dwellings in flood-prone areas (%)")
plt.show()

"""