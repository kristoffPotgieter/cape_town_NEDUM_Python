# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:40:39 2020

@author: Charlotte Liotta
"""

import numpy as np
import pandas as pd

####### INFER FLOOD DEPTH WBUS2

### Method 1

def infer_WBUS2_depth(housing_types, param, path_folder):
    
    path_data = path_folder + "FATHOM/"
    FATHOM_20yr = np.squeeze(pd.read_excel(path_data + 'FD_20yr' + ".xlsx"))
    FATHOM_50yr = np.squeeze(pd.read_excel(path_data + 'FD_50yr' + ".xlsx"))
    FATHOM_100yr = np.squeeze(pd.read_excel(path_data + 'FD_100yr' + ".xlsx"))

    FATHOM_20yr.pop_flood_prone = FATHOM_20yr.prop_flood_prone * (housing_types.informal_grid + housing_types.formal_grid + housing_types.backyard_formal_grid + housing_types.backyard_informal_grid)
    FATHOM_50yr.pop_flood_prone = FATHOM_50yr.prop_flood_prone * (housing_types.informal_grid + housing_types.formal_grid + housing_types.backyard_formal_grid + housing_types.backyard_informal_grid)
    FATHOM_100yr.pop_flood_prone = FATHOM_100yr.prop_flood_prone * (housing_types.informal_grid + housing_types.formal_grid + housing_types.backyard_formal_grid + housing_types.backyard_informal_grid)
   
    param["depth_WBUS2_20yr"] = np.nansum(FATHOM_20yr.pop_flood_prone * FATHOM_20yr.flood_depth) / np.nansum(FATHOM_20yr.pop_flood_prone)
    param["depth_WBUS2_50yr"] = np.nansum(FATHOM_50yr.pop_flood_prone * FATHOM_50yr.flood_depth) / np.nansum(FATHOM_50yr.pop_flood_prone)
    param["depth_WBUS2_100yr"] = np.nansum(FATHOM_100yr.pop_flood_prone * FATHOM_100yr.flood_depth) / np.nansum(FATHOM_100yr.pop_flood_prone)

    return param
    