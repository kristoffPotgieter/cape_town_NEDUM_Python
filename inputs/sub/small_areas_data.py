# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:24:49 2020

@author: Charlotte Liotta
"""

sal_data = pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/CT Dwelling type data validation workbook 20201204 v2.xlsx", header = 6)
sal_data["informal"] = sal_data["Informal dwelling (shack; not in backyard; e.g. in an informal/squatter settlement or on a farm)"]
sal_data["backyard_formal"] = sal_data["House/flat/room in backyard"]
sal_data["backyard_informal"] = sal_data["Informal dwelling (shack; in backyard)"]
sal_data["formal"] = np.nansum(sal_data.iloc[:, 3:15], 1)

grid_intersect = pd.read_csv('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Basile data/grid_SAL_intersect.csv', sep = ';') 

informal_grid = small_areas_to_grid(grid, grid_intersect, sal_data["informal"], sal_data["Small Area Code"])
backyard_formal_grid = small_areas_to_grid(grid, grid_intersect, sal_data["backyard_formal"], sal_data["Small Area Code"])
backyard_informal_grid = small_areas_to_grid(grid, grid_intersect, sal_data["backyard_informal"], sal_data["Small Area Code"])
formal_grid = small_areas_to_grid(grid, grid_intersect, sal_data["formal"], sal_data["Small Area Code"])

informal_grid = informal_grid * (np.nansum(sal_data["informal"]) / np.nansum(informal_grid))
backyard_formal_grid = backyard_formal_grid * (np.nansum(sal_data["backyard_formal"]) / np.nansum(backyard_formal_grid))
backyard_informal_grid = backyard_informal_grid * (np.nansum(sal_data["backyard_informal"]) / np.nansum(backyard_informal_grid))
formal_grid = formal_grid * ((626770 + 194258) / np.nansum(formal_grid))

housing_types_grid_sal = pd.DataFrame()
housing_types_grid_sal["informal_grid"] = informal_grid
housing_types_grid_sal["backyard_formal_grid"] = backyard_formal_grid
housing_types_grid_sal["backyard_informal_grid"] = backyard_informal_grid
housing_types_grid_sal["formal_grid"] = formal_grid

housing_types_grid_sal.to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/housing_types_grid_sal.xlsx')



def small_areas_to_grid(grid, grid_intersect, small_area_data, small_area_code):
    grid_data = np.zeros(len(grid.dist))
    for index in range (0, len(grid.dist)):
        intersect = np.unique(grid_intersect.SAL_CODE[grid_intersect.ID_grille == grid.id[index]])
        if len(intersect) == 0:
            grid_data[index] = np.nan
        else:
            for i in range(0, len(intersect)):
                sal_code = intersect[i]
                sal_area_intersect = np.nansum(grid_intersect.Area_inter[(grid_intersect.ID_grille == grid.id[index]) & (grid_intersect.SAL_CODE == sal_code)].squeeze())
                sal_area = np.nansum(grid_intersect.Area_inter[(grid_intersect.SAL_CODE == sal_code)])
                if len(sal_data.informal[sal_data["Small Area Code"] == sal_code]) > 0:
                    add = small_area_data[small_area_code == sal_code] * (sal_area_intersect / sal_area)
                else:
                    add = 0
                grid_data[index] = grid_data[index] + add
    return grid_data
