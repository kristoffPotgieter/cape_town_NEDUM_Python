# -*- coding: utf-8 -*-
"""
Created on Mon May  9 09:20:08 2022

@author: monni
"""

def convert_income_distribution(income_distribution, grid, path_data, data_sp):
    """Import SP data for income distribution in grid form."""
    grid_intersect = pd.read_csv(
        path_data + 'grid_SP_intersect.csv', sep=';')

    income0_grid = gen_small_areas_to_grid(
        grid, grid_intersect, income_distribution[:, 0],
        data_sp["sp_code"], 'SP')
    income1_grid = gen_small_areas_to_grid(
        grid, grid_intersect, income_distribution[:, 1],
        data_sp["sp_code"], 'SP')
    income2_grid = gen_small_areas_to_grid(
        grid, grid_intersect, income_distribution[:, 2],
        data_sp["sp_code"], 'SP')
    income3_grid = gen_small_areas_to_grid(
        grid, grid_intersect, income_distribution[:, 3],
        data_sp["sp_code"], 'SP')

    # We correct the values per pixel by reweighting with the
    # ratio of total original number over total estimated number
    income0_grid = (income0_grid * (np.nansum(income_distribution[:, 0])
                                    / np.nansum(income0_grid)))
    income1_grid = (income1_grid * (np.nansum(income_distribution[:, 1])
                                    / np.nansum(income1_grid)))
    income2_grid = (income2_grid * (np.nansum(income_distribution[:, 2])
                                    / np.nansum(income2_grid)))
    income3_grid = (income3_grid * (np.nansum(income_distribution[:, 3])
                                    / np.nansum(income3_grid)))

    income_grid = np.stack(
        [income0_grid, income1_grid, income2_grid, income3_grid])

    # Replace missing values by zero
    income_grid[np.isnan(income_grid)] = 0

    np.save(path_data + "income_distrib_grid.npy", income_grid)

    return income_grid

# TODO: check if deprecated with Basile

def SP_to_grid_2011_1(data_SP, grid, path_data):
    """Adapt SP data to grid dimension."""
    grid_intersect = pd.read_csv(path_data + 'grid_SP_intersect.csv', sep=';')
    data_grid = np.zeros(len(grid.dist))
    for index in range(0, len(grid.dist)):
        # A priori, each SP code is associated to several pixels
        intersect = np.unique(
            grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.id[index]]
            )
        area_exclu = 0
        for i in range(0, len(intersect)):
            if len(data_SP['sp_code' == intersect[i]]) == 0:
                # We exclude the area of (all) pixel(s) corresponding to
                # unmatched SP
                area_exclu = (
                    area_exclu
                    + sum(grid_intersect.Area[(
                        grid_intersect.ID_grille == grid.id[index])
                        & (grid_intersect.SP_CODE == intersect[i])])
                    )
            else:
                # We add the SP data times the area of matched pixel(s)
                data_grid[index] = (
                    data_grid[index]
                    + sum(grid_intersect.Area[(
                        grid_intersect.ID_grille == grid.id[index])
                        & (grid_intersect.SP_CODE == intersect[i])])
                    * data_SP['sp_code' == intersect[i]]
                    )
        # We do not update data if excluded area is bigger than 90% of the
        # matched SP area (not used in practice)
        if area_exclu > (0.9 * sum(
                grid_intersect.Area[grid_intersect.ID_grille == grid.id[index]]
                )):
            data_grid[index] = np.nan
        # Else, if there is some positive matching, we update data with
        # nonsense?
        elif sum(
               grid_intersect.Area[grid_intersect.ID_grille == grid.id[index]]
               ) - area_exclu > 0:
            data_grid[index] = (
                data_grid[index]
                / (sum(grid_intersect.Area[
                    grid_intersect.ID_grille == grid.id[index]]
                    ) - area_exclu))
        else:
            data_grid[index] = np.nan

    return data_grid