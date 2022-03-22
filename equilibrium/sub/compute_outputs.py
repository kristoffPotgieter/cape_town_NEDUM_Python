# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:01:05 2020.

@author: Charlotte Liotta
"""
import numpy as np
import equilibrium.sub.functions_solver as eqsol


def compute_outputs(housing_type,
                    utility,
                    amenities,
                    param,
                    income_net_of_commuting_costs,
                    fraction_capital_destroyed,
                    grid,
                    income_class_by_housing_type,
                    options,
                    housing_limit,
                    agricultural_rent,
                    interest_rate,
                    coeff_land,
                    minimum_housing_supply,
                    construction_param,
                    housing_in,
                    param_pockets,
                    param_backyards_pockets):
    """d."""
    # %% Dwelling size in selected pixels per (endogenous) housing type

    if housing_type == 'formal':

        dwelling_size = eqsol.compute_dwelling_size_formal(
            utility, amenities, param, income_net_of_commuting_costs,
            fraction_capital_destroyed)

        # Here, we introduce the minimum lot-size
        dwelling_size = np.maximum(dwelling_size, param["mini_lot_size"])
        # And we make sure we do not consider cases where some income groups
        # would have no access to formal housing
        dwelling_size[income_class_by_housing_type.formal == 0, :] = np.nan

    elif housing_type == 'backyard':

        # Defined exogenously
        dwelling_size = param["shack_size"] * np.ones((4, len(grid.dist)))
        # As before
        dwelling_size[income_class_by_housing_type.backyard == 0, :] = np.nan

    elif housing_type == 'informal':

        # Defined exogenously
        dwelling_size = param["shack_size"] * np.ones((4, len(grid.dist)))
        # As before
        dwelling_size[income_class_by_housing_type.settlement == 0, :] = np.nan

    # %% Bid rent functions in selected pixels per (endogenous) housing type

    # What is the point? Set as parameter?
    # fraction_capital_destroyed = 0

    if housing_type == 'formal':

        # See research note, p.11
        R_mat = (param["beta"] * (income_net_of_commuting_costs)
                 / (dwelling_size - (param["alpha"] * param["q0"])))
        R_mat[income_net_of_commuting_costs < 0] = 0
        R_mat[income_class_by_housing_type.formal == 0, :] = 0

    elif housing_type == 'backyard':

        # See research note, p.12: shouldn't we distinguish between structural
        # and content damage in fraction_capital_destroyed?
        R_mat = (
            (1 / param["shack_size"])
            * (income_net_of_commuting_costs
               - ((1 + np.array(fraction_capital_destroyed.contents_backyard)[
                   None, :] * param["fraction_z_dwellings"])
                  * ((utility[:, None] / (amenities[None, :]
                                          * param_backyards_pockets[None, :]
                                          * ((dwelling_size - param["q0"])
                                             ** param["beta"])))
                     ** (1 / param["alpha"])))
               - (param["informal_structure_value"]
                  * (interest_rate + param["depreciation_rate"]))
               - (np.array(fraction_capital_destroyed.structure_backyards)[
                   None, :] * param["informal_structure_value"]))
            )
        R_mat[income_class_by_housing_type.backyard == 0, :] = 0

    elif housing_type == 'informal':

        # See research note, p.12: same definition as for backyards
        R_mat = (
            (1 / param["shack_size"])
            * (income_net_of_commuting_costs
               - ((1 + np.array(fraction_capital_destroyed.contents_informal)[
                   None, :] * param["fraction_z_dwellings"])
                  * ((utility[:, None] / (amenities[None, :]
                                          * param_pockets[None, :]
                                          * ((dwelling_size - param["q0"])
                                             ** param["beta"])))
                     ** (1 / param["alpha"])))
               - (param["informal_structure_value"]
                  * (interest_rate + param["depreciation_rate"]))
               - (np.array(
                   fraction_capital_destroyed.structure_informal_settlements
                   )[None, :] * param["informal_structure_value"]))
            )
        R_mat[income_class_by_housing_type.settlement == 0, :] = 0

    # We clean the results just in case
    R_mat[R_mat < 0] = 0
    R_mat[np.isnan(R_mat)] = 0

    # Income group in each location
    proba = (R_mat == np.nanmax(R_mat, 0))
    limit = ((income_net_of_commuting_costs > 0)
             & (proba > 0)
             & (~np.isnan(income_net_of_commuting_costs))
             & (R_mat > 0))
    proba = proba * limit

    which_group = np.nanargmax(R_mat, 0)

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
        housing_supply = eqsol.compute_housing_supply_formal(
            R, options, housing_limit, param, agricultural_rent, interest_rate,
            fraction_capital_destroyed, minimum_housing_supply,
            construction_param, housing_in, dwelling_size)
        housing_supply[R == 0] = 0
    elif housing_type == 'backyard':
        housing_supply = eqsol.compute_housing_supply_backyard(
            R, param, income_net_of_commuting_costs,
            fraction_capital_destroyed, dwelling_size)
        housing_supply[R == 0] = 0
    elif housing_type == 'informal':
        housing_supply = 1000000 * np.ones(len(which_group))
        housing_supply[R == 0] = 0

    # %% Outputs

    people_init = housing_supply / dwelling_size * (np.nansum(limit, 0) > 0)
    people_init[np.isnan(people_init)] = 0
    people_init_land = people_init * coeff_land * 0.25

    people_center = np.array(people_init_land)[None, :] * proba
    people_center[np.isnan(people_center)] = 0
    job_simul = np.nansum(people_center, 1)

    if housing_type == 'formal':
        R = np.maximum(R, agricultural_rent)

    return (job_simul, R, people_init, people_center, housing_supply,
            dwelling_size, R_mat)
