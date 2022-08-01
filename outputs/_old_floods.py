# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:33:18 2022

@author: monni
"""

# TODO: put in old script

for item in fluviald_floods:

    print(item)
    df = fluviald_damages_2d_data[item]
    FP_damages = df.formal_structure_damages + df.formal_content_damages
    FS_damages = (
        df.subsidized_structure_damages + df.subsidized_content_damages)
    IB_damages = df.backyard_structure_damages + df.backyard_content_damages
    IS_damages = df.informal_structure_damages + df.informal_content_damages

    try:
        outexp.export_map(FP_damages, grid, geo_grid,
                          path_plots, item + '_FP_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(FP_damages[FP_damages > 0], 0.9),
                          lbnd=np.min(FP_damages[FP_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(FS_damages, grid, geo_grid,
                          path_plots, item + '_FS_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(FS_damages[FS_damages > 0], 0.9),
                          lbnd=np.min(FS_damages[FS_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(IB_damages, grid, geo_grid,
                          path_plots, item + '_IB_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(IB_damages[IB_damages > 0], 0.9),
                          lbnd=np.min(IB_damages[IB_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(IS_damages, grid, geo_grid,
                          path_plots, item + '_IS_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(IS_damages[IS_damages > 0], 0.9),
                          lbnd=np.min(IS_damages[IS_damages > 0]))
    except IndexError:
        pass

for item in fluvialu_floods:

    print(item)
    df = fluvialu_damages_2d_data[item]
    FP_damages = df.formal_structure_damages + df.formal_content_damages
    FS_damages = (
        df.subsidized_structure_damages + df.subsidized_content_damages)
    IB_damages = df.backyard_structure_damages + df.backyard_content_damages
    IS_damages = df.informal_structure_damages + df.informal_content_damages

    try:
        outexp.export_map(FP_damages, grid, geo_grid,
                          path_plots, item + '_FP_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(FP_damages[FP_damages > 0], 0.9),
                          lbnd=np.min(FP_damages[FP_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(FS_damages, grid, geo_grid,
                          path_plots, item + '_FS_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(FS_damages[FS_damages > 0], 0.9),
                          lbnd=np.min(FS_damages[FS_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(IB_damages, grid, geo_grid,
                          path_plots, item + '_IB_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(IB_damages[IB_damages > 0], 0.9),
                          lbnd=np.min(IB_damages[IB_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(IS_damages, grid, geo_grid,
                          path_plots, item + '_IS_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(IS_damages[IS_damages > 0], 0.9),
                          lbnd=np.min(IS_damages[IS_damages > 0]))
    except IndexError:
        pass

for item in pluvial_floods:

    print(item)
    df = pluvial_damages_2d_data[item]
    FP_damages = df.formal_structure_damages + df.formal_content_damages
    FS_damages = (
        df.subsidized_structure_damages + df.subsidized_content_damages)
    IB_damages = df.backyard_structure_damages + df.backyard_content_damages
    IS_damages = df.informal_structure_damages + df.informal_content_damages

    try:
        outexp.export_map(FP_damages, grid, geo_grid,
                          path_plots, item + '_FP_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(FP_damages[FP_damages > 0], 0.9),
                          lbnd=np.min(FP_damages[FP_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(FS_damages, grid, geo_grid,
                          path_plots, item + '_FS_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(FS_damages[FS_damages > 0], 0.9),
                          lbnd=np.min(FS_damages[FS_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(IB_damages, grid, geo_grid,
                          path_plots, item + '_IB_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(IB_damages[IB_damages > 0], 0.9),
                          lbnd=np.min(IB_damages[IB_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(IS_damages, grid, geo_grid,
                          path_plots, item + '_IS_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(IS_damages[IS_damages > 0], 0.9),
                          lbnd=np.min(IS_damages[IS_damages > 0]))
    except IndexError:
        pass

for item in coastal_floods:

    print(item)
    df = coastal_damages_2d_data[item]
    FP_damages = df.formal_structure_damages + df.formal_content_damages
    FS_damages = (
        df.subsidized_structure_damages + df.subsidized_content_damages)
    IB_damages = df.backyard_structure_damages + df.backyard_content_damages
    IS_damages = df.informal_structure_damages + df.informal_content_damages

    try:
        outexp.export_map(FP_damages, grid, geo_grid,
                          path_plots, item + '_FP_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(FP_damages[FP_damages > 0], 0.9),
                          lbnd=np.min(FP_damages[FP_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(FS_damages, grid, geo_grid,
                          path_plots, item + '_FS_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(FS_damages[FS_damages > 0], 0.9),
                          lbnd=np.min(FS_damages[FS_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(IB_damages, grid, geo_grid,
                          path_plots, item + '_IB_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(IB_damages[IB_damages > 0], 0.9),
                          lbnd=np.min(IB_damages[IB_damages > 0]))
    except IndexError:
        pass
    try:
        outexp.export_map(IS_damages, grid, geo_grid,
                          path_plots, item + '_IS_damages_data', "",
                          path_tables,
                          ubnd=np.quantile(IS_damages[IS_damages > 0], 0.9),
                          lbnd=np.min(IS_damages[IS_damages > 0]))
    except IndexError:
        pass

def compute_share_income_destroyed(
        input_dict, selected_net_income_formal, selected_net_income_rdp,
        selected_net_income_backyard, selected_net_income_informal,
        nb_households_formal, nb_households_rdp, nb_households_backyard,
        nb_households_informal):
    """Compute share of income net of commuting costs destroyed by floods."""
    output_dict = {}
    for key in input_dict.keys():
        df = input_dict[key]
        new_df = df.copy()

        new_df["formal_structure_damages"] = (
            new_df["formal_structure_damages"]
            / (nb_households_formal * selected_net_income_formal))
        new_df["formal_content_damages"] = (
            new_df["formal_content_damages"] /
            (nb_households_formal * selected_net_income_formal))

        new_df["subsidized_structure_damages"] = (
            new_df["subsidized_structure_damages"] /
            (nb_households_rdp * selected_net_income_rdp))
        new_df["subsidized_content_damages"] = (
            new_df["subsidized_content_damages"] /
            (nb_households_rdp * selected_net_income_rdp))

        new_df["backyard_structure_damages"] = (
            new_df["backyard_structure_damages"] /
            (nb_households_backyard * selected_net_income_backyard))
        new_df["backyard_content_damages"] = (
            new_df["backyard_content_damages"] /
            (nb_households_backyard * selected_net_income_backyard))

        new_df["informal_structure_damages"] = (
            new_df["informal_structure_damages"] /
            (nb_households_informal * selected_net_income_informal))
        new_df["informal_content_damages"] = (
            new_df["informal_content_damages"] /
            (nb_households_informal * selected_net_income_informal))

        new_df = new_df.fillna(value=0)
        output_dict[key] = new_df

    return output_dict


def compute_share_housingval_destroyed(
        input_dict, selected_net_income_formal, selected_net_income_rdp,
        selected_net_income_backyard, selected_net_income_informal,
        nb_households_formal, nb_households_rdp, nb_households_backyard,
        nb_households_informal):
    """Compute share of income net of commuting costs destroyed by floods."""
    output_dict = {}
    for key in input_dict.keys():
        df = input_dict[key]
        new_df = df.copy()

        new_df["formal_structure_damages"] = (
            new_df["formal_structure_damages"]
            / (nb_households_formal * selected_net_income_formal))
        new_df["formal_content_damages"] = (
            new_df["formal_content_damages"] /
            (nb_households_formal * selected_net_income_formal))

        new_df["subsidized_structure_damages"] = (
            new_df["subsidized_structure_damages"] /
            (nb_households_rdp * selected_net_income_rdp))
        new_df["subsidized_content_damages"] = (
            new_df["subsidized_content_damages"] /
            (nb_households_rdp * selected_net_income_rdp))

        new_df["backyard_structure_damages"] = (
            new_df["backyard_structure_damages"] /
            (nb_households_backyard * selected_net_income_backyard))
        new_df["backyard_content_damages"] = (
            new_df["backyard_content_damages"] /
            (nb_households_backyard * selected_net_income_backyard))

        new_df["informal_structure_damages"] = (
            new_df["informal_structure_damages"] /
            (nb_households_informal * selected_net_income_informal))
        new_df["informal_content_damages"] = (
            new_df["informal_content_damages"] /
            (nb_households_informal * selected_net_income_informal))

        new_df = new_df.fillna(value=0)
        output_dict[key] = new_df

    return output_dict