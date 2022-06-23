# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:32:48 2020.

@author: Charlotte Liotta
"""

import scipy
import pandas as pd
import numpy as np
# import os
import matplotlib as mpl
import matplotlib.pyplot as plt
# import geopandas as gpd
import inspect
from scipy.interpolate import griddata


def retrieve_name(var, depth):
    """Retrieve name of a variable."""
    if depth == 0:
        (callers_local_vars
         ) = inspect.currentframe().f_back.f_back.f_locals.items()
    elif depth == 1:
        (callers_local_vars
         ) = inspect.currentframe().f_back.f_back.f_back.f_locals.items()
    name = [var_name for var_name, var_val
            in callers_local_vars if var_val is var]
    return name[0]


def from_df_to_gdf(array, geo_grid):
    """Convert map array/series inputs into grid-level GeoDataFrames."""
    # Convert array or series to data frame
    df = pd.DataFrame(array)
    str_array = retrieve_name(array, depth=1)
    df = df.rename(columns={df.columns[0]: str_array})
    gdf = pd.merge(geo_grid, df, left_index=True, right_index=True)

    return gdf


def export_map(value, grid, geo_grid, export_name, title,
               path_tables,
               ubnd, lbnd=0, cmap='Reds'):
    """Generate 2D heat maps of any spatial input."""
    plt.figure(figsize=(10, 7))
    Map = plt.scatter(grid.x,
                      grid.y,
                      s=None,
                      c=value,
                      cmap=cmap,
                      marker='.')
    plt.colorbar(Map)
    plt.axis('off')
    plt.clim(lbnd, ubnd)
    plt.title(title)
    plt.savefig(export_name)
    plt.close()

    # value.to_csv(path_tables + str(value))
    gdf = from_df_to_gdf(value, geo_grid)
    str_value = retrieve_name(value, depth=0)
    gdf.to_file(path_tables + str_value + '.shp')
    print(str_value + ' done')

    return gdf


def import_employment_geodata(households_per_income_class, param, path_data):
    """Import number of jobs per selected employment center."""
    # Number of jobs per Transport Zone (TZ)
    TAZ = pd.read_csv(path_data + 'TAZ_amp_2013_proj_centro2.csv')

    # Number of employees in each TZ for the 12 income classes
    # NB: we break income classes given by CoCT to stick better to equivalent
    # from census data, as we will reweight towards the end
    jobsCenters12Class = np.array(
        [np.zeros(len(TAZ.Ink1)), TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink1/3,
         TAZ.Ink2/2, TAZ.Ink2/2, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink3/3,
         TAZ.Ink4/3, TAZ.Ink4/3, TAZ.Ink4/3]
        )

    # We get geographic coordinates
    codeCentersInitial = TAZ.TZ2013
    xCoord = TAZ.X / 1000
    yCoord = TAZ.Y / 1000

    # We arbitrarily set the threshold at 2,500 jobs
    selectedCenters = (
        sum(jobsCenters12Class, 0) > param["job_center_threshold"])

    # Corrections where we don't have reliable transport data
    selectedCenters[xCoord > -10] = np.zeros(1, 'bool')
    selectedCenters[yCoord > -3719] = np.zeros(1, 'bool')
    selectedCenters[(xCoord > -20) & (yCoord > -3765)] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1010] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1012] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1394] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1499] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 4703] = np.zeros(1, 'bool')

    # Number of workers per group for the selected centers
    jobsCentersNgroup = np.zeros((len(xCoord), param["nb_of_income_classes"]))
    for j in range(0, param["nb_of_income_classes"]):
        jobsCentersNgroup[:, j] = np.sum(
            jobsCenters12Class[param["income_distribution"] == j + 1, :], 0)

    # jobsCentersNgroup = jobsCentersNgroup[selectedCenters, :]
    # Rescale (wrt census data) to keep the correct global income distribution
    # More specifically, this allows to go from individual to household level
    jobsCentersNGroupRescaled = (
        jobsCentersNgroup * households_per_income_class[None, :]
        / np.nansum(jobsCentersNgroup, 0)
        )

    jobsTable = pd.DataFrame()
    # jobsTable["code"] = codeCentersInitial
    jobsTable["x"] = xCoord
    jobsTable["y"] = yCoord
    jobsTable["poor_jobs"] = jobsCentersNGroupRescaled[:, 0]
    jobsTable["midpoor_jobs"] = jobsCentersNGroupRescaled[:, 1]
    jobsTable["midrich_jobs"] = jobsCentersNGroupRescaled[:, 2]
    jobsTable["rich_jobs"] = jobsCentersNGroupRescaled[:, 3]
    selected_centers = pd.DataFrame(selectedCenters)
    selected_centers = selected_centers.rename(
        columns={selected_centers.columns[0]: 'selected_centers'})

    return jobsTable, selected_centers


def export_housing_types(
        housing_type_1, housing_type_2,
        legend1, legend2, path_plots, path_tables):
    """Bar plot for equilibrium output across housing types."""
    figure, axis = plt.subplots(1, 1, figsize=(10, 7))
    # figure.tight_layout()
    data = pd.DataFrame(
        {legend1: np.nansum(housing_type_1, 1), legend2: housing_type_2},
        index=["Formal private", "Informal \n backyards",
               "Informal \n settlements", "Formal subsidized"])
    data.plot(kind="bar", ax=axis)
    axis.set_ylabel("Households", labelpad=15)
    axis.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    axis.tick_params(labelrotation=0)

    figure.savefig(path_plots + 'validation_housing_type.png')
    plt.close(figure)

    data.to_csv(path_tables + 'validation_housing_type.png')
    print('validation_housing_type done')

    return data


def export_households(
        initial_state_households, households_per_income_and_housing,
        legend1, legend2, path_plots, path_tables):
    """Bar plot for equilibrium output across housing and income groups."""
    # We apply same reweighting as in equilibrium to match aggregate
    # SAL data
    ratio = (np.nansum(initial_state_households)
             / np.nansum(households_per_income_and_housing))
    households_per_income_and_housing = (
        households_per_income_and_housing * ratio)

    # We take RDP out of validation data for formal private
    households_per_income_and_housing[0, 0] = np.max(
        households_per_income_and_housing[0, 0]
        - np.nansum(initial_state_households[3, :, :]),
        0)
    # NB: note that we do not plot RDP as we do not have its breakdown
    # across income groups (and we assumed only the poorest were eligible)
    # TODO: make sure that formal private does not contain formal backyards

    data0 = pd.DataFrame(
        {legend1: np.nansum(initial_state_households[0, :, :], 1),
         legend2: households_per_income_and_housing[0, :]},
        index=["Poor", "Mid-poor", "Mid-rich", "Rich"])
    data1 = pd.DataFrame(
        {legend1: np.nansum(initial_state_households[1, :, :], 1),
         legend2: households_per_income_and_housing[1, :]},
        index=["Poor", "Mid-poor", "Mid-rich", "Rich"])
    data2 = pd.DataFrame(
        {legend1: np.nansum(initial_state_households[2, :, :], 1),
         legend2: households_per_income_and_housing[2, :]},
        index=["Poor", "Mid-poor", "Mid-rich", "Rich"])

    figure, axis = plt.subplots(3, 1, figsize=(10, 7))
    # figure.tight_layout()
    data0.plot(kind="bar", ax=axis[0])
    axis[0].set_title("Formal private")
    # axis[0].get_legend().remove()
    axis[0].set_xticks([])
    axis[0].yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    data1.plot(kind="bar", ax=axis[1])
    axis[1].set_title("Informal backyards")
    # axis[1].get_legend().remove()
    axis[1].set_ylabel("Households", labelpad=15)
    axis[1].set_xticks([])
    axis[1].yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    data2.plot(kind="bar", ax=axis[2])
    axis[2].set_title("Informal settlements")
    axis[2].tick_params(labelrotation=0)
    axis[2].yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    figure.savefig(path_plots + 'validation_housing_per_income.png')
    plt.close(figure)

    data0.to_csv(path_tables + 'validation_formal_per_income.csv')
    data1.to_csv(path_tables + 'validation_backyard_per_income.csv')
    data2.to_csv(path_tables + 'validation_informal_per_income.csv')
    print('validation_housing_per_income done')

    return data0, data1, data2


def validation_density(
        grid, initial_state_households_housing_types, housing_types,
        path_plots, path_tables):
    """Line plot for household density across space in 1D."""
    # Note that formal data here includes RDP
    sum_housing_types = (housing_types.informal_grid
                         + housing_types.formal_grid
                         + housing_types.backyard_informal_grid)

    # Population
    xData = grid.dist
    yData = sum_housing_types / 0.25
    ySimul = np.nansum(
        initial_state_households_housing_types, 0) / 0.25

    df = pd.DataFrame(data=np.transpose(
        np.array([xData, yData, ySimul])), columns=["x", "yData", "ySimul"])
    df["round"] = round(df.x)
    new_df = df.groupby(['round']).mean()
    q1_df = df.groupby(['round']).quantile(0.25)
    q3_df = df.groupby(['round']).quantile(0.75)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.yData, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.ySimul, color="green", label="Simulation")
    # axes = plt.axes()
    ax.set_ylim([0, 1700])
    ax.set_xlim([0, 50])
    ax.fill_between(np.arange(
        max(df["round"] + 1)), q1_df.ySimul, q3_df.ySimul, color="lightgreen",
        label="Simul. interquart. range")
    ax.fill_between(np.arange(
        max(df["round"] + 1)), q1_df.yData, q3_df.yData, color="lightgrey",
        alpha=0.5, label="Data. interquart. range")
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Average households density (per km²)", labelpad=15)
    plt.tick_params(bottom=True, labelbottom=True)
    plt.tick_params(labelbottom=True)
    # plt.title("Population density")
    plt.savefig(path_plots + 'validation_density.png')
    plt.close()

    df.to_csv(path_tables + 'validation_density.csv')
    print('validation_density done')

    return df


def validation_density_housing_types(
        grid, initial_state_households_housing_types, housing_types,
        path_plots, path_tables):
    """Line plot for number of households per housing type across 1D-space."""
    # Housing types
    xData = grid.dist
    #  Here, we take RDP out of formal housing to plot formal private
    formal_data = (housing_types.formal_grid
                   - initial_state_households_housing_types[3, :])
    backyard_data = housing_types.backyard_informal_grid
    informal_data = housing_types.informal_grid
    formal_simul = initial_state_households_housing_types[0, :]
    informal_simul = initial_state_households_housing_types[2, :]
    backyard_simul = initial_state_households_housing_types[1, :]
    # Note that RDP population is not actually simulated but taken from data
    rdp_simul = initial_state_households_housing_types[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, formal_data, backyard_data, informal_data, formal_simul,
             backyard_simul, informal_simul, rdp_simul]
            )),
        columns=["xData", "formal_data", "backyard_data", "informal_data",
                 "formal_simul", "backyard_simul", "informal_simul",
                 "rdp_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    # plt.figure(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_simul, color="green", label="Simulation")
    # axes = plt.axes()
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    # plt.title("Formal")
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (formal private)", labelpad=15)
    plt.savefig(path_plots + 'validation_density_formal.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    # plt.title("Informal")
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (informal settlements)",
               labelpad=15)
    plt.legend()
    plt.tick_params(labelbottom=True)
    # plt.xticks(
    #     [10.5, 13, 16, 18, 24, 25, 27, 30, 37, 39, 46.5],
    #     ["Joe Slovo", "Hout Bay", "Du Noon", "Philippi", "Khayelitsa",
    #      "Wallacedene", "Khayelitsa", "Witsand", "Enkanini", "Pholile"],
    #     rotation='vertical')
    plt.savefig(path_plots + 'validation_density_informal.png')
    plt.close()

    # print("1")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    # plt.title("Backyard")
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (informal backyards)",
               labelpad=15)
    plt.savefig(path_plots + 'validation_density_backyard.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rdp_simul, color="black")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)")
    plt.ylabel("Total number of households (formal subsidized)",
               labelpad=15)
    plt.savefig(path_plots + 'validation_density_rdp.png')
    plt.close()

    df.to_csv(path_tables + 'validation_density_per_housing.csv')
    print('validation_density_per_housing done')

    return df


def validation_density_income_groups(
        grid, initial_state_household_centers, income_distribution_grid,
        path_plots, path_tables):
    """Line plot for number of households per income group across 1D-space."""
    # We apply same reweighting as in equilibrium to match aggregate
    # SAL data
    ratio = (np.nansum(initial_state_household_centers)
             / np.nansum(income_distribution_grid))
    income_distribution_grid = (
        income_distribution_grid * ratio)

    # Income groups
    xData = grid.dist
    poor_data = income_distribution_grid[0, :]
    midpoor_data = income_distribution_grid[1, :]
    midrich_data = income_distribution_grid[2, :]
    rich_data = income_distribution_grid[3, :]
    poor_simul = initial_state_household_centers[0, :]
    midpoor_simul = initial_state_household_centers[1, :]
    midrich_simul = initial_state_household_centers[2, :]
    rich_simul = initial_state_household_centers[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, poor_data, midpoor_data, midrich_data, rich_data,
             poor_simul, midpoor_simul, midrich_simul, rich_simul]
            )),
        columns=["xData", "poor_data", "midpoor_data", "midrich_data",
                 "rich_data", "poor_simul", "midpoor_simul", "midrich_simul",
                 "rich_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.poor_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.poor_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (poor)", labelpad=15)
    plt.savefig(path_plots + 'validation_density_poor.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midpoor_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midpoor_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (mid-poor)", labelpad=15)
    plt.savefig(path_plots + 'validation_density_midpoor.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midrich_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midrich_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (mid-rich)", labelpad=15)
    plt.savefig(path_plots + 'validation_density_midrich.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rich_data, color="black", label="Data")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rich_simul, color="green", label="Simulation")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (rich)", labelpad=15)
    plt.savefig(path_plots + 'validation_density_rich.png')
    plt.close()

    df.to_csv(path_tables + 'validation_density_per_income.csv')
    print('validation_density_per_income done')

    return df


def validation_density_housing_and_income_groups(
        grid, initial_state_households, path_plots, path_tables):
    """Plot number of HHs per housing and income group across 1D-space."""
    # Housing and income groups
    xData = grid.dist

    formal_poor_simul = initial_state_households[0, 0, :]
    formal_midpoor_simul = initial_state_households[0, 1, :]
    formal_midrich_simul = initial_state_households[0, 2, :]
    formal_rich_simul = initial_state_households[0, 3, :]

    backyard_poor_simul = initial_state_households[1, 0, :]
    backyard_midpoor_simul = initial_state_households[1, 1, :]

    informal_poor_simul = initial_state_households[2, 0, :]
    informal_midpoor_simul = initial_state_households[2, 1, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, formal_poor_simul, formal_midpoor_simul,
             formal_midrich_simul, formal_rich_simul,
             backyard_poor_simul, backyard_midpoor_simul,
             informal_poor_simul, informal_midpoor_simul]
            )),
        columns=["xData", "formal_poor_simul", "formal_midpoor_simul",
                 "formal_midrich_simul", "formal_rich_simul",
                 "backyard_poor_simul", "backyard_midpoor_simul",
                 "informal_poor_simul", "informal_midpoor_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_poor_simul, color="darkblue", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_midpoor_simul, color="darkgreen", label="Mid-poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_midrich_simul, color="gold", label="Mid-rich")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_rich_simul, color="darkorange", label="Rich")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (formal private)", labelpad=15)
    plt.savefig(path_plots + 'validation_density_FP_income.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_poor_simul, color="darkblue", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_midpoor_simul, color="darkgreen", label="Mid-poor")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (informal backyards)", labelpad=15)
    plt.savefig(path_plots + 'validation_density_IB_income.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_poor_simul, color="darkblue", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_midpoor_simul, color="darkgreen", label="Mid-poor")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total number of households (informal settlements)",
               labelpad=15)
    plt.savefig(path_plots + 'validation_density_IS_income.png')
    plt.close()

    df.to_csv(path_tables + 'validation_density_per_housing_and_income.csv')
    print('validation_density_per_housing_and_income done')

    return df


def plot_income_net_of_commuting_costs(
        grid, income_net_of_commuting_costs, path_plots, path_tables):
    """Plot avg income net of commuting costs across 1D-space."""
    # Housing and income groups
    xData = grid.dist

    poor_simul = income_net_of_commuting_costs[0, :]
    midpoor_simul = income_net_of_commuting_costs[1, :]
    midrich_simul = income_net_of_commuting_costs[2, :]
    rich_simul = income_net_of_commuting_costs[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, poor_simul, midpoor_simul,
             midrich_simul, rich_simul]
            )),
        columns=["xData", "poor_simul", "midpoor_simul",
                 "midrich_simul", "rich_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.poor_simul, color="darkblue", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midpoor_simul, color="darkgreen", label="Mid-poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midrich_simul, color="gold", label="Mid-rich")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rich_simul, color="darkorange", label="Rich")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Estimated average incomes net of commuting costs", labelpad=15)
    plt.savefig(path_plots + 'avg_income_net_of_commuting_1d.png')
    plt.close()

    df.to_csv(path_tables + 'avg_income_net_of_commuting_1d.csv')
    print('avg_income_net_of_commuting_1d done')

    return df


def plot_average_income(
        grid, average_income, path_plots, path_tables):
    """Plot average income across 1D-space."""
    # Housing and income groups
    xData = grid.dist

    poor_simul = average_income[0, :]
    midpoor_simul = average_income[1, :]
    midrich_simul = average_income[2, :]
    rich_simul = average_income[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, poor_simul, midpoor_simul,
             midrich_simul, rich_simul]
            )),
        columns=["xData", "poor_simul", "midpoor_simul",
                 "midrich_simul", "rich_simul"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.poor_simul, color="darkblue", label="Poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midpoor_simul, color="darkgreen", label="Mid-poor")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.midrich_simul, color="gold", label="Mid-rich")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rich_simul, color="darkorange", label="Rich")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Estimated average incomes", labelpad=15)
    plt.savefig(path_plots + 'avg_income_1d.png')
    plt.close()

    df.to_csv(path_tables + 'avg_income_1d.csv')
    print('avg_income_1d done')

    return df


def validate_average_income(
        grid, overall_avg_income, data_avg_income,
        path_plots, path_tables):
    """Validate overall average income across 1D-space."""
    # Housing and income groups
    xData = grid.dist

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, overall_avg_income, data_avg_income]
            )),
        columns=["xData", "income_simul", "income_data"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.income_simul, color="green", label="Simulation")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.income_data, color="black", label="Data")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Overall average incomes", labelpad=15)
    plt.savefig(path_plots + 'overall_avg_income_valid_1d.png')
    plt.close()

    df.to_csv(path_tables + 'overall_avg_income_valid_1d.csv')
    print('overall_avg_income_valid_1d done')

    return df


def plot_housing_supply(grid, initial_state_housing_supply, path_plots,
                        path_tables):
    """Line plot of avg housing supply per type and unit of available land."""
    xData = grid.dist
    formal_simul = initial_state_housing_supply[0, :]
    backyard_simul = initial_state_housing_supply[1, :]
    informal_simul = initial_state_housing_supply[2, :]
    rdp_simul = initial_state_housing_supply[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, formal_simul, backyard_simul, informal_simul,
             rdp_simul]
            )),
        columns=["xData", "formal_simul", "backyard_simul", "informal_simul",
                 "rdp_simul"]
        )
    df["round"] = round(df.xData)
    # NB: we take average instead of sum to be able to plot all housing types
    # in the same graph
    new_df = df.groupby(['round']).mean()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_simul, color="darkorange", label="Formal")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_simul, color="gold", label="Backyard")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_simul, color="darkgreen", label="Informal")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rdp_simul, color="darkblue", label="RDP")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Avg housing supply (in m² per km² of available land)",
               labelpad=15)
    plt.savefig(path_plots + 'validation_housing_supply.png')
    plt.close()

    df.to_csv(path_tables + 'validation_housing_supply.csv')
    print('validation_housing_supply done')

    return df


def plot_housing_supply_noland(grid, housing_supply, path_plots,
                               path_tables):
    """Line plot of total housing supply per type across 1D-space."""
    xData = grid.dist
    formal_simul = housing_supply[0, :]
    backyard_simul = housing_supply[1, :]
    informal_simul = housing_supply[2, :]
    rdp_simul = housing_supply[3, :]

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, formal_simul, backyard_simul, informal_simul,
             rdp_simul]
            )),
        columns=["xData", "formal_simul", "backyard_simul", "informal_simul",
                 "rdp_simul"]
        )
    df["round"] = round(df.xData)
    # NB: we take average instead of sum to be able to plot all housing types
    # in the same graph
    new_df = df.groupby(['round']).mean()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.formal_simul, color="darkorange", label="Formal")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.backyard_simul, color="gold", label="Backyard")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.informal_simul, color="darkgreen", label="Informal")
    ax.plot(np.arange(max(df["round"] + 1)),
            new_df.rdp_simul, color="darkblue", label="RDP")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    plt.ylabel("Total housing supply (in m²)",
               labelpad=15)
    plt.savefig(path_plots + 'validation_housing_supply_noland.png')
    plt.close()

    df.to_csv(path_tables + 'validation_housing_supply_noland.csv')
    print('validation_housing_supply_noland done')

    return df


def validation_housing_price(
        grid, initial_state_rent, interest_rate, param, center,
        housing_types_sp, data_sp,
        path_plots, path_tables, land_price):
    """Plot land price per housing type across space."""
    sp_x = housing_types_sp["x_sp"]
    sp_y = housing_types_sp["y_sp"]
    sp_price = data_sp["price"]

    if land_price == 1:
        priceSimul = (
            ((initial_state_rent[0:3, :] * param["coeff_A"])
             / (interest_rate + param["depreciation_rate"]))
            ** (1 / param["coeff_a"])
            * param["coeff_a"]
            * param["coeff_b"] ** (param["coeff_b"] / param["coeff_a"])
            )
    elif land_price == 0:
        priceSimul = initial_state_rent

    priceSimulPricePoints_formal = griddata(
        np.transpose(np.array([grid.x, grid.y])),
        priceSimul[0, :],
        np.transpose(np.array([sp_x, sp_y]))
        )
    priceSimulPricePoints_informal = griddata(
        np.transpose(np.array([grid.x, grid.y])),
        priceSimul[1, :],
        np.transpose(np.array([sp_x, sp_y]))
        )
    priceSimulPricePoints_backyard = griddata(
        np.transpose(np.array([grid.x, grid.y])),
        priceSimul[2, :],
        np.transpose(np.array([sp_x, sp_y]))
        )

    xData = np.sqrt((sp_x - center[0]) ** 2 + (sp_y - center[1]) ** 2)
    # TODO: check need to redefine
    if land_price == 1:
        yData = sp_price
    elif land_price == 0:
        yData = (sp_price ** param["coeff_a"]
                 / (param["coeff_a"] ** param["coeff_a"]
                    * param["coeff_b"]**param["coeff_b"])
                 * (interest_rate + param["depreciation_rate"])
                 / param["coeff_A"]
                 )
    # xSimulation = xData
    ySimulation = priceSimulPricePoints_formal
    informalSimul = priceSimulPricePoints_informal
    backyardSimul = priceSimulPricePoints_backyard

    df = pd.DataFrame(
        data=np.transpose(np.array([xData, yData, ySimulation, informalSimul,
                                    backyardSimul])),
        columns=["xData", "yData", "ySimulation", "informalSimul",
                 "backyardSimul"])
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    which = ~np.isnan(new_df.yData) & ~np.isnan(new_df.ySimulation)
    which_informal = ~np.isnan(new_df.informalSimul)
    which_backyard = ~np.isnan(new_df.backyardSimul)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(new_df.xData[which], new_df.yData[which],
            color="black", label="Data")
    ax.plot(new_df.xData[which], new_df.ySimulation[which],
            color="green", label="Simul")
    ax.plot(new_df.xData[which_informal], new_df.informalSimul[which_informal],
            color="red", label="Informal")
    ax.plot(new_df.xData[which_backyard], new_df.backyardSimul[which_backyard],
            color="blue", label="Backyard")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    if land_price == 1:
        plt.ylabel("Land price (R/m² of land)", labelpad=15)
    if land_price == 0:
        plt.ylabel("Housing price (R/m² of housing)", labelpad=15)
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.tick_params(bottom=True, labelbottom=True)
    plt.savefig(path_plots + '/validation_housing_price'
                + str(land_price) + '.png')
    plt.close()

    df.to_csv(path_tables + 'validation_housing_price'
              + str(land_price) + '.csv')
    print('validation_housing_price' + str(land_price) + ' done')

    return df, yData


def validation_housing_price_test(
        grid, initial_state_rent, interest_rate, param, center,
        housing_types_sp, data_sp,
        path_plots, path_tables, land_price):
    """Plot land price per housing type across space."""
    sp_x = housing_types_sp["x_sp"]
    sp_y = housing_types_sp["y_sp"]
    sp_price = data_sp["price"]

    if land_price == 1:
        priceSimul = (
            ((initial_state_rent[0:3, :] * param["coeff_A"])
             / (interest_rate + param["depreciation_rate"]))
            ** (1 / param["coeff_a"])
            * param["coeff_a"]
            * param["coeff_b"] ** (param["coeff_b"] / param["coeff_a"])
            )
    elif land_price == 0:
        priceSimul = initial_state_rent

    # TODO: check need to redefine
    if land_price == 1:
        yData = sp_price
    elif land_price == 0:
        yData = (sp_price ** param["coeff_a"]
                 / (param["coeff_a"] ** param["coeff_a"]
                    * param["coeff_b"]**param["coeff_b"])
                 * (interest_rate + param["depreciation_rate"])
                 / param["coeff_A"]
                 )
    xData = grid.dist
    # TODO: check coordinate issues and lack of data in center
    yData = griddata(
        np.transpose(np.array([sp_x, sp_y])),
        yData,
        np.transpose(np.array([grid.x, grid.y]))
        )
    ySimulation = priceSimul[0, :]
    informalSimul = priceSimul[1, :]
    backyardSimul = priceSimul[2, :]

    df = pd.DataFrame(
        data=np.transpose(np.array([xData, yData, ySimulation, informalSimul,
                                    backyardSimul])),
        columns=["xData", "yData", "ySimulation", "informalSimul",
                 "backyardSimul"])
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()
    # test_df = df.groupby(['round']).agg({'yData': np.nanmean,
    #                                 'ySimulation': np.nanmean,
    #                                 'informalSimul': np.nanmean,
    #                                 'backyardSimul': np.nanmean}).reset_index()

    which_data = ~np.isnan(new_df.yData)
    which_simul = ~np.isnan(new_df.ySimulation)
    which_informal = ~np.isnan(new_df.informalSimul)
    which_backyard = ~np.isnan(new_df.backyardSimul)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(new_df.xData[which_data], new_df.yData[which_data],
            color="black", label="Data")
    ax.plot(new_df.xData[which_simul], new_df.ySimulation[which_simul],
            color="green", label="Simul")
    ax.plot(new_df.xData[which_informal], new_df.informalSimul[which_informal],
            color="red", label="Informal")
    ax.plot(new_df.xData[which_backyard], new_df.backyardSimul[which_backyard],
            color="blue", label="Backyard")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.xlabel("Distance to the city center (km)", labelpad=15)
    if land_price == 1:
        plt.ylabel("Land price (R/m² of land)", labelpad=15)
    if land_price == 0:
        plt.ylabel("Housing price (R/m² of housing)", labelpad=15)
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.tick_params(bottom=True, labelbottom=True)
    plt.savefig(path_plots + '/validation_housing_price'
                + str(land_price) + '.png')
    plt.close()

    df.to_csv(path_tables + 'validation_housing_price'
              + str(land_price) + '.csv')
    print('validation_housing_price' + str(land_price) + ' done')

    return df, yData

def plot_housing_demand(grid, center, initial_state_dwelling_size,
                        path_precalc_inp, path_outputs):
    """d."""
    data = scipy.io.loadmat(path_precalc_inp + 'data.mat')['data']
    # Number of informal settlements in backyard per grid cell
    sp_x = data['spX'][0][0].squeeze()
    sp_y = data['spY'][0][0].squeeze()
    sp_size = data['spDwellingSize'][0][0].squeeze()

    sizeSimulPoints = griddata(
        np.transpose(np.array([grid.x, grid.y])),
        initial_state_dwelling_size[0, :],
        np.transpose(np.array([sp_x, sp_y]))
        )

    xData = np.sqrt((sp_x - center[0]) ** 2 + (sp_y - center[1]) ** 2)
    yData = sp_size
    # xSimulation = xData
    ySimulation = sizeSimulPoints

    df = pd.DataFrame(
        data=np.transpose(np.array(
            [xData, yData, ySimulation]
            )),
        columns=["xData", "yData", "ySimulation"]
        )
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    which = ~np.isnan(new_df.yData) & ~np.isnan(new_df.ySimulation)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(new_df.xData[which], new_df.yData[which],
            color="black", label="Data")
    ax.plot(new_df.xData[which], new_df.ySimulation[which],
            color="green", label="Simul")

    ax.set_ylim(0, 500)
    ax.set_xlim([0, 40])
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.xlabel("Distance to the city center (km)")
    plt.ylabel("Avg dwelling size in formal sector (in m²)")
    plt.savefig(path_outputs + 'validation_housing_demand.png')
    plt.close()

