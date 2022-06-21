# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:32:48 2020.

@author: Charlotte Liotta
"""

import scipy
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

import inputs.data as inpdt


def export_housing_types(
        housing_type_1, housing_type_2,
        legend1, legend2, path_plots):
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

    return data


def export_households(
        initial_state_households, households_per_income_and_housing,
        legend1, legend2, path_plots):
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

    return data0, data1, data2


def validation_density(
        grid, initial_state_households_housing_types, housing_types,
        path_plots):
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

    return df


def validation_density_housing_types(
        grid, initial_state_households_housing_types, housing_types,
        path_plots):
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

    return df


def validation_density_income_groups(
        grid, initial_state_household_centers, income_distribution_grid,
        path_plots):
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

    return df


def validation_density_housing_and_income_groups(
        grid, initial_state_households, path_plots):
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

    return df


def plot_income_net_of_commuting_costs(
        grid, income_net_of_commuting_costs, path_plots):
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

    return df


def plot_average_income(
        grid, average_income, path_plots):
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

    return df


def export_map(value, grid, export_name, title, ubnd, lbnd=0, cmap='Reds'):
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


def plot_housing_supply(grid, initial_state_housing_supply, path_plots):
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

    return df


def plot_housing_supply_noland(grid, housing_supply, path_plots):
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

    return df


def validation_housing_price(
        grid, initial_state_rent, interest_rate, param, center,
        precalculated_inputs, path_outputs):
    """d."""
    data = scipy.io.loadmat(path_precalc_inp + 'data.mat')['data']
    # Number of informal settlements in backyard per grid cell
    sp_x = data['spX'][0][0].squeeze()
    sp_y = data['spY'][0][0].squeeze()
    sp_price = data['spPrice'][0][0].squeeze()[2, :]

    priceSimul = (
        ((initial_state_rent[0:3, :] * param["coeff_A"])
         / (interest_rate + param["depreciation_rate"]))
        ** (1 / param["coeff_a"])
        * param["coeff_a"]
        * param["coeff_b"] ** (param["coeff_b"] / param["coeff_a"])
        )

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
    # TODO: a priori, no need to redefine
    yData = sp_price
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
    plt.xlabel("Distance to the city center (km)")
    plt.ylabel("Price (R/m² of land)")
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.tick_params(bottom=True, labelbottom=True)
    plt.savefig(path_outputs + '/validation_housing_price.png')
    plt.close()

# %% Validation



def error_map(error, grid, export_name):
    """d."""
    Map = plt.scatter(grid.x,
                      grid.y,
                      s=None,
                      c=error,
                      cmap='RdYlGn',
                      marker='.')
    plt.colorbar(Map)
    plt.axis('off')
    plt.clim(-100, 100)
    plt.savefig(export_name)
    plt.close()




# TODO: Not used in plots.py
def export_density_rents_sizes(
        grid, data_rdp, housing_types_grid,
        initial_state_households_housing_types, initial_state_dwelling_size,
        initial_state_rent, simul1_households_housing_type, simul1_rent,
        simul1_dwelling_size, dwelling_size_sp, path_outputs):
    """d."""
    # 1. Housing types

    count_formal = housing_types_grid.formal_grid - data_rdp["count"]
    count_formal[count_formal < 0] = 0

    try:
        os.mkdir(path_outputs + 'housing_types')
    except OSError as error:
        print(error)

    # Formal
    error = (
        initial_state_households_housing_types[0, :] / count_formal - 1) * 100
    error_map(
        error, grid, path_outputs + 'housing_types/formal_diff_with_data.png')
    export_map(
        count_formal, grid, path_outputs + 'housing_types/formal_data.png',
        1200)
    export_map(
        initial_state_households_housing_types[0, :], grid,
        path_outputs + 'housing_types/formal_simul.png', 1200)
    export_map(
        simul1_households_housing_type[0, 0, :], grid,
        path_outputs + 'housing_types/formal_Basile1.png', 1200)

    # Subsidized
    error = (
        initial_state_households_housing_types[3, :] / data_rdp["count"] - 1
        ) * 100
    error_map(error, grid, path_outputs +
              'housing_types/subsidized_diff_with_data.png')
    export_map(data_rdp["count"], grid, path_outputs +
               'housing_types/subsidized_data.png', 1200)
    export_map(initial_state_households_housing_types[3, :], grid,
               path_outputs + 'housing_types/subsidized_simul.png', 1200)
    export_map(simul1_households_housing_type[0, 3, :], grid,
               path_outputs + 'housing_types/subsidized_Basile1.png', 1200)

    # Informal
    error = (
        initial_state_households_housing_types[2, :]
        / housing_types_grid.informal_grid - 1) * 100
    error_map(error, grid, path_outputs +
              'housing_types/informal_diff_with_data.png')
    export_map(housing_types_grid.informal_grid, grid,
               path_outputs + 'housing_types/informal_data.png', 800)
    export_map(initial_state_households_housing_types[2, :], grid,
               path_outputs + 'housing_types/informal_simul.png', 800)
    export_map(simul1_households_housing_type[0, 2, :], grid,
               path_outputs + 'housing_types/informal_Basile1.png', 800)

    # Backyard: note that we actually have formal and informal backyarding
    error = (initial_state_households_housing_types[1, :] /
             housing_types_grid.backyard_informal_grid - 1) * 100
    error_map(error, grid, path_outputs +
              'housing_types/backyard_diff_with_data.png')
    export_map(housing_types_grid.backyard_informal_grid, grid,
               path_outputs + 'housing_types/backyard_data.png', 800)
    export_map(initial_state_households_housing_types[1, :], grid,
               path_outputs + 'housing_types/backyard_simul.png', 800)
    export_map(simul1_households_housing_type[0, 1, :], grid,
               path_outputs + 'housing_types/backyard_Basile1.png', 800)

    # 2. Dwelling size

    try:
        os.mkdir(path_outputs + 'dwelling_size')
    except OSError as error:
        print(error)

    # TODO: Check if deprecated
    dwelling_size = inpdt.SP_to_grid_2011_1(
        dwelling_size_sp, grid)

    # Data
    export_map(dwelling_size, grid, path_outputs +
               'dwelling_size/data.png', 300)

    # Class 1
    error = (initial_state_dwelling_size[0, :] / dwelling_size - 1) * 100
    error_map(error, grid, path_outputs +
              'dwelling_size/class1_diff_with_data.png')
    export_map(initial_state_dwelling_size[0, :], grid,
               path_outputs + 'dwelling_size/class1_simul.png', 300)
    export_map(simul1_dwelling_size[0, 0, :], grid,
               path_outputs + 'dwelling_size/class1_Basile1.png', 300)

    # Class 2
    error = (initial_state_dwelling_size[1, :] / dwelling_size - 1) * 100
    error_map(error, grid, path_outputs +
              'dwelling_size/class2_diff_with_data.png')
    export_map(initial_state_dwelling_size[1, :], grid,
               path_outputs + 'dwelling_size/class2_simul.png', 200)
    export_map(simul1_dwelling_size[0, 1, :], grid,
               path_outputs + 'dwelling_size/class2_Basile1.png', 200)

    # Class 3
    error = (initial_state_dwelling_size[2, :] / dwelling_size - 1) * 100
    error_map(error, grid, path_outputs +
              'dwelling_size/class3_diff_with_data.png')
    export_map(initial_state_dwelling_size[2, :], grid,
               path_outputs + 'dwelling_size/class3_simul.png', 200)
    export_map(simul1_dwelling_size[0, 2, :], grid,
               path_outputs + 'dwelling_size/class3_Basile1.png', 200)

    # Class 4
    error = (initial_state_dwelling_size[3, :] / dwelling_size - 1) * 100
    error_map(error, grid, path_outputs +
              'dwelling_size/class4_diff_with_data.png')
    export_map(initial_state_dwelling_size[3, :], grid,
               path_outputs + 'dwelling_size/class4_simul.png', 100)
    export_map(simul1_dwelling_size[0, 3, :], grid,
               path_outputs + 'dwelling_size/class4_Basile1.png', 100)

    # 3. Rents

    try:
        os.mkdir(path_outputs + '/rents')
    except OSError as error:
        print(error)

    # Class 1
    export_map(initial_state_rent[0, :], grid,
               path_outputs + 'rents/class1_simul.png', 800)
    export_map(simul1_rent[0, 0, :], grid,
               path_outputs + 'rents/class1_Basile1.png', 800)

    # Class 2
    export_map(initial_state_rent[1, :], grid,
               path_outputs + 'rents/class2_simul.png', 700)
    export_map(simul1_rent[0, 1, :], grid,
               path_outputs + 'rents/class2_Basile1.png', 700)

    # Class 3
    export_map(initial_state_rent[2, :], grid,
               path_outputs + 'rents/class3_simul.png', 600)
    export_map(simul1_rent[0, 2, :], grid,
               path_outputs + 'rents/class3_Basile1.png', 600)

    # Class 4
    export_map(initial_state_rent[3, :], grid,
               path_outputs + 'rents/class4_simul.png', 500)
    export_map(simul1_rent[0, 3, :], grid,
               path_outputs + 'rents/class4_Basile1.png', 500)










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


def validation_cal_income(path_data, path_outputs, center,
                          income_centers_w, income_centers_precalc_w):
    """d."""
    TAZ = pd.read_csv(path_data + 'TAZ_amp_2013_proj_centro2.csv')

    jobsCenters12Class = np.array(
        [np.zeros(len(TAZ.Ink1)), TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink1/3,
         TAZ.Ink2/2, TAZ.Ink2/2, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink3/3,
         TAZ.Ink4/3, TAZ.Ink4/3, TAZ.Ink4/3]
        )

    codeCentersInitial = TAZ.TZ2013
    xCoord = TAZ.X / 1000
    yCoord = TAZ.Y / 1000

    selectedCenters = sum(jobsCenters12Class, 0) > 2500
    selectedCenters[xCoord > -10] = np.zeros(1, 'bool')
    selectedCenters[yCoord > -3719] = np.zeros(1, 'bool')
    selectedCenters[(xCoord > -20) & (yCoord > -3765)] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1010] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1012] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1394] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 1499] = np.zeros(1, 'bool')
    selectedCenters[codeCentersInitial == 4703] = np.zeros(1, 'bool')

    xCenter = xCoord[selectedCenters]
    yCenter = yCoord[selectedCenters]

    xData = np.sqrt((xCenter - center[0]) ** 2 + (yCenter - center[1]) ** 2)
    yData = income_centers_precalc_w
    ySimul = income_centers_w

    df = pd.DataFrame(
        data=np.transpose(np.array([xData, yData, ySimul])),
        columns=["xData", "yData", "ySimul"])
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    which = ~np.isnan(new_df.yData) & ~np.isnan(new_df.ySimul)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(new_df.xData[which], new_df.yData[which],
            color="black", label="Precalc.")
    ax.plot(new_df.xData[which], new_df.ySimul[which],
            color="green", label="Recalc.")
    ax.set_ylim(0)
    ax.set_xlim([0, 50])
    plt.xlabel("Distance to the city center (km)")
    plt.ylabel("Normalized income per job center (rich)")
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.tick_params(bottom=True, labelbottom=True)
    plt.savefig(path_outputs + '/validation_rich_income.png')
    plt.close()


def plot_diagnosis_map_informl(
        grid, coeff_land, initial_state_households_housing_types, path_outputs
        ):
    """d."""
    plt.scatter(grid.x, grid.y, color="lightgrey")
    plt.scatter(grid.x, grid.y, s=None,
                c=coeff_land[2, :], cmap='Greys', marker='.')
    plt.scatter(
        grid.x[initial_state_households_housing_types[2, :] > 0],
        grid.y[initial_state_households_housing_types[2, :] > 0],
        s=None,
        c=initial_state_households_housing_types[2, :][
            initial_state_households_housing_types[2, :] > 0],
        cmap='Reds',
        marker='.')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(path_outputs + '/diagnosis_informal.png')
    plt.close()
