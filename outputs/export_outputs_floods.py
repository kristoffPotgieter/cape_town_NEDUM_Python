# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:38:21 2020.

@author: Charlotte Liotta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import outputs.flood_outputs as outfld


# %% Floods

def validation_flood(stats1, stats2, legend1, legend2, type_flood,
                     path_plots):
    """Bar plot flood depth and area across some RPs per housing types."""
    label = ["Formal private", "Formal subsidized",
             "Informal \n settlements", "Informal \n backyards"]

    # FLOOD DEPTH

    # First for validation data
    # RP = 20 yrs
    tshirt = [stats1.flood_depth_formal[2], stats1.flood_depth_subsidized[2],
              stats1.flood_depth_informal[2], stats1.flood_depth_backyard[2]]
    # RP = 50 yrs
    tshirtb = [stats1.flood_depth_formal[3], stats1.flood_depth_subsidized[3],
               stats1.flood_depth_informal[3], stats1.flood_depth_backyard[3]]
    # RP = 100 yrs
    # TODO: change name?
    formal_shirt = [
        stats1.flood_depth_formal[5], stats1.flood_depth_subsidized[5],
        stats1.flood_depth_informal[5], stats1.flood_depth_backyard[5]]

    # Then for simulation
    # RP = 20 yrs
    tshirt2 = [stats2.flood_depth_formal[2], stats2.flood_depth_subsidized[2],
               stats2.flood_depth_informal[2], stats2.flood_depth_backyard[2]]
    # RP = 50 yrs
    tshirtb2 = [stats2.flood_depth_formal[3], stats2.flood_depth_subsidized[3],
                stats2.flood_depth_informal[3], stats2.flood_depth_backyard[3]]
    # RP = 100 yrs
    # TODO: change name?
    formal_shirt2 = [
        stats2.flood_depth_formal[5], stats2.flood_depth_subsidized[5],
        stats2.flood_depth_informal[5], stats2.flood_depth_backyard[5]]

    # TODO: check uses
    colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
    r = np.arange(len(label))
    barWidth = 0.25
    plt.figure(figsize=(10, 7))
    plt.bar(r, np.array(tshirt),
            color=colors[1], edgecolor='white', width=barWidth,
            label='20 years')
    # valueb = np.maximum(np.array(tshirtb) - np.array(tshirt),
    # np.full(4, 0.003))
    # floorb = np.maximum(np.array(tshirtb), np.array(tshirt))
    plt.bar(r, np.array(tshirtb) - np.array(tshirt),
            bottom=np.array(tshirt), color=colors[2], edgecolor='white',
            width=barWidth, label='50 years')
    # valuec = np.maximum(np.array(formal_shirt) - floorb, np.full(4, 0.003))
    plt.bar(r, np.array(formal_shirt) - np.array(tshirtb),
            bottom=np.array(tshirtb), color=colors[3], edgecolor='white',
            width=barWidth, label='100 years')
    plt.bar(r + barWidth, np.array(tshirt2),
            color=colors[1], edgecolor='white', width=barWidth)
    # valueb2 = np.maximum(np.array(tshirtb2) - np.array(tshirt2),
    # np.full(4, 0.003))
    # floorb2 = np.maximum(np.array(tshirtb2), np.array(tshirt2))
    plt.bar(r + barWidth, np.array(tshirtb2) - np.array(tshirt2),
            bottom=np.array(tshirt2), color=colors[2], edgecolor='white',
            width=barWidth)
    # valuec2 = np.maximum(np.array(formal_shirt2) - floorb2,
    # np.full(4, 0.003))
    plt.bar(r + barWidth, np.array(formal_shirt2) - np.array(tshirtb2),
            bottom=np.array(tshirtb2), color=colors[3], edgecolor='white',
            width=barWidth)
    plt.legend()
    plt.xticks(r + barWidth/2, label)
    # plt.ylim(0, 1)
    # TODO: check look
    plt.text(r[0] - 0.1,
             np.maximum(
                 stats1.flood_depth_formal[5], stats1.flood_depth_formal[2]
                 ) + 0.002,
             legend1)
    plt.text(r[1] - 0.1,
             np.maximum(
                 stats1.flood_depth_subsidized[5],
                 stats1.flood_depth_subsidized[2]
                 ) + 0.002,
             legend1)
    plt.text(r[2] - 0.1,
             np.maximum(
                 stats1.flood_depth_informal[5], stats1.flood_depth_informal[2]
                 ) + 0.002,
             legend1)
    plt.text(r[3] - 0.1,
             np.maximum(
                 stats1.flood_depth_backyard[5], stats1.flood_depth_backyard[2]
                 ) + 0.002,
             legend1)
    plt.text(r[0] + 0.15,
             np.maximum(
                 stats2.flood_depth_formal[5], stats2.flood_depth_formal[2]
                 ) + 0.002,
             legend2)
    plt.text(r[1] + 0.15,
             np.maximum(
                 stats2.flood_depth_subsidized[5],
                 stats2.flood_depth_subsidized[2]
                 ) + 0.002,
             legend2)
    plt.text(r[2] + 0.15,
             np.maximum(
                 stats2.flood_depth_informal[5], stats2.flood_depth_informal[2]
                 ) + 0.002,
             legend2)
    plt.text(r[3] + 0.15,
             np.maximum(
                 stats2.flood_depth_backyard[5], stats2.flood_depth_backyard[2]
                 ) + 0.002,
             legend2)
    plt.ylabel("Average flood depth (m)", labelpad=15)
    plt.tick_params(labelbottom=True)
    plt.savefig(path_plots + 'validation_flood_depth_' + type_flood + '.png')
    # plt.show()
    plt.close()

    # FLOOD-PRONE AREA
    # TODO: really need to change names...
    jeans = [
        stats1.fraction_formal_in_flood_prone_area[2],
        stats1.fraction_subsidized_in_flood_prone_area[2],
        stats1.fraction_informal_in_flood_prone_area[2],
        stats1.fraction_backyard_in_flood_prone_area[2]]
    tshirt = [
        stats1.fraction_formal_in_flood_prone_area[3],
        stats1.fraction_subsidized_in_flood_prone_area[3],
        stats1.fraction_informal_in_flood_prone_area[3],
        stats1.fraction_backyard_in_flood_prone_area[3]]
    formal_shirt = [
        stats1.fraction_formal_in_flood_prone_area[5],
        stats1.fraction_subsidized_in_flood_prone_area[5],
        stats1.fraction_informal_in_flood_prone_area[5],
        stats1.fraction_backyard_in_flood_prone_area[5]]
    jeans2 = [
        stats2.fraction_formal_in_flood_prone_area[2],
        stats2.fraction_subsidized_in_flood_prone_area[2],
        stats2.fraction_informal_in_flood_prone_area[2],
        stats2.fraction_backyard_in_flood_prone_area[2]]
    tshirt2 = [
        stats2.fraction_formal_in_flood_prone_area[3],
        stats2.fraction_subsidized_in_flood_prone_area[3],
        stats2.fraction_informal_in_flood_prone_area[3],
        stats2.fraction_backyard_in_flood_prone_area[3]]
    formal_shirt2 = [
        stats2.fraction_formal_in_flood_prone_area[5],
        stats2.fraction_subsidized_in_flood_prone_area[5],
        stats2.fraction_informal_in_flood_prone_area[5],
        stats2.fraction_backyard_in_flood_prone_area[5]]
    colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
    r = np.arange(len(label))
    barWidth = 0.25
    plt.figure(figsize=(10, 7))
    plt.bar(r, jeans, color=colors[0], edgecolor='white',
            width=barWidth, label="20 years")
    plt.bar(r, np.array(tshirt) - np.array(jeans), bottom=np.array(jeans),
            color=colors[1], edgecolor='white', width=barWidth,
            label='50 years')
    plt.bar(r, np.array(formal_shirt) - np.array(tshirt),
            bottom=np.array(tshirt), color=colors[2], edgecolor='white',
            width=barWidth, label='100 years')
    plt.bar(r + barWidth, np.array(jeans2),
            color=colors[0], edgecolor='white', width=barWidth)
    plt.bar(r + barWidth, np.array(tshirt2) - np.array(jeans2),
            bottom=np.array(jeans2), color=colors[1], edgecolor='white',
            width=barWidth)
    plt.bar(r + barWidth, np.array(formal_shirt2) - np.array(tshirt2),
            bottom=np.array(tshirt2), color=colors[2], edgecolor='white',
            width=barWidth)
    # TODO: why?
    # plt.legend(loc='upper right')
    plt.legend()
    plt.xticks(r + barWidth/2, label)
    plt.text(
        r[0] - 0.1,
        stats1.fraction_formal_in_flood_prone_area[5] + 1000,
        legend1)
    plt.text(
        r[1] - 0.1,
        stats1.fraction_subsidized_in_flood_prone_area[5] + 1000,
        legend1)
    plt.text(
        r[2] - 0.1,
        stats1.fraction_informal_in_flood_prone_area[5] + 1000,
        legend1)
    plt.text(
        r[3] - 0.1,
        stats1.fraction_backyard_in_flood_prone_area[5] + 1000,
        legend1)
    plt.text(
        r[0] + 0.15,
        stats2.fraction_formal_in_flood_prone_area[5] + 1000,
        legend2)
    plt.text(
        r[1] + 0.15,
        stats2.fraction_subsidized_in_flood_prone_area[5] + 1000,
        legend2)
    plt.text(
        r[2] + 0.15,
        stats2.fraction_informal_in_flood_prone_area[5] + 1000,
        legend2)
    # TODO: no need to correct for max?
    plt.text(
        r[3] + 0.15,
        stats2.fraction_backyard_in_flood_prone_area[5] + 1000,
        legend2)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Dwellings in flood-prone areas", labelpad=15)
    plt.savefig(path_plots + 'validation_flood_proportion_'
                + type_flood + '.png')
    # plt.show()
    plt.close()


def validation_flood_coastal(stats1, stats2, legend1, legend2, type_flood,
                             path_plots):
    """Bar plot flood depth and area across some RPs per housing types."""
    label = ["Formal private", "Formal subsidized",
             "Informal \n settlements", "Informal \n backyards"]

    # FLOOD DEPTH

    # First for validation data
    # RP = 25 yrs
    tshirt = [stats1.flood_depth_formal[4], stats1.flood_depth_subsidized[4],
              stats1.flood_depth_informal[4], stats1.flood_depth_backyard[4]]
    # RP = 50 yrs
    tshirtb = [stats1.flood_depth_formal[5], stats1.flood_depth_subsidized[5],
               stats1.flood_depth_informal[5], stats1.flood_depth_backyard[5]]
    # RP = 100 yrs
    # TODO: change name?
    formal_shirt = [
        stats1.flood_depth_formal[6], stats1.flood_depth_subsidized[6],
        stats1.flood_depth_informal[6], stats1.flood_depth_backyard[6]]

    # Then for simulation
    # RP = 20 yrs
    tshirt2 = [stats2.flood_depth_formal[4], stats2.flood_depth_subsidized[4],
               stats2.flood_depth_informal[4], stats2.flood_depth_backyard[4]]
    # RP = 50 yrs
    tshirtb2 = [stats2.flood_depth_formal[5], stats2.flood_depth_subsidized[5],
                stats2.flood_depth_informal[5], stats2.flood_depth_backyard[5]]
    # RP = 100 yrs
    # TODO: change name?
    formal_shirt2 = [
        stats2.flood_depth_formal[6], stats2.flood_depth_subsidized[6],
        stats2.flood_depth_informal[6], stats2.flood_depth_backyard[6]]

    # TODO: check uses
    colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
    r = np.arange(len(label))
    barWidth = 0.25
    plt.figure(figsize=(10, 7))
    plt.bar(r, np.array(tshirt),
            color=colors[1], edgecolor='white', width=barWidth,
            label='25 years')
    # valueb = np.maximum(np.array(tshirtb) - np.array(tshirt),
    # np.full(4, 0.003))
    # floorb = np.maximum(np.array(tshirtb), np.array(tshirt))
    plt.bar(r, np.array(tshirtb) - np.array(tshirt),
            bottom=np.array(tshirt), color=colors[2], edgecolor='white',
            width=barWidth, label='50 years')
    # valuec = np.maximum(np.array(formal_shirt) - floorb, np.full(4, 0.003))
    plt.bar(r, np.array(formal_shirt) - np.array(tshirtb),
            bottom=np.array(tshirtb), color=colors[3], edgecolor='white',
            width=barWidth, label='100 years')
    plt.bar(r + barWidth, np.array(tshirt2),
            color=colors[1], edgecolor='white', width=barWidth)
    # valueb2 = np.maximum(np.array(tshirtb2) - np.array(tshirt2),
    # np.full(4, 0.003))
    # floorb2 = np.maximum(np.array(tshirtb2), np.array(tshirt2))
    plt.bar(r + barWidth, np.array(tshirtb2) - np.array(tshirt2),
            bottom=np.array(tshirt2), color=colors[2], edgecolor='white',
            width=barWidth)
    # valuec2 = np.maximum(np.array(formal_shirt2) - floorb2,
    # np.full(4, 0.003))
    plt.bar(r + barWidth, np.array(formal_shirt2) - np.array(tshirtb2),
            bottom=np.array(tshirtb2), color=colors[3], edgecolor='white',
            width=barWidth)
    plt.legend()
    plt.xticks(r + barWidth/2, label)
    # plt.ylim(0, 1)
    # TODO: check look
    plt.text(r[0] - 0.1,
             np.maximum(
                 stats1.flood_depth_formal[6], stats1.flood_depth_formal[4]
                 ) + 0.01,
             legend1)
    plt.text(r[1] - 0.1,
             np.maximum(
                 stats1.flood_depth_subsidized[6],
                 stats1.flood_depth_subsidized[4]
                 ) + 0.01,
             legend1)
    plt.text(r[2] - 0.1,
             np.maximum(
                 stats1.flood_depth_informal[6], stats1.flood_depth_informal[4]
                 ) + 0.01,
             legend1)
    plt.text(r[3] - 0.1,
             np.maximum(
                 stats1.flood_depth_backyard[6], stats1.flood_depth_backyard[4]
                 ) + 0.01,
             legend1)
    plt.text(r[0] + 0.15,
             np.maximum(
                 stats2.flood_depth_formal[6], stats2.flood_depth_formal[4]
                 ) + 0.01,
             legend2)
    plt.text(r[1] + 0.15,
             np.maximum(
                 stats2.flood_depth_subsidized[6],
                 stats2.flood_depth_subsidized[4]
                 ) + 0.01,
             legend2)
    plt.text(r[2] + 0.15,
             np.maximum(
                 stats2.flood_depth_informal[6], stats2.flood_depth_informal[4]
                 ) + 0.01,
             legend2)
    plt.text(r[3] + 0.15,
             np.maximum(
                 stats2.flood_depth_backyard[6], stats2.flood_depth_backyard[4]
                 ) + 0.01,
             legend2)
    plt.ylabel("Average flood depth (m)", labelpad=15)
    plt.tick_params(labelbottom=True)
    plt.savefig(path_plots + 'validation_flood_depth_' + type_flood + '.png')
    # plt.show()
    plt.close()

    # FLOOD-PRONE AREA
    # TODO: really need to change names...
    jeans = [
        stats1.fraction_formal_in_flood_prone_area[4],
        stats1.fraction_subsidized_in_flood_prone_area[4],
        stats1.fraction_informal_in_flood_prone_area[4],
        stats1.fraction_backyard_in_flood_prone_area[4]]
    tshirt = [
        stats1.fraction_formal_in_flood_prone_area[5],
        stats1.fraction_subsidized_in_flood_prone_area[5],
        stats1.fraction_informal_in_flood_prone_area[5],
        stats1.fraction_backyard_in_flood_prone_area[5]]
    formal_shirt = [
        stats1.fraction_formal_in_flood_prone_area[6],
        stats1.fraction_subsidized_in_flood_prone_area[6],
        stats1.fraction_informal_in_flood_prone_area[6],
        stats1.fraction_backyard_in_flood_prone_area[6]]
    jeans2 = [
        stats2.fraction_formal_in_flood_prone_area[4],
        stats2.fraction_subsidized_in_flood_prone_area[4],
        stats2.fraction_informal_in_flood_prone_area[4],
        stats2.fraction_backyard_in_flood_prone_area[4]]
    tshirt2 = [
        stats2.fraction_formal_in_flood_prone_area[5],
        stats2.fraction_subsidized_in_flood_prone_area[5],
        stats2.fraction_informal_in_flood_prone_area[5],
        stats2.fraction_backyard_in_flood_prone_area[5]]
    formal_shirt2 = [
        stats2.fraction_formal_in_flood_prone_area[6],
        stats2.fraction_subsidized_in_flood_prone_area[6],
        stats2.fraction_informal_in_flood_prone_area[6],
        stats2.fraction_backyard_in_flood_prone_area[6]]
    colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
    r = np.arange(len(label))
    barWidth = 0.25
    plt.figure(figsize=(10, 7))
    plt.bar(r, jeans, color=colors[0], edgecolor='white',
            width=barWidth, label="25 years")
    plt.bar(r, np.array(tshirt) - np.array(jeans), bottom=np.array(jeans),
            color=colors[1], edgecolor='white', width=barWidth,
            label='50 years')
    plt.bar(r, np.array(formal_shirt) - np.array(tshirt),
            bottom=np.array(tshirt), color=colors[2], edgecolor='white',
            width=barWidth, label='100 years')
    plt.bar(r + barWidth, np.array(jeans2),
            color=colors[0], edgecolor='white', width=barWidth)
    plt.bar(r + barWidth, np.array(tshirt2) - np.array(jeans2),
            bottom=np.array(jeans2), color=colors[1], edgecolor='white',
            width=barWidth)
    plt.bar(r + barWidth, np.array(formal_shirt2) - np.array(tshirt2),
            bottom=np.array(tshirt2), color=colors[2], edgecolor='white',
            width=barWidth)
    # TODO: why?
    # plt.legend(loc='upper right')
    plt.legend()
    plt.xticks(r + barWidth/2, label)
    plt.text(
        r[0] - 0.1,
        stats1.fraction_formal_in_flood_prone_area[6] + 20,
        legend1)
    plt.text(
        r[1] - 0.1,
        stats1.fraction_subsidized_in_flood_prone_area[6] + 20,
        legend1)
    plt.text(
        r[2] - 0.1,
        stats1.fraction_informal_in_flood_prone_area[6] + 20,
        legend1)
    plt.text(
        r[3] - 0.1,
        stats1.fraction_backyard_in_flood_prone_area[6] + 20,
        legend1)
    plt.text(
        r[0] + 0.15,
        stats2.fraction_formal_in_flood_prone_area[6] + 20,
        legend2)
    plt.text(
        r[1] + 0.15,
        stats2.fraction_subsidized_in_flood_prone_area[6] + 20,
        legend2)
    plt.text(
        r[2] + 0.15,
        stats2.fraction_informal_in_flood_prone_area[6] + 20,
        legend2)
    # TODO: no need to correct for max?
    plt.text(
        r[3] + 0.15,
        stats2.fraction_backyard_in_flood_prone_area[6] + 20,
        legend2)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Dwellings in flood-prone areas", labelpad=15)
    plt.savefig(path_plots + 'validation_flood_proportion_'
                + type_flood + '.png')
    # plt.show()
    plt.close()


def plot_damages(damages1, damages2, path_plots, flood_categ, options):
    """Plot aggregate annualized damages per housing type."""
    # TODO: Check look
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True

    data1 = [
        [outfld.annualize_damages(
            damages1.formal_structure_damages, flood_categ, 'formal', options)
            / 1000000,
         outfld.annualize_damages(
             damages1.subsidized_structure_damages, flood_categ, 'subsidized',
             options) / 1000000,
         outfld.annualize_damages(
             damages1.informal_structure_damages, flood_categ, 'informal',
             options) / 1000000,
         outfld.annualize_damages(
             damages1.backyard_structure_damages, flood_categ, 'backyard',
             options) / 1000000],
        [outfld.annualize_damages(
            damages1.formal_content_damages, flood_categ, 'formal', options)
            / 1000000,
         outfld.annualize_damages(
             damages1.subsidized_content_damages, flood_categ, 'subsidized',
             options) / 1000000,
         outfld.annualize_damages(
             damages1.informal_content_damages, flood_categ, 'informal',
             options) / 1000000,
         outfld.annualize_damages(
             damages1.backyard_content_damages, flood_categ, 'backyard',
             options) / 1000000]
        ]
    data2 = [
        [outfld.annualize_damages(
            damages2.formal_structure_damages, flood_categ, 'formal', options)
            / 1000000,
         outfld.annualize_damages(
             damages2.subsidized_structure_damages, flood_categ, 'subsidized',
             options) / 1000000,
         outfld.annualize_damages(
             damages2.informal_structure_damages, flood_categ, 'informal',
             options) / 1000000,
         outfld.annualize_damages(
             damages2.backyard_structure_damages, flood_categ, 'backyard',
             options) / 1000000],
        [outfld.annualize_damages(
            damages2.formal_content_damages, flood_categ, 'formal', options)
            / 1000000,
         outfld.annualize_damages(
             damages2.subsidized_content_damages, flood_categ, 'subsidized',
             options) / 1000000,
         outfld.annualize_damages(
             damages2.informal_content_damages, flood_categ, 'informal',
             options) / 1000000,
         outfld.annualize_damages(
             damages2.backyard_content_damages, flood_categ, 'backyard',
             options) / 1000000]
        ]
    barWidth = 0.25
    X = np.arange(4)
    colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
    plt.figure(figsize=(10, 7))
    plt.bar(
        X - barWidth/2, data1[0], color=colors[1], width=barWidth,
        label="Structures (sim)")
    plt.bar(
        X + barWidth/2, data2[0], color=colors[2], width=barWidth,
        label="Structures (data)")
    plt.legend()
    # plt.ylim(0, 25)
    quarter = ["Formal private", "Formal subsidized",
               "Informal \n settlements", "Informal \n in backyards"]
    plt.xticks(X, quarter)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year")
    plt.savefig(path_plots + flood_categ + '_structures_damages.png')
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.bar(
        X - barWidth/2, data1[1], color=colors[1], width=barWidth,
        label="Contents (sim)")
    plt.bar(
        X + barWidth/2, data2[1], color=colors[2], width=barWidth,
        label="Contents (data)")
    plt.legend()
    # plt.ylim(0, 25)
    quarter = ["Formal private", "Formal subsidized",
               "Informal \n settlements", "Informal \n in backyards"]
    plt.xticks(X, quarter)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year")
    plt.savefig(path_plots + flood_categ + '_contents_damages.png')
    # plt.show()
    plt.close()

    # data = [
    #     [outfld.annualize_damages(damages.subsidized_structure_damages)
    #      / 1000000,
    #      outfld.annualize_damages(damages.informal_structure_damages)/1000000,
    #      outfld.annualize_damages(damages.backyard_structure_damages)/1000000],
    #     [outfld.annualize_damages(damages.subsidized_content_damages)/1000000,
    #      outfld.annualize_damages(damages.informal_content_damages)/1000000,
    #      outfld.annualize_damages(damages.backyard_content_damages)/1000000]
    #     ]
    # X = np.arange(3)
    # plt.bar(X + 0.00, data[0], color='b', width=0.25, label="Structures")
    # plt.bar(X + 0.25, data[1], color='g', width=0.25, label="Contents")
    # plt.legend()
    # plt.ylim(0, 4)
    # quarter = ["Formal subsidized",
    #            "Informal \n settlements", "Informal \n in backyards"]
    # plt.xticks(X, quarter)
    # plt.tick_params(labelbottom=True)
    # plt.ylabel("Million R per year")
    # plt.show()
    # plt.savefig(path_outputs + name + 'flood_damages_zoom.png')
    # plt.close()


def simul_damages(damages, path_plots, flood_categ, options):
    """Plot aggregate annualized damages per housing type."""
    # TODO: Check look
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True

    data = [
        [outfld.annualize_damages(
            damages.formal_structure_damages, flood_categ, 'formal', options)
            / 1000000,
         outfld.annualize_damages(
             damages.subsidized_structure_damages, flood_categ, 'subsidized',
             options) / 1000000,
         outfld.annualize_damages(
             damages.informal_structure_damages, flood_categ, 'informal',
             options) / 1000000,
         outfld.annualize_damages(
             damages.backyard_structure_damages, flood_categ, 'backyard',
             options) / 1000000],
        [outfld.annualize_damages(
            damages.formal_content_damages, flood_categ, 'formal', options)
            / 1000000,
         outfld.annualize_damages(
             damages.subsidized_content_damages, flood_categ, 'subsidized',
             options) / 1000000,
         outfld.annualize_damages(
             damages.informal_content_damages, flood_categ, 'informal',
             options) / 1000000,
         outfld.annualize_damages(
             damages.backyard_content_damages, flood_categ, 'backyard',
             options) / 1000000]
        ]

    barWidth = 0.25
    X = np.arange(4)
    colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
    plt.figure(figsize=(10, 7))
    plt.bar(
        X, data[0], color=colors[1], width=barWidth,
        label="Structures")
    plt.legend()
    # plt.ylim(0, 25)
    quarter = ["Formal private", "Formal subsidized",
               "Informal \n settlements", "Informal \n in backyards"]
    plt.xticks(X, quarter)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year")
    plt.savefig(path_plots + flood_categ + '_structures_damages.png')
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.bar(
        X, data[1], color=colors[1], width=barWidth,
        label="Contents")
    plt.legend()
    # plt.ylim(0, 25)
    quarter = ["Formal private", "Formal subsidized",
               "Informal \n settlements", "Informal \n in backyards"]
    plt.xticks(X, quarter)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year")
    plt.savefig(path_plots + flood_categ + '_contents_damages.png')
    # plt.show()
    plt.close()


def simul_damages_time(list_damages, path_plots, flood_categ, options):
    """Plot aggregate annualized damages per housing type."""
    list_data = []
    for damages in list_damages:
        data = [
            [outfld.annualize_damages(
                damages.formal_structure_damages, flood_categ,
                'formal', options) / 1000000,
             outfld.annualize_damages(
                 damages.subsidized_structure_damages, flood_categ,
                 'subsidized', options) / 1000000,
             outfld.annualize_damages(
                 damages.informal_structure_damages, flood_categ, 'informal',
                 options) / 1000000,
             outfld.annualize_damages(
                 damages.backyard_structure_damages, flood_categ, 'backyard',
                 options) / 1000000],
            [outfld.annualize_damages(
                damages.formal_content_damages, flood_categ, 'formal', options)
                / 1000000,
             outfld.annualize_damages(
                 damages.subsidized_content_damages, flood_categ, 'subsidized',
                 options) / 1000000,
             outfld.annualize_damages(
                 damages.informal_content_damages, flood_categ, 'informal',
                 options) / 1000000,
             outfld.annualize_damages(
                 damages.backyard_content_damages, flood_categ, 'backyard',
                 options) / 1000000]
            ]
    list_data.append(data)

    years_simul = np.arange(2011, 2011 + 30)
    colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']

    # It is best to separate housing types for visualisation

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(years_simul, list_data[:, 0, 0],
            color=colors[1], label="Structure")
    ax.plot(years_simul, list_data[:, 1, 0],
            color=colors[2], label="Contents")
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year", labelpad=15)
    plt.savefig(path_plots + flood_categ + '_evol_FP_damages.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(years_simul, list_data[:, 0, 1],
            color=colors[1], label="Structure")
    ax.plot(years_simul, list_data[:, 1, 1],
            color=colors[2], label="Contents")
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year)", labelpad=15)
    plt.savefig(path_plots + flood_categ + '_evol_FS_damages.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(years_simul, list_data[:, 0, 2],
            color=colors[1], label="Structure")
    ax.plot(years_simul, list_data[:, 1, 2],
            color=colors[2], label="Contents")
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year", labelpad=15)
    plt.savefig(path_plots + flood_categ + 'evol_IS_damages.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(years_simul, list_data[:, 0, 3],
            color=colors[1], label="Structure")
    ax.plot(years_simul, list_data[:, 1, 3],
            color=colors[2], label="Contents")
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year", labelpad=15)
    plt.savefig(path_plots + flood_categ + 'evol_IB_damages.png')
    plt.close()


# TODO: not used?
# def compare_damages(damages1, damages2, label1, label2, name, path_outputs):
#     """d."""
#     data = [
#         [outfld.annualize_damages(damages1.formal_structure_damages),
#          outfld.annualize_damages(damages1.formal_content_damages)],
#         [outfld.annualize_damages(damages2.formal_structure_damages),
#          outfld.annualize_damages(damages2.formal_content_damages)]
#         ]
#     X = np.arange(2)
#     plt.bar(X + 0.00, data[0], color='b', width=0.25, label=label1)
#     plt.bar(X + 0.25, data[1], color='g', width=0.25, label="Data")
#     plt.legend()
#     plt.ylim(0, 28000000)
#     plt.title("Formal")
#     plt.text(0.125, 26000000, "Structures")
#     plt.text(1.125, 26000000, "Contents")
#     plt.show()
#     plt.savefig(path_outputs + name + 'flood_damages_formal.png')
#     plt.close()

#     data = [
#         [outfld.annualize_damages(damages1.subsidized_structure_damages),
#          outfld.annualize_damages(damages1.subsidized_content_damages)],
#         [outfld.annualize_damages(damages2.subsidized_structure_damages),
#          outfld.annualize_damages(damages2.subsidized_content_damages), ]
#         ]
#     plt.bar(X + 0.00, data[0], color='b', width=0.25, label=label1)
#     plt.bar(X + 0.25, data[1], color='g', width=0.25, label=label2)
#     plt.ylim(0, 1800000)
#     plt.title("Subsidized")
#     plt.text(0.125, 1600000, "Structures")
#     plt.text(1.125, 1600000, "Contents")
#     plt.show()
#     plt.savefig(path_outputs + name + 'flood_damages_subsidized.png')
#     plt.close()

#     data = [
#         [outfld.annualize_damages(damages1.informal_structure_damages),
#          outfld.annualize_damages(damages1.informal_content_damages)],
#         [outfld.annualize_damages(damages2.informal_structure_damages),
#          outfld.annualize_damages(damages2.informal_content_damages), ]
#         ]
#     plt.bar(X + 0.00, data[0], color='b', width=0.25, label=label1)
#     plt.bar(X + 0.25, data[1], color='g', width=0.25, label=label2)
#     plt.ylim(0, 200000)
#     plt.title("Informal")
#     plt.text(0.125, 180000, "Structures")
#     plt.text(1.125, 180000, "Contents")
#     plt.show()
#     plt.savefig(path_outputs + name + 'flood_damages_informal.png')
#     plt.close()

#     data = [
#         [outfld.annualize_damages(damages1.backyard_structure_damages),
#          outfld.annualize_damages(damages1.backyard_content_damages)],
#         [outfld.annualize_damages(damages2.backyard_structure_damages),
#          outfld.annualize_damages(damages2.backyard_content_damages), ]
#         ]
#     plt.bar(X + 0.00, data[0], color='b', width=0.25, label=label1)
#     plt.bar(X + 0.25, data[1], color='g', width=0.25, label=label2)
#     plt.ylim(0, 800000)
#     plt.title("Backyard")
#     plt.text(0.125, 750000, "Structures")
#     plt.text(1.125, 750000, "Contents")
#     plt.show()
#     plt.savefig(path_outputs + name + 'flood_damages_backyard.png')
#     plt.close()


def round_nearest(x, a):
    """Round to nearest decimal number."""
    return round(round(x / a) * a, 2)


def plot_flood_severity_distrib(barWidth, transparency, dictio, flood_type,
                                path_plots, ylim):
    """Plot distribution of flood severity across income groups for some RP."""
    if flood_type == "FD":
        df_1 = dictio[flood_type + '_20yr']
        df_2 = dictio[flood_type + '_50yr']
        df_3 = dictio[flood_type + '_100yr']
    elif flood_type == "FU":
        df_1 = dictio[flood_type + '_20yr']
        df_2 = dictio[flood_type + '_50yr']
        df_3 = dictio[flood_type + '_100yr']
    if flood_type == "P":
        df_1 = dictio[flood_type + '_20yr']
        df_2 = dictio[flood_type + '_50yr']
        df_3 = dictio[flood_type + '_100yr']
    if flood_type == "C_MERITDEM_1":
        df_1 = dictio[flood_type + '_0025']
        df_2 = dictio[flood_type + '_0050']
        df_3 = dictio[flood_type + '_0100']

    df = pd.DataFrame(data=np.transpose(
        np.array([df_1.flood_depth, df_1.sim_poor, df_1.sim_midpoor,
                  df_1.sim_midrich, df_1.sim_rich,
                  df_2.flood_depth, df_2.sim_poor, df_2.sim_midpoor,
                  df_2.sim_midrich, df_2.sim_rich,
                  df_3.flood_depth, df_3.sim_poor, df_3.sim_midpoor,
                  df_3.sim_midrich, df_3.sim_rich])),
        columns=["x_1", "ypoor_1", "ymidpoor_1", "ymidrich_1", "yrich_1",
                 "x_2", "ypoor_2", "ymidpoor_2", "ymidrich_2", "yrich_2",
                 "x_3", "ypoor_3", "ymidpoor_3", "ymidrich_3", "yrich_3"])
    df["round_1"] = round_nearest(df.x_1, barWidth)
    df["round_2"] = round_nearest(df.x_2, barWidth)
    df["round_3"] = round_nearest(df.x_3, barWidth)
    new_df_1 = df[["round_1", "ypoor_1", "ymidpoor_1", "ymidrich_1", "yrich_1"]
                  ].groupby(['round_1']).sum()
    new_df_1["rounded"] = new_df_1.index
    new_df_2 = df[["round_2", "ypoor_2", "ymidpoor_2", "ymidrich_2", "yrich_2"]
                  ].groupby(['round_2']).sum()
    new_df_2["rounded"] = new_df_2.index
    new_df_3 = df[["round_3", "ypoor_3", "ymidpoor_3", "ymidrich_3", "yrich_3"]
                  ].groupby(['round_3']).sum()
    new_df_3["rounded"] = new_df_3.index

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].bar(new_df_1.rounded[barWidth:3], new_df_1.ypoor_1[barWidth:3],
                 width=barWidth, color='royalblue', alpha=transparency[0],
                 label='20 years')
    ax[0, 0].bar(new_df_2.rounded[barWidth:3], new_df_2.ypoor_2[barWidth:3],
                 width=barWidth, color='cornflowerblue', alpha=transparency[1],
                 label='50 years')
    ax[0, 0].bar(new_df_3.rounded[barWidth:3], new_df_3.ypoor_3[barWidth:3],
                 width=barWidth, color='lightsteelblue', alpha=transparency[2],
                 label='100 years')
    ax[0, 0].set_ylabel("Households (nb)")
    ax[0, 0].set_xlabel("Severity of floods (m)")
    ax[0, 0].yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax[0, 0].set_title("Poor")
    ax[0, 0].set_ylim([0, ylim])
    ax[0, 0].legend()
    ax[0, 1].bar(new_df_1.rounded[barWidth:3], new_df_1.ymidpoor_1[barWidth:3],
                 width=barWidth, color='royalblue', alpha=transparency[0],
                 label='25 years')
    ax[0, 1].bar(new_df_2.rounded[barWidth:3], new_df_2.ymidpoor_2[barWidth:3],
                 width=barWidth, color='cornflowerblue', alpha=transparency[1],
                 label='50 years')
    ax[0, 1].bar(new_df_3.rounded[barWidth:3], new_df_3.ymidpoor_3[barWidth:3],
                 width=barWidth, color='lightsteelblue', alpha=transparency[2],
                 label='100 years')
    ax[0, 1].set_ylabel("Households (nb)")
    ax[0, 1].set_xlabel("Severity of floods (m)")
    ax[0, 1].yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax[0, 1].set_title("Mid-poor")
    ax[0, 1].set_ylim([0, ylim])
    ax[0, 1].legend()
    ax[1, 0].bar(new_df_1.rounded[barWidth:3], new_df_1.ymidrich_1[barWidth:3],
                 width=barWidth, color='royalblue', alpha=transparency[0],
                 label='25 years')
    ax[1, 0].bar(new_df_2.rounded[barWidth:3], new_df_2.ymidrich_2[barWidth:3],
                 width=barWidth, color='cornflowerblue', alpha=transparency[1],
                 label='50 years')
    ax[1, 0].bar(new_df_3.rounded[barWidth:3], new_df_3.ymidrich_3[barWidth:3],
                 width=barWidth, color='lightsteelblue', alpha=transparency[2],
                 label='100 years')
    ax[1, 0].set_ylabel("Households (nb)")
    ax[1, 0].set_xlabel("Severity of floods (m)")
    ax[1, 0].yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax[1, 0].set_title("Mid-rich")
    ax[1, 0].set_ylim([0, ylim])
    ax[1, 0].legend()
    ax[1, 1].bar(new_df_1.rounded[barWidth:3], new_df_1.yrich_1[barWidth:3],
                 width=barWidth, color='royalblue', alpha=transparency[0],
                 label='25 years')
    ax[1, 1].bar(new_df_2.rounded[barWidth:3], new_df_2.yrich_2[barWidth:3],
                 width=barWidth, color='cornflowerblue', alpha=transparency[1],
                 label='50 years')
    ax[1, 1].bar(new_df_3.rounded[barWidth:3], new_df_3.yrich_3[barWidth:3],
                 width=barWidth, color='lightsteelblue', alpha=transparency[2],
                 label='100 years')
    ax[1, 1].set_ylabel("Households (nb)")
    ax[1, 1].set_xlabel("Severity of floods (m)")
    ax[1, 1].yaxis.set_major_formatter(
        mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax[1, 1].set_title("Rich")
    ax[1, 1].set_ylim([0, ylim])
    ax[1, 1].legend()
    plt.savefig(path_plots + flood_type + '_severity_distrib.png')
    plt.show()
    plt.close()
