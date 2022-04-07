# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:38:21 2020.

@author: Charlotte Liotta
"""

import numpy as np
import matplotlib.pyplot as plt

import outputs.flood_outputs as outfld


# %% Floods

def plot_damages(damages, name, path_outputs):
    """d."""
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True

    data = [
        [outfld.annualize_damages(damages.formal_structure_damages)/1000000,
         outfld.annualize_damages(damages.subsidized_structure_damages)
         / 1000000,
         outfld.annualize_damages(damages.informal_structure_damages)/1000000,
         outfld.annualize_damages(damages.backyard_structure_damages)/1000000],
        [outfld.annualize_damages(damages.formal_content_damages)/1000000,
         outfld.annualize_damages(damages.subsidized_content_damages)/1000000,
         outfld.annualize_damages(damages.informal_content_damages)/1000000,
         outfld.annualize_damages(damages.backyard_content_damages)/1000000]
        ]
    X = np.arange(4)
    plt.bar(X + 0.00, data[0], color='b', width=0.25, label="Structures")
    plt.bar(X + 0.25, data[1], color='g', width=0.25, label="Contents")
    plt.legend()
    plt.ylim(0, 25)
    quarter = ["Formal private", "Formal subsidized",
               "Informal \n settlements", "Informal \n in backyards"]
    plt.xticks(X, quarter)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year")
    plt.savefig(path_outputs + name + 'flood_damages.png')
    plt.close()

    data = [
        [outfld.annualize_damages(damages.subsidized_structure_damages)
         / 1000000,
         outfld.annualize_damages(damages.informal_structure_damages)/1000000,
         outfld.annualize_damages(damages.backyard_structure_damages)/1000000],
        [outfld.annualize_damages(damages.subsidized_content_damages)/1000000,
         outfld.annualize_damages(damages.informal_content_damages)/1000000,
         outfld.annualize_damages(damages.backyard_content_damages)/1000000]
        ]
    X = np.arange(3)
    plt.bar(X + 0.00, data[0], color='b', width=0.25, label="Structures")
    plt.bar(X + 0.25, data[1], color='g', width=0.25, label="Contents")
    plt.legend()
    plt.ylim(0, 4)
    quarter = ["Formal subsidized",
               "Informal \n settlements", "Informal \n in backyards"]
    plt.xticks(X, quarter)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Million R per year")
    plt.show()
    plt.savefig(path_outputs + name + 'flood_damages_zoom.png')
    plt.close()


# TODO: not used?
def compare_damages(damages1, damages2, label1, label2, name, path_outputs):
    """d."""
    data = [
        [outfld.annualize_damages(damages1.formal_structure_damages),
         outfld.annualize_damages(damages1.formal_content_damages)],
        [outfld.annualize_damages(damages2.formal_structure_damages),
         outfld.annualize_damages(damages2.formal_content_damages)]
        ]
    X = np.arange(2)
    plt.bar(X + 0.00, data[0], color='b', width=0.25, label=label1)
    plt.bar(X + 0.25, data[1], color='g', width=0.25, label="Data")
    plt.legend()
    plt.ylim(0, 28000000)
    plt.title("Formal")
    plt.text(0.125, 26000000, "Structures")
    plt.text(1.125, 26000000, "Contents")
    plt.show()
    plt.savefig(path_outputs + name + 'flood_damages_formal.png')
    plt.close()

    data = [
        [outfld.annualize_damages(damages1.subsidized_structure_damages),
         outfld.annualize_damages(damages1.subsidized_content_damages)],
        [outfld.annualize_damages(damages2.subsidized_structure_damages),
         outfld.annualize_damages(damages2.subsidized_content_damages), ]
        ]
    plt.bar(X + 0.00, data[0], color='b', width=0.25, label=label1)
    plt.bar(X + 0.25, data[1], color='g', width=0.25, label=label2)
    plt.ylim(0, 1800000)
    plt.title("Subsidized")
    plt.text(0.125, 1600000, "Structures")
    plt.text(1.125, 1600000, "Contents")
    plt.show()
    plt.savefig(path_outputs + name + 'flood_damages_subsidized.png')
    plt.close()

    data = [
        [outfld.annualize_damages(damages1.informal_structure_damages),
         outfld.annualize_damages(damages1.informal_content_damages)],
        [outfld.annualize_damages(damages2.informal_structure_damages),
         outfld.annualize_damages(damages2.informal_content_damages), ]
        ]
    plt.bar(X + 0.00, data[0], color='b', width=0.25, label=label1)
    plt.bar(X + 0.25, data[1], color='g', width=0.25, label=label2)
    plt.ylim(0, 200000)
    plt.title("Informal")
    plt.text(0.125, 180000, "Structures")
    plt.text(1.125, 180000, "Contents")
    plt.show()
    plt.savefig(path_outputs + name + 'flood_damages_informal.png')
    plt.close()

    data = [
        [outfld.annualize_damages(damages1.backyard_structure_damages),
         outfld.annualize_damages(damages1.backyard_content_damages)],
        [outfld.annualize_damages(damages2.backyard_structure_damages),
         outfld.annualize_damages(damages2.backyard_content_damages), ]
        ]
    plt.bar(X + 0.00, data[0], color='b', width=0.25, label=label1)
    plt.bar(X + 0.25, data[1], color='g', width=0.25, label=label2)
    plt.ylim(0, 800000)
    plt.title("Backyard")
    plt.text(0.125, 750000, "Structures")
    plt.text(1.125, 750000, "Contents")
    plt.show()
    plt.savefig(path_outputs + name + 'flood_damages_backyard.png')
    plt.close()


def validation_flood(name, stats1, stats2, legend1, legend2, type_flood,
                     path_outputs):
    """d."""
    label = ["Formal private", "Formal subsidized",
             "Informal \n settlements", "Informal \n in backyards"]
    tshirt = [stats1.flood_depth_formal[2], stats1.flood_depth_subsidized[2],
              stats1.flood_depth_informal[2], stats1.flood_depth_backyard[2]]
    tshirtb = [stats1.flood_depth_formal[3], stats1.flood_depth_subsidized[3],
               stats1.flood_depth_informal[3], stats1.flood_depth_backyard[3]]
    formal_shirt = [
        stats1.flood_depth_formal[5], stats1.flood_depth_subsidized[5],
        stats1.flood_depth_informal[5], stats1.flood_depth_backyard[5]]
    tshirt2 = [stats2.flood_depth_formal[2], stats2.flood_depth_subsidized[2],
               stats2.flood_depth_informal[2], stats2.flood_depth_backyard[2]]
    tshirtb2 = [stats2.flood_depth_formal[3], stats2.flood_depth_subsidized[3],
                stats2.flood_depth_informal[3], stats2.flood_depth_backyard[3]]
    formal_shirt2 = [
        stats2.flood_depth_formal[5], stats2.flood_depth_subsidized[5],
        stats2.flood_depth_informal[5], stats2.flood_depth_backyard[5]]
    colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
    r = np.arange(len(label))
    barWidth = 0.25
    plt.figure(figsize=(10, 7))
    plt.bar(r, np.array(tshirt),
            color=colors[1], edgecolor='white', width=barWidth,
            label='20 years')
    plt.bar(r, np.array(tshirtb) - np.array(tshirt),
            bottom=np.array(tshirt), color=colors[2], edgecolor='white',
            width=barWidth, label='50 years')
    plt.bar(r, np.array(formal_shirt) - np.array(tshirtb),
            bottom=(np.array(tshirtb)), color=colors[3], edgecolor='white',
            width=barWidth, label='100 years')
    plt.bar(r + 0.25, np.array(tshirt2),
            color=colors[1], edgecolor='white', width=barWidth)
    plt.bar(r + 0.25, np.array(tshirtb2) - np.array(tshirt2),
            bottom=np.array(tshirt2), color=colors[2], edgecolor='white',
            width=barWidth)
    plt.bar(r + 0.25, np.array(formal_shirt2) - np.array(tshirtb2),
            bottom=np.array(tshirtb2), color=colors[3], edgecolor='white',
            width=barWidth)
    plt.legend()
    plt.xticks(r, label)
    # plt.ylim(0, 1)
    plt.text(r[0] - 0.1, stats1.flood_depth_formal[5] + 0.01, legend1)
    plt.text(r[1] - 0.1, stats1.flood_depth_subsidized[5] + 0.01, legend1)
    plt.text(r[2] - 0.1, stats1.flood_depth_informal[5] + 0.01, legend1)
    plt.text(r[3] - 0.1, stats1.flood_depth_backyard[5] + 0.01, legend1)
    plt.text(r[0] + 0.15, stats2.flood_depth_formal[5] + 0.01, legend2)
    plt.text(r[1] + 0.15, stats2.flood_depth_subsidized[5] + 0.01, legend2)
    plt.text(r[2] + 0.15, stats2.flood_depth_informal[5] + 0.01, legend2)
    plt.text(r[3] + 0.15,
             max(stats2.flood_depth_backyard[2],
             stats2.flood_depth_backyard[3],
             stats2.flood_depth_backyard[5]) + 0.01,
             legend2)
    plt.ylabel("Average flood depth (m)")
    plt.tick_params(labelbottom=True)
    plt.show()
    plt.savefig(path_outputs +
                name + '/validation_flood_depth_' + type_flood + '.png')
    plt.close()

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
    plt.bar(r + 0.25, np.array(jeans2),
            color=colors[0], edgecolor='white', width=barWidth)
    plt.bar(r + 0.25, np.array(tshirt2) - np.array(jeans2),
            bottom=np.array(jeans2), color=colors[1], edgecolor='white',
            width=barWidth)
    plt.bar(r + 0.25, np.array(formal_shirt2) - np.array(tshirt2),
            bottom=np.array(tshirt2), color=colors[2], edgecolor='white',
            width=barWidth)
    plt.legend(loc='upper right')
    plt.xticks(r, label)
    plt.text(
        r[0] - 0.1,
        stats1.fraction_formal_in_flood_prone_area[5] + 0.005,
        legend1)
    plt.text(
        r[1] - 0.1,
        stats1.fraction_subsidized_in_flood_prone_area[5] + 0.005,
        legend1)
    plt.text(
        r[2] - 0.1,
        stats1.fraction_informal_in_flood_prone_area[5] + 0.005,
        legend1)
    plt.text(
        r[3] - 0.1,
        stats1.fraction_backyard_in_flood_prone_area[5] + 0.005,
        legend1)
    plt.text(
        r[0] + 0.15,
        stats2.fraction_formal_in_flood_prone_area[5] + 0.005,
        legend2)
    plt.text(
        r[1] + 0.15,
        stats2.fraction_subsidized_in_flood_prone_area[5] + 0.005,
        legend2)
    plt.text(
        r[2] + 0.15,
        stats2.fraction_informal_in_flood_prone_area[5] + 0.005,
        legend2)
    plt.text(
        r[3] + 0.15,
        stats2.fraction_backyard_in_flood_prone_area[5] + 0.005,
        legend2)
    plt.tick_params(labelbottom=True)
    plt.ylabel("Dwellings in flood-prone areas (%)")
    plt.show()
    plt.savefig(path_outputs + name + 'validation_flood_proportion_'
                + type_flood + '.png')
    plt.close()
