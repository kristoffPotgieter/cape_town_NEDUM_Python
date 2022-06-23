# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:22:55 2020.

@author: Charlotte Liotta
"""

import numpy as np
import pandas as pd


def import_employment_data(households_per_income_class, param, path_data):
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
    jobsCentersNgroup = jobsCentersNgroup[selectedCenters, :]
    # Rescale (wrt census data) to keep the correct global income distribution
    # More specifically, this allows to go from individual to household level
    jobsCentersNGroupRescaled = (
        jobsCentersNgroup * households_per_income_class[None, :]
        / np.nansum(jobsCentersNgroup, 0)
        )

    return jobsCentersNGroupRescaled
