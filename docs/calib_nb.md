---
date: '2022-09-23T09:22:05.479Z'
docname: calib_nb
images: {}
path: /calib-nb
title: 'Notebook: run calibration'
---

# Notebook: run calibration

## Preamble

### Import packages

### Define file paths

This corresponds to the architecture described in the README file (introduction tab of the documentation): the data folder is not hosted on the Github repository and should be placed in the root folder enclosing the repo

### Create associated directories if needed

## Import parameters and options

### We import default parameter and options

### We also set custom options for this simulation

#### We first set options regarding structural assumptions used in the model

#### Then we set options regarding flood data used

#### We also set options for scenarios on time-moving exogenous variables

#### Finally, we set options regarding data processing

Default is set at zero to save computing time (data is simply loaded in the model)

NB: this is only needed to create the data for the first time, or when the source is changed, so that pre-processed data is updated

## Load data

### Basic geographic data

### Macro data

### Households and income data

### Land use projections

### Import flood data (takes some time when agents anticipate floods)

### Import scenarios (for time-moving variables)

### Import income net of commuting costs (for all time periods)

## Prepare data

### Define dominant income group in each census block (SP)

Although the second option seems more logical, we take the first one as default since we are going to consider median SP prices, and we want associated net income to be in line with those values to avoid a sample selection bias in our regressions.

### Obtain number of formal private housing units per SP

NB: it is not clear whether RDP are included in SP formal count, and if they should be taken out based on imperfect cadastral estimations. For reference, we include the two options.

Given the uncertainty surrounding RDP counts, we take the second option as default and prefer to rely on sample selection (see below) to exclude the SPs where RDP housing is likely to drive most of our results

### Sample selection

As the relations we are going to estimate are only true for the formal private sector, we exclude SPs in the bottom quintile of property prices and for which more than 5% of households are reported to live in informal housing (settlements + backyards). We also exclude “rural” SPs (i.e., those that are large, with a small share than can be urbanized).

We also add options to consider other criteria, namely we offer to exclude poorest income group (which is in effect crowded out from the formal sector), as well as Mitchell’s Plain (as its housing market is very specific) and far-away land (for which we have few observations).

We pick the second choice as our default since it is more conservative than the first, and less ad hoc than the third one.

## Calibrate construction function parameters

NB: The results are automatically saved for later use in simulations

## Calibrate incomes and gravity parameter

We scan values for the gravity parameter to estimate incomes as a function of it. The value range is set by trial and error: the wider the range you want to test, the longer. In principle, we should find a value within a coarse interval before going to the finer level: this may require several iterations if the underlying data changes.

NB: we do that as it is too long and complex to run a solver directly.

Then, we select the income-gravity pair that best fits the distribution of commuters over distance from the CBD.

NB: we need to proceed in two steps as there is no separate identification of the gravity parameter and the net incomes.

Note that incomes are fitted to reproduce the observed distribution of jobs across income groups (in selected job centers), based on commuting choice model.

### Let us first visualize inputs

### We visualize the associated calibrated outputs

## Calibrate utility function parameters

We compute local incomes net of commuting costs at the SP (not grid) level that is used in calibration.

Then we calibrate utility function parameters based on the maximization of a composite likelihood that reproduces the fit on exogenous amenities, dwelling sizes, and income sorting.

The warnings in the execution come from the fact that rent interpolation is discountinuous when underlying data is scattered. This is not a big issue to the extent that we are not interested in the shape of the rent function per se, but it causes optimization solver to break down when the associated likelihood function is not smooth. The calibration therefore essentially relies on parameter scanning.

## Calibrate disamenity index for informal backyards + settlements

NB: Since disamenity index calibration relies on the model fit and is not computed a priori (contrary to other parameters), the options set in the preamble should be the same as the ones used in the main script, so that the calibrated values are in line with the structural assumptions used

### We start with a general (not location-specific) calibration

We are going to compute the initial state equilibrium for each pair of parameters, and retain the one that best fits the observed number of households in informal settlements + backyards

### Calibrate location-specific disamenity index
