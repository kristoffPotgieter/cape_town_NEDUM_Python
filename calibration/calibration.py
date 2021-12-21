# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:31:00 2020

@author: Charlotte Liotta
"""

print("\n*** NEDUM-Cape-Town - Construction function calibration ***\n")

import numpy as np
import scipy.io
import pandas as pd
import numpy.matlib
from sklearn.linear_model import LinearRegression
import copy

from inputs.data import *
from inputs.parameters_and_options import *
from equilibrium.compute_equilibrium import *
from outputs.export_outputs import *
from outputs.export_outputs_floods import *
from outputs.flood_outputs import *
from equilibrium.functions_dynamic import *
from equilibrium.run_simulations import *
from calibration.calibration import *
from calibration.compute_income import *
from inputs.WBUS2_depth import *

# %% Import parameters and options

print("\n*** Load parameters and options ***\n")

#IMPORT PARAMETERS AND OPTIONS

options = import_options()
param = import_param(options)
t = np.arange(0, 1)

#OPTIONS FOR THIS SIMULATION
options["coeff_land"] = 'old'
options["WBUS2"] = 1

# %% Load data

print("\n*** Load data ***\n")

#DATA

#Grid
grid, center = import_grid()
amenities = import_amenities()

#Households data
income_class_by_housing_type = import_hypothesis_housing_type()
income_2011 = pd.read_csv('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Basile data/Income_distribution_2011.csv')
mean_income = np.sum(income_2011.Households_nb * income_2011.INC_med) / sum(income_2011.Households_nb)
households_per_income_class, average_income = import_income_classes_data(param, income_2011)
income_mult = average_income / mean_income
income_net_of_commuting_costs = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/SP_year_0.npy")
param["income_year_reference"] = mean_income
data_rdp, housing_types_sp, housing_types_grid, data_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, threshold_income_distribution, income_distribution, cape_town_limits = import_households_data(options)

#Macro data
interest_rate, population = import_macro_data(param)

#Land-use   
options["urban_edge"] = 1
spline_estimate_RDP, spline_land_backyard, spline_land_RDP, spline_RDP, spline_land_constraints, informal, coeff_land_backyard = import_land_use(grid, options, param, data_rdp, housing_types_grid)
number_properties_RDP = spline_estimate_RDP(0)
total_RDP = spline_RDP(0)
coeff_land = import_coeff_land(spline_land_constraints, spline_land_backyard, informal, spline_land_RDP, param, 0)
housing_limit = import_housig_limit(grid, param)
param = import_construction_parameters(param, grid, housing_types_sp, data_sp["dwelling_size"], mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land)
minimum_housing_supply = param["minimum_housing_supply"]
agricultural_rent = param["agricultural_rent_2011"] ** (param["coeff_a"]) * (param["depreciation_rate"] + interest_rate) / (param["coeff_A"] * param["coeff_b"] ** param["coeff_b"])

#FLOOD DATA
param = infer_WBUS2_depth(housing_types_grid, param)
if options["agents_anticipate_floods"] == 1:
    fraction_capital_destroyed, depth_damage_function_structure, depth_damage_function_contents = import_floods_data(options, param)
elif options["agents_anticipate_floods"] == 0:
    fraction_capital_destroyed = pd.DataFrame()
    fraction_capital_destroyed["structure"] = np.zeros(24014)
    fraction_capital_destroyed["contents"] = np.zeros(24014)
    
# %% Estimation of coefficient of construction function

data_income_group = np.zeros(len(data_sp["income"]))
for j in range(0, 3):
    data_income_group[data_sp["income"] > threshold_income_distribution[j]] = j+1

#Import amenities at the SP level
amenities_sp = import_amenities_SP()
variables_regression = ['distance_ocean', 'distance_ocean_2_4', 'slope_1_5', 'slope_5', 'airport_cone2', 'distance_distr_parks', 'distance_biosphere_reserve', 'distance_train', 'distance_urban_herit']


#Regression
data_number_formal = (housing_types_sp.total_dwellings_SP_2011 - housing_types_sp.backyard_SP_2011 - housing_types_sp.informal_SP_2011)
data_density = data_number_formal / (data_sp["unconstrained_area"] * param["max_land_use"] / 1000000)
selected_density = (data_sp["unconstrained_area"] > 0.6 * 1000000 * data_sp["area"]) & (data_income_group > 0) & (data_sp["mitchells_plain"] == 0) & (data_sp["distance"] < 40) & (data_sp["price"] > np.nanquantile(data_sp["price"], 0.2)) & (data_sp["unconstrained_area"] < np.nanquantile(data_sp["unconstrained_area"], 0.8))
X = np.transpose(np.array([np.log(data_sp["price"][selected_density]), np.log(param["max_land_use"] * data_sp["unconstrained_area"][selected_density]), np.log(data_sp["dwelling_size"][selected_density])]))
y = np.log(data_number_formal[selected_density])
model_construction = LinearRegression().fit(X, y)
model_construction.score(X, y)
model_construction.coef_
model_construction.intercept_

#Export outputs
coeff_b = model_construction.coef_[0]
coeff_a = 1 - coeff_b
coeffKappa = (1 /coeff_b ** coeff_b) * np.exp(model_construction.intercept_)

#Correcting data for rents
#dataRent = dataPrice ** (coeff_a) * (param["depreciation_rate"] + InterpolateInterestRateEvolution(macro_data, t[0])) / (coeffKappa * coeff_b ** coeff_b)
#interestRate = (param["depreciation_rate"] + InterpolateInterestRateEvolution(macro_data, t[0]))

#Cobb-Douglas: 
#simulHousing_CD = coeffKappa.^(1/coeff_a)...
#        .*(coeff_b/interestRate).^(coeff_b/coeff_a)...
#        .*(dataRent).^(coeff_b/coeff_a);

#f1=fit(data.sp2011Distance(selectedDensity), data.spFormalDensityHFA(selectedDensity)','poly5');
#f2=fit(data.sp2011Distance(~isnan(simulHousing_CD)), simulHousing_CD(~isnan(simulHousing_CD))','poly5');

# %% Estimation of incomes and commuting parameters

#listLambda = [4.027, 0]
#list_lambda = 10 ** np.arange(0.6, 0.65, 0.01)
list_lambda = 10 ** np.arange(0.6, 0.605, 0.005)

timeOutput, distanceOutput, monetaryCost, costTime = import_transport_costs(income_2011, param, grid)
job_centers = import_employment_data(households_per_income_class, param)
incomeCenters, distanceDistribution = EstimateIncome(param, timeOutput, distanceOutput, monetaryCost, costTime, job_centers, average_income, income_distribution, list_lambda)

data_modal_shares = np.array([7.8, 14.8, 39.5+0.7, 16, 8]) / (7.8+14.8+39.5+0.7+16+8) * 100
data_time_distribution = np.array([18.3, 32.7, 35.0, 10.5, 3.4]) / (18.3+32.7+35.0+10.5+3.4)
data_distance_distribution = np.array([45.6174222, 18.9010734, 14.9972971, 9.6725616, 5.9425438, 2.5368754, 0.9267125, 0.3591011, 1.0464129])

#Compute accessibility index
bhattacharyyaDistances = -np.log(np.nansum(np.sqrt(data_distance_distribution[:, None] /100 * distanceDistribution), 0))
whichLambda = np.argmin(bhattacharyyaDistances)

lambdaKeep = list_lambda[whichLambda]
#modalSharesKeep = modalShares[:, whichLambda]
#timeDistributionKeep = timeDistribution[:, whichLambda]
distanceDistributionKeep = distanceDistribution[:, whichLambda]
incomeCentersKeep = incomeCenters[:,:,whichLambda]
sns.distplot(np.abs(((incomeCentersKeep - scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/incomeCentersKeep.mat')['incomeCentersKeep']) /scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/incomeCentersKeep.mat')['incomeCentersKeep']) * 100))
#incomeCentersKeep = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/incomeCentersKeep.mat')['incomeCentersKeep']
#lamdaKepp = 10 ** 0.605
incomeNetOfCommuting = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/SP_year_0.npy")

# %% Calibration utility function

#In which areas we actually measure the likelihood
selectedSP = ((housing_types_sp.backyard_SP_2011 + housing_types_sp.informal_SP_2011) / housing_types_sp.total_dwellings_SP_2011 < 0.1) & (data_income_group > 0) #I remove the areas where there is informal housing, because dwelling size data is not reliable
        
#Coefficients of the model
listBeta = np.arange(0.1, 0.55, 0.2)
listBasicQ = np.arange(5, 16, 5)

#Utilities
utilityTarget = np.array([300, 1000, 3000, 10000])
listVariation = np.arange(0.5, 2, 0.3)
initUti2 = utilityTarget[1] 
listUti3 = utilityTarget[2] * listVariation
listUti4 = utilityTarget[3] * listVariation

dataRent = data_sp["price"] ** (coeff_a) * (param["depreciation_rate"] + interpolate_interest_rate(spline_interest_rate, 0)) / (coeffKappa * coeff_b ** coeff_b)



[parametersScan, scoreScan, parametersAmenitiesScan, modelAmenityScan, parametersHousing, ~] = ...
    EstimateParametersByScanning(incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup, ...
    dataDensity, selectedDensity, xData, yData, selectedSPForEstimation, tableAmenities, variablesRegression, ...
    listRho, listBeta, listBasicQ, initUti2, listUti3, listUti4);

#Now run the optimization algo with identified value of the parameters
initBeta = parametersScan[0] 
initBasicQ = max(parametersScan[1], 5.1) 

#Utilities
initUti3 = parametersScan[2]
initUti4 = parametersScan[3]

[parameters, scoreTot, parametersAmenities, modelAmenity, parametersHousing, selectedSPRent] = ...
    EstimateParametersByOptimization(incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup, ...
    dataDensity, selectedDensity, xData, yData, selectedSPForEstimation, tableAmenities, variablesRegression, ...
    listRho, initBeta, initBasicQ, initUti2, initUti3, initUti4);

#Generating the map of amenities
modelAmenity
save('./0. Precalculated inputs/modelAmenity', 'modelAmenity')
ImportAmenitiesGrid
amenities = exp(parametersAmenities(2:end)' * table2array(tableAmenitiesGrid(:,variablesRegression))');

#Exporting and saving
utilitiesCorrected = parameters(3:end) ./ exp(parametersAmenities(1));
calibratedUtility_beta = parameters(1);
calibratedUtility_q0 = parameters(2);










def import_amenities_SP():
    print('**************** NEDUM-Cape-Town - Import amenity data at the SP level ****************')

    #Import of the amenity files at the SP level
    amenity_data = pd.read_csv('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Basile data/SP_amenities.csv', sep = ',')

    #Airport cones
    airport_cone = copy.deepcopy(amenity_data.airport_cone)
    airport_cone[airport_cone==55] = 1
    airport_cone[airport_cone==60] = 1
    airport_cone[airport_cone==65] = 1    
    airport_cone[airport_cone==70] = 1
    airport_cone[airport_cone==75] = 1
    
    #Distance to RDP houses
    SP_distance_RDP = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Basile data/SPdistanceRDP.mat')["SP_distance_RDP"].squeeze()

    table_amenities = pd.DataFrame(data = np.transpose(np.array([amenity_data.SP_CODE, amenity_data.distance_distr_parks < 2,
                                                                amenity_data.distance_ocean < 2, ((amenity_data.distance_ocean > 2) & (amenity_data.distance_ocean < 4)),
                                                                amenity_data.distance_world_herit < 2, ((amenity_data.distance_world_herit > 2) & (amenity_data.distance_world_herit < 4)),
                                                                amenity_data.distance_urban_herit < 2, amenity_data.distance_UCT < 2,
                                                                airport_cone, ((amenity_data.slope > 1) & (amenity_data.slope < 5)), amenity_data.slope > 5, 
                                                                amenity_data.distance_train < 2, amenity_data.distance_protected_envir < 2, 
                                                                ((amenity_data.distance_protected_envir > 2) & (amenity_data.distance_protected_envir < 4)),
                                                                SP_distance_RDP, amenity_data.distance_power_station < 2, amenity_data.distance_biosphere_reserve < 2])), columns = ['SP_CODE', 'distance_distr_parks', 'distance_ocean', 'distance_ocean_2_4', 'distance_world_herit', 'distance_world_herit_2_4', 'distance_urban_herit', 'distance_UCT', 'airport_cone2', 'slope_1_5', 'slope_5', 'distance_train', 'distance_protected_envir', 'distance_protected_envir_2_4', 'RDP_proximity', 'distance_power_station', 'distance_biosphere_reserve'])

    return table_amenities

def import_transport_costs(income_2011, param, grid):
        """ Compute travel times and costs """

        #### STEP 1: IMPORT TRAVEL TIMES AND COSTS

        # Import travel times and distances
        transport_times = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Basile data/Transport_times_SP.mat')
        #Import Scenarios
        spline_agricultural_rent, spline_interest_rate, spline_RDP, spline_population_income_distribution, spline_inflation, spline_income_distribution, spline_population, spline_interest_rate, spline_income, spline_minimum_housing_supply, spline_fuel = import_scenarios(income_2011, param, grid)

        #Price per km
        priceTrainPerKMMonth = 0.164 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceTrainFixedMonth = 4.48 * 40 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceTaxiPerKMMonth = 0.785 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceTaxiFixedMonth = 4.32 * 40 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceBusPerKMMonth = 0.522 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceBusFixedMonth = 6.24 * 40 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        inflation = spline_inflation(0)
        infla_2012 = spline_inflation(2012 - param["baseline_year"])
        priceTrainPerKMMonth = priceTrainPerKMMonth * inflation / infla_2012
        priceTrainFixedMonth = priceTrainFixedMonth * inflation / infla_2012
        priceTaxiPerKMMonth = priceTaxiPerKMMonth * inflation / infla_2012
        priceTaxiFixedMonth = priceTaxiFixedMonth * inflation / infla_2012
        priceBusPerKMMonth = priceBusPerKMMonth * inflation / infla_2012
        priceBusFixedMonth = priceBusFixedMonth * inflation / infla_2012
        priceFuelPerKMMonth = spline_fuel(0)
        
        #Fixed costs
        priceFixedVehiculeMonth = 400 
        priceFixedVehiculeMonth = priceFixedVehiculeMonth * inflation / infla_2012
        
        #### STEP 2: TRAVEL TIMES AND COSTS AS MATRIX
        
        #parameters
        numberDaysPerYear = 235
        numberHourWorkedPerDay= 8
        
        #Time by each mode, aller-retour, en minute
        timeOutput = np.empty((transport_times["durationTrain"].shape[0], transport_times["durationTrain"].shape[1], 5))
        timeOutput[:] = np.nan
        timeOutput[:,:,0] = transport_times["distanceCar"] / param["walking_speed"] * 60 * 1.2 * 2
        timeOutput[:,:,0][np.isnan(transport_times["durationCar"])] = np.nan
        timeOutput[:,:,1] = copy.deepcopy(transport_times["durationTrain"])
        timeOutput[:,:,2] = copy.deepcopy(transport_times["durationCar"])
        timeOutput[:,:,3] = copy.deepcopy(transport_times["durationMinibus"])
        timeOutput[:,:,4] = copy.deepcopy(transport_times["durationBus"])

        #Length (km) using each mode
        multiplierPrice = np.empty((timeOutput.shape))
        multiplierPrice[:] = np.nan
        multiplierPrice[:,:,0] = np.zeros((timeOutput[:,:,0].shape))
        multiplierPrice[:,:,1] = transport_times["distanceCar"]
        multiplierPrice[:,:,2] = transport_times["distanceCar"]
        multiplierPrice[:,:,3] = transport_times["distanceCar"]
        multiplierPrice[:,:,4] = transport_times["distanceCar"]

        #Multiplying by 235 (days per year)
        pricePerKM = np.empty(5)
        pricePerKM[:] = np.nan
        pricePerKM[0] = np.zeros(1)
        pricePerKM[1] = priceTrainPerKMMonth*numberDaysPerYear
        pricePerKM[2] = priceFuelPerKMMonth*numberDaysPerYear          
        pricePerKM[3] = priceTaxiPerKMMonth*numberDaysPerYear
        pricePerKM[4] = priceBusPerKMMonth*numberDaysPerYear
        
        #Distances (not useful to calculate price but useful output)
        distanceOutput = np.empty((timeOutput.shape))
        distanceOutput[:] = np.nan
        distanceOutput[:,:,0] = transport_times["distanceCar"]
        distanceOutput[:,:,1] = transport_times["distanceCar"]
        distanceOutput[:,:,2] = transport_times["distanceCar"]
        distanceOutput[:,:,3] = transport_times["distanceCar"]
        distanceOutput[:,:,4] = transport_times["distanceCar"]

        #Monetary price per year
        monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
        trans_monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
        for index2 in range(0, 5):
            monetaryCost[:,:,index2] = pricePerKM[index2] * multiplierPrice[:,:,index2]
            
        monetaryCost[:,:,1] = monetaryCost[:,:,1] + priceTrainFixedMonth * 12 #train (monthly fare)
        monetaryCost[:,:,2] = monetaryCost[:,:,2] + priceFixedVehiculeMonth * 12 #private car
        monetaryCost[:,:,3] = monetaryCost[:,:,3] + priceTaxiFixedMonth * 12 #minibus-taxi
        monetaryCost[:,:,4] = monetaryCost[:,:,4] + priceBusFixedMonth * 12 #bus
        trans_monetaryCost = copy.deepcopy(monetaryCost)
    
        #### STEP 3: COMPUTE PROBA TO WORK IN C, EXPECTED INCOME AND EXPECTED NB OF
        #RESIDENTS OF INCOME GROUP I WORKING IN C


        costTime = (timeOutput * param["time_cost"]) / (60 * numberHourWorkedPerDay) #en h de transport par h de travail
        costTime[np.isnan(costTime)] = 10 ** 2

        return timeOutput, distanceOutput, monetaryCost, costTime
    
def import_employment_data(households_per_income_class, param):
        
    # %% Import data
    TAZ = pd.read_csv('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Basile data/TAZ_amp_2013_proj_centro2.csv') #Number of jobs per Transport Zone (TZ)

    #Number of employees in each TZ for the 12 income classes
    jobsCenters12Class = np.array([np.zeros(len(TAZ.Ink1)), TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink2/2, TAZ.Ink2/2, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink4/3, TAZ.Ink4/3, TAZ.Ink4/3])
     
    codeCentersInitial = TAZ.TZ2013
    xCoord = TAZ.X / 1000
    yCoord = TAZ.Y / 1000
        
    selectedCenters = sum(jobsCenters12Class, 0) > 2500

    #Where we don't have reliable transport data
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

    #Number of workers per group for the selected 
    jobsCentersNgroup = np.zeros((len(xCoord), param["nb_of_income_classes"]))
    for j in range(0, param["nb_of_income_classes"]):
        jobsCentersNgroup[:, j] = np.sum(jobsCenters12Class[param["income_distribution"] == j + 1, :], 0)

    jobsCentersNgroup = jobsCentersNgroup[selectedCenters, :]
    #Rescale to keep the correct global income distribution
    jobsCentersNGroupRescaled = jobsCentersNgroup * households_per_income_class[None, :] / np.nansum(jobsCentersNgroup, 0)

    return jobsCentersNGroupRescaled


        
