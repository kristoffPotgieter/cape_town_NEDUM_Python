# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:54:56 2020

@author: Charlotte Liotta
"""

import numpy as np
import scipy.io
import copy

from equilibrium.functions_dynamic import *
from data import *

#Import Scenarios

spline_agricultural_rent, spline_interest_rate, spline_RDP, spline_population_income_distribution, spline_inflation, spline_income_distribution, spline_population, spline_interest_rate, spline_income, spline_minimum_housing_supply, spline_fuel = import_scenarios(income_2011, param, grid, path_scenarios)

for t_temp in np.arange(0, 30):
    print(t_temp)
    incomeNetOfCommuting, modalShares, ODflows, averageIncome = import_transport_data(grid, param, t_temp, spline_inflation, spline_fuel)
    #np.save("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/SP_year_" + str(t_temp), incomeNetOfCommuting)
    np.save("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_20211103/averageIncome_" + str(t_temp), averageIncome)
    np.save("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_20211103/incomeNetOfCommuting_" + str(t_temp), incomeNetOfCommuting)
    np.save("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_20211103/modalShares_" + str(t_temp), modalShares)
    np.save("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_20211103/ODflows_" + str(t_temp), ODflows)


def import_transport_data(grid, param, yearTraffic, spline_inflation, spline_fuel):
        """ Compute travel times and costs """

        #### STEP 1: IMPORT TRAVEL TIMES AND COSTS

        # Import travel times and distances
        transport_times = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/Basile data/Transport_times_GRID.mat')
             
        #Price per km
        priceTrainPerKMMonth = 0.164 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceTrainFixedMonth = 4.48 * 40 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceTaxiPerKMMonth = 0.785 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceTaxiFixedMonth = 4.32 * 40 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceBusPerKMMonth = 0.522 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        priceBusFixedMonth = 6.24 * 40 * spline_inflation(2011 - param["baseline_year"]) / spline_inflation(2013 - param["baseline_year"])
        inflation = spline_inflation(yearTraffic)
        infla_2012 = spline_inflation(2012 - param["baseline_year"])
        priceTrainPerKMMonth = priceTrainPerKMMonth * inflation / infla_2012
        priceTrainFixedMonth = priceTrainFixedMonth * inflation / infla_2012
        priceTaxiPerKMMonth = priceTaxiPerKMMonth * inflation / infla_2012
        priceTaxiFixedMonth = priceTaxiFixedMonth * inflation / infla_2012
        priceBusPerKMMonth = priceBusPerKMMonth * inflation / infla_2012
        priceBusFixedMonth = priceBusFixedMonth * inflation / infla_2012
        priceFuelPerKMMonth = spline_fuel(yearTraffic)
        if yearTraffic > 8:
            priceFuelPerKMMonth = priceFuelPerKMMonth * 1.2
            #priceBusPerKMMonth = priceBusPerKMMonth * 1.2
            #priceTaxiPerKMMonth = priceTaxiPerKMMonth * 1.2
        #Fixed costs
        priceFixedVehiculeMonth = 400 
        priceFixedVehiculeMonth = priceFixedVehiculeMonth * inflation / infla_2012
        
        #### STEP 2: TRAVEL TIMES AND COSTS AS MATRIX
        
        #parameters
        numberDaysPerYear = 235
        numberHourWorkedPerDay= 8
        annualToHourly = 1 / (8*20*12)
        

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
        param_lambda = param["lambda"].squeeze()

        incomeNetOfCommuting = np.zeros((param["nb_of_income_classes"], transport_times["durationCar"].shape[1]))
        averageIncome = np.zeros((param["nb_of_income_classes"], transport_times["durationCar"].shape[1]))
        modalShares = np.zeros((185, transport_times["durationCar"].shape[1], 5, param["nb_of_income_classes"]))
        ODflows = np.zeros((185, transport_times["durationCar"].shape[1], param["nb_of_income_classes"]))
       
        #income
        incomeGroup, households_per_income_class = compute_average_income(spline_population_income_distribution, spline_income_distribution, param, yearTraffic)
        #income in 2011
        households_per_income_class, incomeGroupRef = import_income_classes_data(param, income_2011)
        #income centers
        income_centers_init = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/0. Precalculated inputs/incomeCentersKeep.mat')['incomeCentersKeep']
        incomeCenters = income_centers_init * incomeGroup / incomeGroupRef
    
        #switch to hourly
        monetaryCost = trans_monetaryCost * annualToHourly #en coût par heure
        monetaryCost[np.isnan(monetaryCost)] = 10**3 * annualToHourly
        incomeCenters = incomeCenters * annualToHourly
        
        xInterp = grid.x
        yInterp = grid.y
        
        #if changes
        #monetaryCost[:, (grid.dist < 15) & (grid.dist > 10), :] = monetaryCost[:, (grid.dist < 15) & (grid.dist > 10), :] * 1.2
        #monetaryCost[:, (grid.dist < 30) & (grid.dist > 22), :] = monetaryCost[:, (grid.dist < 30) & (grid.dist > 22), :] * 0.7
        #costTime[:, (grid.dist < 15) & (grid.dist > 10), :] = costTime[:, (grid.dist < 15) & (grid.dist > 10), :] * 1.2
        #costTime[:, (grid.dist < 30) & (grid.dist > 22), :] = costTime[:, (grid.dist < 30) & (grid.dist > 22), :] * 0.7
        #monetaryCost[:, (grid.dist < 25) & (grid.dist > 22), :] = monetaryCost[:, (grid.dist < 25) & (grid.dist > 22), :] * 0.8
        #costTime[:, (grid.dist < 25) & (grid.dist > 22), :] = costTime[:, (grid.dist < 25) & (grid.dist > 22), :] * 0.8
        #monetaryCost[:, (grid.dist < 11) & (grid.dist > 8), :] = monetaryCost[:, (grid.dist < 11) & (grid.dist > 8), :] * 0.8
        #costTime[:, (grid.dist < 11) & (grid.dist > 8), :] = costTime[:, (grid.dist < 11) & (grid.dist > 8), :] * 0.8
        #monetaryCost[:, (grid.dist < 22) & (grid.dist > 14), :] = monetaryCost[:, (grid.dist < 22) & (grid.dist > 14), :] * 0.8
        #costTime[:, (grid.dist < 22) & (grid.dist > 14), :] = costTime[:, (grid.dist < 22) & (grid.dist > 14), :] * 0.8
        
        for j in range(0, param["nb_of_income_classes"]):
                    
            #Household size varies with transport costs
            householdSize = param["household_size"][j]
            whichCenters = incomeCenters[:,j] > -100000
            incomeCentersGroup = incomeCenters[whichCenters, j]
           
            #Transport costs and employment allocation (cout par heure)
            #transportCostModes = householdSize * (monetaryCost[whichCenters,:,:] + (costTime[whichCenters,:,:] * np.repeat(np.transpose(np.matlib.repmat(incomeCentersGroup, costTime.shape[1], 1))[:, :, np.newaxis], 5, axis=2)))
            transportCostModes = (householdSize * monetaryCost[whichCenters,:,:] + (costTime[whichCenters,:,:] * incomeCentersGroup[:, None, None]))
                
            #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
            valueMax = (np.min(param_lambda * transportCostModes, axis = 2) - 500) #-500
        
            #Modal shares
            modalShares[whichCenters,:,:,j] = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]
            #destination, origin, mode, income class
            
            #Transport costs
            #transportCost = - 1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + np.repeat(valueMax[:, :, np.newaxis], 5, axis=2)), 2) - valueMax))
            #transportCost = - 1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + np.repeat(valueMax[:, :, np.newaxis], 5, axis=2)), 2) - valueMax))
            transportCost = - 1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)
            #transportCost = -1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax) #Eq 2. Somme sur m de 1 à 5.

            #minIncome is also to prevent diverging exponentials
            minIncome = np.nanmax(param_lambda * (incomeCentersGroup[:, None] - transportCost), 0) - 700
            #minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCentersFull[:, None] - transportCost))) - 500

            #OD flows
            #ODflows[whichCenters,:,j] = np.exp(param_lambda * (np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1)) - transportCost) - minIncome) / np.transpose(np.matlib.repmat(np.nansum(np.exp(param_lambda * (np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1)) - transportCost) - minIncome), 1), 24014, 1))
            #ODflows[whichCenters,:,j, index] = np.exp(param_lambda * ((np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))) - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * ((np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))) - transportCost) - minIncome), 0)
            #ODflows[whichCenters,:,j,index] = np.exp(param_lambda * (incomeCentersGroup[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCentersGroup[:, None] - transportCost) - minIncome), 0)[None, :]
            ODflows[whichCenters,:,j] = np.exp(param_lambda * (incomeCentersGroup[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCentersGroup[:, None] - transportCost) - minIncome), 0)[None, :]

            #Income net of commuting (correct formula)
            #incomeNetOfCommuting[j,:, index] = 1 /param_lambda * (np.log(np.nansum(np.exp(param_lambda * ((np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))) - transportCost) - minIncome), 0)) + minIncome)
            incomeNetOfCommuting[j,:] = 1/param_lambda * (np.log(np.nansum(np.exp(param_lambda * (incomeCentersGroup[:, None] - transportCost) - minIncome), 0)) + minIncome)
        
            #Average income earned by a worker
            #averageIncome[j,:, index] = np.nansum(ODflows[whichCenters,:,j,index] * incomeCentersGroup[:, None], 0)
            averageIncome[j,:] = np.nansum(ODflows[whichCenters,:,j] * incomeCentersGroup[:, None], 0)

        incomeNetOfCommuting = incomeNetOfCommuting / annualToHourly
        averageIncome = averageIncome / annualToHourly
        
        return incomeNetOfCommuting, modalShares, ODflows, averageIncome