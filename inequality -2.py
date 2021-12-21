# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:07:25 2021

@author: charl
"""

#### Average distance to employment centers

#par income class
simulation_households = np.nansum(np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[0, : , :, :], 0)
transport_times = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/Basile data/Transport_times_GRID.mat')
transport_times = transport_times["distanceCar"]

ODflows_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/ODflows_0.npy")
ODflows_noct = ODflows_noct.transpose() * simulation_households[:, :, np.newaxis]

for j in np.arange(4):
    print(np.average(transport_times.transpose()[~np.isnan(transport_times.transpose())], weights = ODflows_noct[j, :, :][~np.isnan(transport_times.transpose())]))

#par housing type
simulation_households = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[0, : , :, :]
transport_times = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/Basile data/Transport_times_GRID.mat')
transport_times = transport_times["distanceCar"]
ODflows_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/ODflows_0.npy")
hh = ODflows_noct[:, :, :, np.newaxis] * simulation_households.transpose()[np.newaxis, :, :, :]

hh = np.nansum(hh, 2)

for j in np.arange(4):
    print(np.average(transport_times[~np.isnan(transport_times)], weights = hh[:, :, j][~np.isnan(transport_times)]))
#15.4, 9.4, 13.8, 13.0

hh = np.nansum(hh, 3)

for j in np.arange(4):
    print(np.average(transport_times[~np.isnan(transport_times)], weights = hh[:, :, j][~np.isnan(transport_times)]))
#12.4, 18.1, 13.4, 16.1

#### STEP 1: Change in transportation costs of car users, keeping employment centers and transportation modes constant

tcost_diff = np.empty((4, 24014))

for j in np.arange(4):
    ODflows_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/ODflows_9.npy")
    modalShares_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/modalShares_9.npy")
    modalShares_noct =modalShares_noct[:, :, 2:5, j] #modal share car 
    ODflows_noct = ODflows_noct[:, :, j]
    vec_share_car = ODflows_noct[:, :, np.newaxis] * modalShares_noct #part des gens résidant en un lieu qui vont dans un employment center en voiture

    transportCostModes_noct = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, j, 0)
    transportCostModes_ct = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, j, 1)

    transportCostModes_noct = transportCostModes_noct[:, :, 2:5]
    transportCostModes_ct = transportCostModes_ct[:, :, 2:5]

    finalTransportCost_noct = np.empty(24014)
    for i in np.arange(0, 24014):
        finalTransportCost_noct[i] = np.average(transportCostModes_noct[:, i, :][~np.isinf(transportCostModes_noct[:, i, :])], weights = vec_share_car[:, i, :][~np.isinf(transportCostModes_noct[:, i, :])])
    
    finalTransportCost_ct = np.empty(24014)
    for i in np.arange(0, 24014):
        finalTransportCost_ct[i] = np.average(transportCostModes_ct[:, i, :][~np.isinf(transportCostModes_ct[:, i, :])], weights = vec_share_car[:, i, :][~np.isinf(transportCostModes_noct[:, i, :])])

    simulation_households = np.nansum(np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[9, : , j, :], 0)
    tcost_diff[j] = 100 * (finalTransportCost_ct - finalTransportCost_noct) / finalTransportCost_noct
    tcost_diff[j][simulation_households == 0] = np.nan

df = pd.DataFrame(tcost_diff)
df.transpose().to_excel("C:/Users/charl/OneDrive/Bureau/inequalityCapeTown/Tax on minibuses/step1-carusecost/tcost_diff.xlsx")


#B- Diff income


finalTransportCost_noct = np.empty((4, 24014))
finalTransportCost_ct = np.empty((4, 24014))

for j in np.arange(4):
    ODflows_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/ODflows_9.npy")
    modalShares_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/modalShares_9.npy")
    modalShares_noct = modalShares_noct[:, :, 2:5, j] #modal share car 
    ODflows_noct = ODflows_noct[:, :, j]
    vec_share_car = ODflows_noct[:, :, np.newaxis] * modalShares_noct #part des gens résidant en un lieu qui vont dans un employment center en voiture

    transportCostModes_noct = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, j, 0)
    transportCostModes_ct = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, j, 1)

    transportCostModes_noct = transportCostModes_noct[:, :, 2:5]
    transportCostModes_ct = transportCostModes_ct[:, :, 2:5]

    
    for i in np.arange(0, 24014):
        finalTransportCost_noct[j, i] = np.average(transportCostModes_noct[:, i, :][~np.isinf(transportCostModes_noct[:, i, :])], weights = vec_share_car[:, i][~np.isinf(transportCostModes_noct[:, i, :])])
    
    
    for i in np.arange(0, 24014):
        finalTransportCost_ct[j, i] = np.average(transportCostModes_ct[:, i, :][~np.isinf(transportCostModes_ct[:, i, :])], weights = vec_share_car[:, i][~np.isinf(transportCostModes_ct[:, i, :])])

    #simulation_households = np.nansum(np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[9, : , j, :], 0)
    #tcost_diff[j] = 100 * (finalTransportCost_ct - finalTransportCost_noct) / finalTransportCost_noct
    #tcost_diff[j][simulation_households == 0] = np.nan

#df = pd.DataFrame(tcost_diff)
#df.transpose().to_excel("C:/Users/charl/OneDrive/Bureau/tcost_diff.xlsx")

averageIncome_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/averageIncome_9.npy")

finalTransportCost_noct = (8*20*12) * finalTransportCost_noct
finalTransportCost_ct = (8*20*12) * finalTransportCost_ct

residual_income_noct = averageIncome_noct - finalTransportCost_noct
residual_income_ct = averageIncome_noct - finalTransportCost_ct

diff = 100 * (residual_income_ct - residual_income_noct) / residual_income_noct

simulation_households = np.nansum(np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[9, : , :, :], 0)
diff[simulation_households == 0] = np.nan

df = pd.DataFrame(diff)
df.transpose().to_excel("C:/Users/charl/OneDrive/Bureau/inequalityCapeTown/Tax on minibuses/step1-carusecost/residual_income.xlsx")


#By income
for j in np.arange(4):
    print(np.average(diff[j, :][~np.isnan(diff[j, :])], weights = simulation_households[j, :][~np.isnan(diff[j, :])]))
    #-2.20, -1.15, -0.34, -0.14

np.nansum(simulation_households, 1) / np.nansum(simulation_households) #array([0.38609923, 0.16705371, 0.29016323, 0.15668383])

diff_average = np.zeros(24014)

for i in np.arange(24014):
    if np.nansum(simulation_households[:, i]) > 0:
        diff_average[i] = np.average(diff[:, i][~np.isnan(diff[:, i])], weights = simulation_households[:, i][~np.isnan(diff[:, i])])

total_hh = np.nansum(simulation_households, 0)

df = pd.DataFrame(np.array([diff_average, total_hh, grid.dist])).transpose()
df.columns = ['diff_average', 'total_hh', 'grid_dist']
df = df.sort_values('grid_dist')

x = 0
sum_hh = 0
while sum_hh < 0.2 * np.nansum(total_hh):
    x = x + 1
    sum_hh = np.nansum(df.total_hh.iloc[0:x])

closest = df.iloc[0:x, :]
np.average(closest.diff_average, weights = closest.total_hh) #-0.33
np.average(closest.grid_dist, weights = closest.total_hh) #7.96

df = pd.DataFrame(np.array([diff_average, total_hh, grid.dist])).transpose()
df.columns = ['diff_average', 'total_hh', 'grid_dist']
df = df.sort_values('grid_dist', ascending = False)

x = 0
sum_hh = 0
while sum_hh < 0.2 * np.nansum(total_hh):
    x = x + 1
    sum_hh = np.nansum(df.total_hh.iloc[0:x])

closest = df.iloc[0:x, :]
np.average(closest.diff_average, weights = closest.total_hh) #-1.51
np.average(closest.grid_dist, weights = closest.total_hh) #32.99


def compute_transport_cost_mode(grid, param, yearTraffic, spline_inflation, spline_fuel, income_class, option_carbon_tax):
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
        if (yearTraffic > 8) & (option_carbon_tax  == 1):
            priceFuelPerKMMonth = priceFuelPerKMMonth * 1.2
            priceBusPerKMMonth = priceBusPerKMMonth * 1.2
            priceTaxiPerKMMonth = priceTaxiPerKMMonth * 1.2
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
        
        householdSize = param["household_size"][income_class]
        #whichCenters = incomeCenters[:,income_class] > -100000
        #print(sum(whichCenters))
        incomeCentersGroup = incomeCenters[:, income_class]
        transportCostModes = (householdSize * monetaryCost[:,:,:] + (costTime[:,:,:] * incomeCentersGroup[:, None, None]))
        return transportCostModes


#### STEP 2: Changes in income net of transportation costs, assuming that people can change transportation mode and employment center

net_income_diff = np.empty((4, 24014))

for j in np.arange(4):
    incomeNetOfCommuting2020_ct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_taxi_bus/incomeNetOfCommuting_9.npy")
    incomeNetOfCommuting2020_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/incomeNetOfCommuting_9.npy")

    incomeNetOfCommuting2020_ct[incomeNetOfCommuting2020_ct <0] = 0
    incomeNetOfCommuting2020_noct[incomeNetOfCommuting2020_noct <0] = 0

    simulation_households = np.nansum(np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[9, : , j, :], 0)
    #A refaire

    diff = 100 * (incomeNetOfCommuting2020_ct[j] - incomeNetOfCommuting2020_noct[j]) / incomeNetOfCommuting2020_noct[j]
    diff[simulation_households == 0] = np.nan
    net_income_diff[j] = diff
    
df = pd.DataFrame(net_income_diff).transpose()
df.to_excel("C:/Users/charl/OneDrive/Bureau/inequalityCapeTown/Tax on minibuses/step2-incomenetofcommutingcost/net_income_v2.xlsx")


### STEP 3

#Rents
simulation_rent_noct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_rent.npy')[9, :]
simulation_rent_ct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_rent.npy')[9, :]

simulation_households_noct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[9, : , :, :]

simulation_households_ct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_households.npy')[9, : , :, :]
#simulation_households_noct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_10cts' + '/simulation_households.npy')[9, : , :, :]
#WHAT???
simulation_rent_noct[(np.nansum(simulation_households_noct[:, :, :], 1) == 0) | (np.nansum(simulation_households_ct[:, :, :], 1) == 0)] = np.nan
simulation_rent_noct[(np.isnan(np.nansum(simulation_households_noct[:, :, :], 1))) | (np.isnan(np.nansum(simulation_households_ct[:, :, :], 1)))] = np.nan


simulation_rent_ct[(np.nansum(simulation_households_noct[:, :, :], 1) == 0) | (np.nansum(simulation_households_ct[:, :, :], 1) == 0)] = np.nan
simulation_rent_ct[(np.isnan(np.nansum(simulation_households_noct[:, :, :], 1))) | (np.isnan(np.nansum(simulation_households_ct[:, :, :], 1)))] = np.nan

diff_rent = 100 * (simulation_rent_ct - simulation_rent_noct) / simulation_rent_noct

df = pd.DataFrame(diff_rent)
df.transpose().to_excel("C:/Users/charl/OneDrive/Bureau/inequalityCapeTown/Tax on minibuses/step3-housingprice/diff_rent.xlsx")

#Changes in the budget dedicated to housing + transportation

#income net of transportation cost: easy
incomeNetOfCommuting2020_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/incomeNetOfCommuting_9.npy")
incomeNetOfCommuting2020_ct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_taxi_bus/incomeNetOfCommuting_9.npy")
incomeNetOfCommuting2020_ct[incomeNetOfCommuting2020_ct <0] = 0
incomeNetOfCommuting2020_noct[incomeNetOfCommuting2020_noct <0] = 0

#Rents: more complicated because we have i) to use dsize ii) match housing type and income class
simulation_rent_noct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_rent.npy')[9, :]
simulation_rent_ct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_rent.npy')[9, :]

simulation_dsize_noct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_dwelling_size.npy')[9, :]
simulation_dsize_ct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_dwelling_size.npy')[9, :]

budget_housing_by_housing_type_ct = simulation_rent_ct * simulation_dsize_noct
budget_housing_by_housing_type_noct = simulation_rent_noct * simulation_dsize_noct
#Est-ce qu'on autorise la taille des logements à bouger ? Non pour l'instant

#A convertir en income class
simulation_households_ct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_households.npy')[9, : , :, :]
simulation_households_noct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[9, : , :, :]
#No ct uniquement car on suppose que les gens ne bougent pas
#housing type, income class, location

budget_housing_by_income_class_ct = np.empty((4, 24014))
budget_housing_by_income_class_noct = np.empty((4, 24014))

for i in range(0, 24014):
    for j in range(0, 4):
        if (np.nansum(simulation_households_noct[:, j, i]) == 0):
            budget_housing_by_income_class_ct[j, i] = np.nan
            budget_housing_by_income_class_noct[j, i] = np.nan
        else:
            budget_housing_by_income_class_ct[j, i] = np.average(budget_housing_by_housing_type_ct[:, i], weights = simulation_households_noct[:, j, i])
            budget_housing_by_income_class_noct[j, i] = np.average(budget_housing_by_housing_type_noct[:, i], weights = simulation_households_noct[:, j, i])
    
averageIncome_ct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_taxi_bus/averageIncome_9.npy")
averageIncome_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/averageIncome_9.npy")

transportion_budget_noct = averageIncome_noct - incomeNetOfCommuting2020_noct
transportion_budget_ct = averageIncome_ct - incomeNetOfCommuting2020_ct

basic_need_noct = transportion_budget_noct + budget_housing_by_income_class_noct
basic_need_ct = transportion_budget_ct + budget_housing_by_income_class_ct

diff = 100 * (basic_need_ct - basic_need_noct) / basic_need_noct

df = pd.DataFrame(diff)
df.transpose().to_excel("C:/Users/charl/OneDrive/Bureau/inequalityCapeTown/Tax on minibuses/step3-housingprice/basic_need_budget_nodsize_change.xlsx")

simulation_households = np.nansum(simulation_households_noct, 0)
diff[simulation_households == 0] = np.nan

#By income
for j in np.arange(4):
    print(np.average(diff[j, :][~np.isnan(diff[j, :])], weights = simulation_households[j, :][~np.isnan(diff[j, :])]))
    #-4.94, 16.91, 11.03, -1.56
    #-4.94, -0.61, 0.73, 0.12

diff_average = np.zeros(24014)

for i in np.arange(24014):
    if (np.nansum(simulation_households[:, i]) > 0) & (~np.isnan(np.nansum(simulation_households[:, i]))) & (np.nansum(diff[:, i]) > 0):
        diff_average[i] = np.average(diff[:, i][~np.isnan(diff[:, i])], weights = simulation_households[:, i][~np.isnan(diff[:, i])])

total_hh = np.nansum(simulation_households, 0)

df = pd.DataFrame(np.array([diff_average, total_hh, grid.dist])).transpose()
df.columns = ['diff_average', 'total_hh', 'grid_dist']
df = df.sort_values('grid_dist')

x = 0
sum_hh = 0
while sum_hh < 0.2 * np.nansum(total_hh):
    x = x + 1
    sum_hh = np.nansum(df.total_hh.iloc[0:x])

closest = df.iloc[0:x, :]
np.average(closest.diff_average, weights = closest.total_hh) #4.60 / 0.73
np.average(closest.grid_dist, weights = closest.total_hh) #7.96

df = pd.DataFrame(np.array([diff_average, total_hh, grid.dist])).transpose()
df.columns = ['diff_average', 'total_hh', 'grid_dist']
df = df.sort_values('grid_dist', ascending = False)

x = 0
sum_hh = 0
while sum_hh < 0.2 * np.nansum(total_hh):
    x = x + 1
    sum_hh = np.nansum(df.total_hh.iloc[0:x])

closest = df.iloc[0:x, :]
np.average(closest.diff_average, weights = closest.total_hh) #1.22 / 0.49
np.average(closest.grid_dist, weights = closest.total_hh) #32.99




### STEP 4: Welfare losses

simulation_utility_noct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_utility.npy')[9, :]
simulation_utility_ct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_utility.npy')[9, :]

100 * (simulation_utility_ct - simulation_utility_noct) / simulation_utility_noct

### QUANTIFYING EMISSIONS REDUCTION

pkm_noct = np.empty((4, 5))
pkm_ct = np.empty((4, 5))

transport_times = scipy.io.loadmat('C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/Basile data/Transport_times_GRID.mat')
    
for j in np.arange(4):
    modalShares_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/modalShares_9.npy")[:, :, :, j]
    ODflows_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/ODflows_9.npy")[:, :, j]



    simulation_households_noct = np.nansum(np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[9, : , j, :], 0)

    ODflows_absolutenb = (simulation_households_noct[np.newaxis, :] * ODflows_noct)
    modalShares_absolutenb = ODflows_absolutenb[:, :, np.newaxis] * modalShares_noct
    
    nb_p_km = modalShares_absolutenb * transport_times["distanceCar"][:, :, np.newaxis]
    
    pkm_noct[j] = np.nansum(np.nansum(nb_p_km, 0), 0)
    
for j in np.arange(4):
    modalShares_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_taxi_bus/modalShares_9.npy")[:, :, :, j]
    ODflows_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_taxi_bus/ODflows_9.npy")[:, :, j]

    simulation_households_noct = np.nansum(np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_households.npy')[9, : , j, :], 0)

    ODflows_absolutenb = (simulation_households_noct[np.newaxis, :] * ODflows_noct)
    modalShares_absolutenb = ODflows_absolutenb[:, :, np.newaxis] * modalShares_noct
    
    nb_p_km = modalShares_absolutenb * transport_times["distanceCar"][:, :, np.newaxis]
    
    pkm_ct[j] = np.nansum(np.nansum(nb_p_km, 0), 0)
    

100 * (np.nansum(pkm_ct, 0) - np.nansum(pkm_noct, 0)) /  np.nansum(pkm_noct, 0)

em_ct = np.nansum(np.nansum(pkm_ct, 0) * np.array([0, 0.05, 0.150, 0.06, 0.04]))
em_noct = np.nansum(np.nansum(pkm_noct, 0) * np.array([0, 0.05, 0.150, 0.06, 0.04]))

100 * (em_ct -em_noct) /  em_noct