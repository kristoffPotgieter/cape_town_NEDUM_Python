# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:33:53 2021

@author: charl
"""

A = amenities

B_formal = 1
B_IS = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/fluvial_and_pluvial/param_pockets.npy')
B_IB = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/fluvial_and_pluvial/param_backyards.npy')
B = np.zeros((24014, 4))
B[:, 0] = 1
B[:, 3] = 1
B[:, 1] = B_IB
B[:, 2] = B_IS

q0 = np.zeros((24014, 4))
#q0[:, 0] = param['q0']
q0 = q0 + param['q0']

interest_rate = interpolate_interest_rate(spline_interest_rate, 9)

structure_value = np.zeros((24014, 4))
structure_value[:, 1] = (param["informal_structure_value"] * (interest_rate + param["depreciation_rate"])) * (spline_inflation(9) / spline_inflation(0))
structure_value[:, 2] = (param["informal_structure_value"] * (interest_rate + param["depreciation_rate"])) * (spline_inflation(9) / spline_inflation(0))

def compute_array_utility(DWELLING_SIZE, RENTS, NET_INCOME, A, B, q0, structure_value):
    array_utility = np.zeros((4, 4, 24014))
    for income_class in np.arange(4):
        for housing_type in np.arange(4):        
            array_utility[housing_type, income_class, :] = A * B[:, housing_type] * ((DWELLING_SIZE[housing_type, :] - q0[:, housing_type]) ** param["beta"]) * (((NET_INCOME[income_class, :] - (DWELLING_SIZE[housing_type, :] * RENTS[housing_type, :]) - structure_value[:, housing_type])) ** param["alpha"])
    return array_utility

R_noct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_rent.npy')[9, :]
q_noct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_dwelling_size.npy')[9, :]
income_net_of_tcost_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/incomeNetOfCommuting_9.npy")
simulation_households_noct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_households.npy')[9, : , :, :]
income_net_of_tcost_noct[income_net_of_tcost_noct < 0] = 0

R_ct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_rent.npy')[9, :]
q_ct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_dwelling_size.npy')[9, :]
income_net_of_tcost_ct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/carbon_tax_car_taxi_bus/incomeNetOfCommuting_9.npy")
simulation_households_ct = np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_households.npy')[9, : , :, :]
income_net_of_tcost_ct[income_net_of_tcost_ct < 0] = 0

housing_supply = (param["alpha"] * (param["RDP_size"] + param["backyard_size"] - param["q0"]) / (param["backyard_size"])) - (param["beta"] * (income_net_of_tcost_noct[0,:]) / (param["backyard_size"] * R_noct[1, :]))
housing_supply[np.isinf(housing_supply)] = np.nan
housing_supply[R_noct[1, :] == 0] = 0
housing_supply = np.minimum(housing_supply, 1)
housing_supply = np.maximum(housing_supply, 0)
housing_supply[np.isnan(housing_supply)] = 0
#q_noct[3, :] = 40 + 70 *(1 - housing_supply)
#income_net_of_tcost_noct[3, :] = income_net_of_tcost_noct[3, :] + ((70  *(1 - housing_supply)) * R_noct[1, :])

housing_supply = (param["alpha"] * (param["RDP_size"] + param["backyard_size"] - param["q0"]) / (param["backyard_size"])) - (param["beta"] * (income_net_of_tcost_ct[0,:]) / (param["backyard_size"] * R_ct[1, :]))
housing_supply[np.isinf(housing_supply)] = np.nan
housing_supply[R_ct[1, :] == 0] = 0
housing_supply = np.minimum(housing_supply, 1)
housing_supply = np.maximum(housing_supply, 0)
housing_supply[np.isnan(housing_supply)] = 0
#q_ct[3, :] = 40+70 *(1 - housing_supply)
#income_net_of_tcost_ct[3, :] = income_net_of_tcost_ct[3, :] + ((70  *(1 - housing_supply)) * R_ct[1, :])

def utility_by_income_class(array_utility, simulation_households):
    
    inc_class_1 = np.average(array_utility[:, 0, :][~np.isnan(array_utility[:, 0, :])], weights = simulation_households[:, 0, :][~np.isnan(array_utility[:, 0, :])])
    inc_class_2 = np.average(array_utility[:, 1, :][~np.isnan(array_utility[:, 1, :])], weights = simulation_households[:, 1, :][~np.isnan(array_utility[:, 1, :])])
    inc_class_3 = np.average(array_utility[:, 2, :][~np.isnan(array_utility[:, 2, :])], weights = simulation_households[:, 2, :][~np.isnan(array_utility[:, 2, :])])
    inc_class_4 = np.average(array_utility[:, 3, :][~np.isnan(array_utility[:, 3, :])], weights = simulation_households[:, 3, :][~np.isnan(array_utility[:, 3, :])])
    return np.array([inc_class_1, inc_class_2, inc_class_3, inc_class_4])

def utility_by_housing_type(array_utility, simulation_households):
    FP = np.average(array_utility[0, :, :][~np.isnan(array_utility[0, :, :])], weights = simulation_households[0, :, :][~np.isnan(array_utility[0, :, :])])
    IB = np.average(array_utility[1, :, :][~np.isnan(array_utility[1, :, :])], weights = simulation_households[1, :, :][~np.isnan(array_utility[1, :, :])])
    IS = np.average(array_utility[2, :, :][~np.isnan(array_utility[2, :, :])], weights = simulation_households[2, :, :][~np.isnan(array_utility[2, :, :])])
    FS = np.average(array_utility[3, :, :][~np.isnan(array_utility[3, :, :])], weights = simulation_households[3, :, :][~np.isnan(array_utility[3, :, :])])
    return np.array([FP, IB, IS, FS])

##### INITIAL STATE
np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'inequality_reference_scenario_20210806' + '/simulation_utility.npy')[9, :]


array_utility = compute_array_utility(q_noct, R_noct, income_net_of_tcost_noct, A, B, q0, structure_value)
#housing type, income class, 24014

array_utility[simulation_households_noct == 0] = np.nan

utility_income_class_init = utility_by_income_class(array_utility, simulation_households_noct)
utility_housing_type_init = utility_by_housing_type(array_utility, simulation_households_noct)

#ajout des revenus, et q= 40: 3154
#;ajout des revenus et de q: 3721
#ajout de q mais pas des revenus: 3721
#rien du tout 3154


#### CHANGE IN TRANSPORTATION COSTS

ODflows_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/ODflows_9.npy")
modalShares_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/modalShares_9.npy")
weighting_noct = ODflows_noct[:, :, np.newaxis, :] * modalShares_noct #part des gens résidant en un lieu qui vont dans un employment center en voiture
transportCostModes_ct_0 = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, 0, 1)
transportCostModes_ct_1 = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, 1, 1)
transportCostModes_ct_2 = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, 2, 1)
transportCostModes_ct_3 = compute_transport_cost_mode(grid, param, 9, spline_inflation, spline_fuel, 3, 1)
transportCostModes_ct = np.array([transportCostModes_ct_0, transportCostModes_ct_1, transportCostModes_ct_2, transportCostModes_ct_3])

finalTransportCost_ct = np.empty((4, 24014))
for i in np.arange(0, 24014):
    for j in np.arange(0, 4):
        finalTransportCost_ct[j, i] = np.average(transportCostModes_ct[j, :, i, :][~np.isinf(transportCostModes_ct[j, :, i, :])], weights = weighting_noct[:, i, :, j][~np.isinf(transportCostModes_ct[j, :, i, :])])

averageIncome_noct = np.load("C:/Users/charl/OneDrive/Bureau/cape_town/2. Data/precalculated_transport/no_carbon_tax/averageIncome_9.npy")
finalTransportCost_ct = (8*20*12) * finalTransportCost_ct
residual_income_ct = averageIncome_noct - finalTransportCost_ct
residual_income_ct[residual_income_ct < 0] = 0

#np.nanmin(100 * (residual_income_ct[(residual_income_ct > 0) & (income_net_of_tcost_ct> 0)] - income_net_of_tcost_ct[(residual_income_ct > 0) & (income_net_of_tcost_ct> 0)]) / income_net_of_tcost_ct[(residual_income_ct > 0) & (income_net_of_tcost_ct> 0)])

array_utility = compute_array_utility(q_noct, R_noct, residual_income_ct, A, B, q0, structure_value)
#housing type, income class, 24014

utility_income_class_change_tcost = utility_by_income_class(array_utility, simulation_households_noct)
utility_housing_type_change_tcost = utility_by_housing_type(array_utility, simulation_households_noct)


#### CHANGE IN TRANSPORTATION COSTS, MODES AND EMPLOYMENT CENTERS

array_utility = compute_array_utility(q_noct, R_noct, income_net_of_tcost_ct, A, B, q0, structure_value)
#housing type, income class, 24014

utility_income_class_change_tcostv2 = utility_by_income_class(array_utility, simulation_households_noct)
utility_housing_type_change_tcostv2 = utility_by_housing_type(array_utility, simulation_households_noct)


#### CHANGE IN TRANSPORTATION COSTS, MODES, EMPLOYMENT CENTERS AND RENTS

array_utility = compute_array_utility(q_noct, R_ct, income_net_of_tcost_ct, A, B, q0, structure_value)
#housing type, income class, 24014

utility_income_class_change_rents = utility_by_income_class(array_utility, simulation_households_noct)
utility_housing_type_change_rents = utility_by_housing_type(array_utility, simulation_households_noct)

#### CHANGE IN TRANSPORTATION COSTS, MODES, EMPLOYMENT CENTERS, RENTS AND DWELLING SIZE

array_utility = compute_array_utility(q_ct, R_ct, income_net_of_tcost_ct, A, B, q0, structure_value)
#housing type, income class, 24014

utility_income_class_change_dsize = utility_by_income_class(array_utility, simulation_households_noct)
utility_housing_type_change_dsize = utility_by_housing_type(array_utility, simulation_households_noct)


#### FINAL STATE
np.load('C:/Users/charl/OneDrive/Bureau/cape_town/4. Sorties/' + 'carbon_tax_car_taxi_bus' + '/simulation_utility.npy')[9, :]

array_utility = compute_array_utility(q_ct, R_ct, income_net_of_tcost_ct, A, B, q0, structure_value)
#housing type, income class, 24014


utility_income_class_final = utility_by_income_class(array_utility, simulation_households_ct)
utility_housing_type_final = utility_by_housing_type(array_utility, simulation_households_ct)

100 * (utility_income_class_final - utility_income_class_init) / utility_income_class_init
100 * (utility_housing_type_final - utility_housing_type_init) / utility_housing_type_init


100 * (utility_income_class_change_tcost - utility_income_class_init) / utility_income_class_init
100 * (utility_income_class_change_tcostv2 - utility_income_class_change_tcost) / utility_income_class_change_tcost
100 * (utility_income_class_change_rents - utility_income_class_change_tcostv2) / utility_income_class_change_tcostv2
100 * (utility_income_class_final - utility_income_class_change_rents) / utility_income_class_change_rents

100 * (utility_housing_type_change_tcost - utility_housing_type_init) / utility_housing_type_init
100 * (utility_housing_type_change_tcostv2 - utility_housing_type_change_tcost) / utility_housing_type_change_tcost
100 * (utility_housing_type_change_rents - utility_housing_type_change_tcostv2) / utility_housing_type_change_tcostv2
100 * (utility_housing_type_final - utility_housing_type_change_rents) / utility_housing_type_change_rents



#array([-0.56497585, -0.97911525, -0.34530937, -0.06237502])
#array([-0.77259484, -0.97911525, -0.34530937, -0.06249169])