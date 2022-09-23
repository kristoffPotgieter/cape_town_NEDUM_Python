---
date: '2022-09-23T09:22:05.479Z'
docname: api_ref
images: {}
path: /api-ref
title: API reference
---

# API reference

The API reference gives an overview of Res-IRF implementation.


* Buildings module details implementation of HousingStock class.


* Policies module presents implementation of PublicPolicies class.

## Inputs

### Set parameters and options

Created on Tue Oct 27 15:50:59 2020.

@author: Charlotte Liotta


### inputs.parameters_and_options.compute_agricultural_rent(rent, scale_fact, interest_rate, param, options)
Convert land price into real estate price for land.


### inputs.parameters_and_options.import_construction_parameters(param, grid, housing_types_sp, dwelling_size_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land, interest_rate, options)
Update parameters with values for construction.


### inputs.parameters_and_options.import_options()
Import default options.


### inputs.parameters_and_options.import_param(path_precalc_inp, path_outputs, path_folder, options)
Import default parameters.

### Import data

Created on Tue Oct 27 15:57:41 2020.

@author: Charlotte Liotta


### inputs.data.compute_fraction_capital_destroyed(d, type_flood, damage_function, housing_type, options)
Define function used to get fraction of capital destroyed by floods.


### inputs.data.convert_income_distribution(income_distribution, grid, path_data, data_sp)
Import SP data for income distribution in grid form.


### inputs.data.gen_small_areas_to_grid(grid, grid_intersect, small_area_data, small_area_code, unit)
Convert SAL/SP to grid dimensions.


### inputs.data.import_amenities(path_precalc_inp, options)
Import amenity index for each pixel.


### inputs.data.import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, t)
Return pixel share for housing scenarios, weighted by max building %.


### inputs.data.import_full_floods_data(options, param, path_folder, housing_type_data)
Add fraction of capital destroyed by floods to initial floods data.


### inputs.data.import_grid(path_data)
Import pixel coordinates and distances to center.


### inputs.data.import_households_data(path_precalc_inp)
Import geographic data with class distributions for households.


### inputs.data.import_housing_limit(grid, param)
Return height limit within and out of historic city radius.


### inputs.data.import_hypothesis_housing_type()
Import dummies to select income classes into housing types.


### inputs.data.import_income_classes_data(param, path_data)
Import population and average income per income class in the model.


### inputs.data.import_init_floods_data(options, param, path_folder)
Import initial floods data and damage functions.


### inputs.data.import_land_use(grid, options, param, data_rdp, housing_types, housing_type_data, path_data, path_folder)
Return linear regression spline estimates for housing building paths.


### inputs.data.import_macro_data(param, path_scenarios, path_folder)
Import interest rate and population per housing type.


### inputs.data.import_sal_data(grid, path_folder, path_data, housing_type_data)
Import SAL data for population density by housing type.


### inputs.data.import_transport_data(grid, param, yearTraffic, households_per_income_class, average_income, spline_inflation, spline_fuel, spline_population_income_distribution, spline_income_distribution, path_precalc_inp, path_precalc_transp, dim, options)
Compute job center distribution, commuting and net income.


### inputs.data.infer_WBUS2_depth(housing_types, param, path_floods)
Update CoCT flood data with FATHOM flood depth (deprecated).

## Equilibrium

### Compute equilibrium

Created on Mon Nov  2 11:21:21 2020.

@author: Charlotte Liotta


### equilibrium.compute_equilibrium.compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, construction_param, income_2011)
Determine static equilibrium allocation from iterative algorithm.

### Run simulations

Created on Fri Nov  6 16:23:16 2020.

@author: Charlotte Liotta


### equilibrium.run_simulations.run_simulation(t, options, param, grid, initial_state_utility, initial_state_error, initial_state_households, initial_state_households_housing_types, initial_state_housing_supply, initial_state_household_centers, initial_state_average_income, initial_state_rent, initial_state_dwelling_size, fraction_capital_destroyed, amenities, housing_limit, spline_estimate_RDP, spline_land_constraints, spline_land_backyard, spline_land_RDP, spline_land_informal, income_class_by_housing_type, precalculated_transport, spline_RDP, spline_agricultural_rent, spline_interest_rate, spline_population_income_distribution, spline_inflation, spline_income_distribution, spline_population, spline_income, spline_minimum_housing_supply, spline_fuel, income_2011)
Run simulations over several years according to scenarios.

### Dynamic functions

Created on Tue Nov  3 14:16:26 2020.

@author: Charlotte Liotta


### equilibrium.functions_dynamic.compute_average_income(spline_population_income_distribution, spline_income_distribution, param, t)
Compute average income from scenario data (includes RDP).


### equilibrium.functions_dynamic.evolution_housing_supply(housing_limit, param, option, t1, t0, housing_supply_1, housing_supply_0)
Yield dynamic housing supply with time inertia and depreciation.


### equilibrium.functions_dynamic.import_scenarios(income_2011, param, grid, path_scenarios, options)
Return linear regression splines for various scenarios.


### equilibrium.functions_dynamic.interpolate_interest_rate(spline_interest_rate, t)
Return an average interest rate over some period of interest.

### Compute intermediate outputs

Created on Wed Oct 28 16:01:05 2020.

@author: Charlotte Liotta


### equilibrium.sub.compute_outputs.compute_outputs(housing_type, utility, amenities, param, income_net_of_commuting_costs, fraction_capital_destroyed, grid, income_class_by_housing_type, options, housing_limit, agricultural_rent, interest_rate, coeff_land, minimum_housing_supply, construction_param, housing_in, param_pockets, param_backyards_pockets)
Compute equilibrium outputs from theoretical formulas.

### Define optimality conditions for solver

Created on Fri Oct 30 14:13:50 2020.

@author: Charlotte Liotta


### equilibrium.sub.functions_solver.compute_dwelling_size_formal(utility, amenities, param, income_net_of_commuting_costs, fraction_capital_destroyed)
Return optimal dwelling size per income group for formal housing.


### equilibrium.sub.functions_solver.compute_housing_supply_backyard(R, param, income_net_of_commuting_costs, fraction_capital_destroyed, grid, income_class_by_housing_type)
Compute backyard housing supply as a function of rents.


### equilibrium.sub.functions_solver.compute_housing_supply_formal(R, options, housing_limit, param, agricultural_rent, interest_rate, fraction_capital_destroyed, minimum_housing_supply, construction_param, housing_in, dwelling_size)
Calculate the formal housing supply as a function of rents.


### equilibrium.sub.functions_solver.implicit_qfunc(q, q_0, alpha)
Implicitely define optimal dwelling size.

## Calibration

### Main functions for calibration

Created on Mon May 23 16:26:37 2022.

@author: monni


### calibration.calib_main_func.estim_construct_func_param(options, param, data_sp, threshold_income_distribution, income_distribution, data_rdp, housing_types_sp, data_number_formal, data_income_group, selected_density, path_data, path_precalc_inp, path_folder)
Estimate coefficients of construction function (Cobb-Douglas).


### calibration.calib_main_func.estim_incomes_and_gravity(param, grid, list_lambda, households_per_income_class, average_income, income_distribution, spline_inflation, spline_fuel, spline_population_income_distribution, spline_income_distribution, path_data, path_precalc_inp, path_precalc_transp, options)
Estimate incomes per job center and group and commuting parameter.


### calibration.calib_main_func.estim_util_func_param(data_number_formal, data_income_group, housing_types_sp, data_sp, coeff_a, coeff_b, coeffKappa, interest_rate, incomeNetOfCommuting, selected_density, path_data, path_precalc_inp, options, param)
Calibrate utility function parameters.

### Calibrate income net of commuting costs and gravity parameter

Created on Fri Apr  1 16:20:51 2022.

@author: monni


### calibration.sub.compute_income.EstimateIncome(param, timeOutput, distanceOutput, monetaryCost, costTime, job_centers, average_income, income_distribution, list_lambda, options)
Solve for income per employment center for some values of lambda.


### calibration.sub.compute_income.compute_ODflows(householdSize, monetaryCost, costTime, incomeCentersFull, whichCenters, param_lambda)
Apply commuting formulas from working paper.


### calibration.sub.compute_income.funSolve(incomeCentersTemp, averageIncomeGroup, popCenters, popResidence, monetaryCost, costTime, param_lambda, householdSize, whichCenters, bracketsDistance, distanceOutput, options)
Compute error in employment allocation.


### calibration.sub.compute_income.import_transport_costs(grid, param, yearTraffic, households_per_income_class, spline_inflation, spline_fuel, spline_population_income_distribution, spline_income_distribution, path_precalc_inp, path_precalc_transp, dim, options)
Compute job center distribution, commuting and net income.

### Estimate utility function parameters by optimization

Created on Tue Oct 20 10:49:58 2020.

@author: Charlotte Liotta


### calibration.sub.estimate_parameters_by_optimization.EstimateParametersByOptimization(incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup, dataHouseholdDensity, selectedDensity, xData, yData, selectedSP, tableAmenities, variablesRegression, initRho, initBeta, initBasicQ, initUti2, initUti3, initUti4, options)
Automatically estimate parameters by maximizing log likelihood.

### Estimate utility function parameters by scanning

Created on Tue Oct 20 10:50:37 2020.

@author: Charlotte Liotta


### calibration.sub.estimate_parameters_by_scanning.EstimateParametersByScanning(incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup, dataHouseholdDensity, selectedDensity, xData, yData, selectedSP, tableAmenities, variablesRegression, initRho, listBeta, listBasicQ, initUti2, listUti3, listUti4, options)
Estimate parameters by maximizing log likelihood.

### Import exogenous amenities (for amenity index)

Created on Wed Apr  6 16:55:17 2022.

@author: monni


### calibration.sub.import_amenities.import_amenities(path_data, path_precalc_inp, dim)
Import relevant amenity data at SP level.

### Import employment data (for income and gravity)

Created on Mon Oct 19 12:22:55 2020.

@author: Charlotte Liotta


### calibration.sub.import_employment_data.import_employment_data(households_per_income_class, param, path_data)
Import number of jobs per selected employment center.

### Log-likelihood (for utility function parameters)

Created on Tue Oct 20 11:46:13 2020.

@author: Charlotte Liotta


### calibration.sub.loglikelihood.InterpolateRents(beta, basicQ, net_income, options)
Interpolate log(rents) as a function of log(beta) and log(q0).


### calibration.sub.loglikelihood.LogLikelihoodModel(X0, Uo2, net_income, groupLivingSpMatrix, dataDwellingSize, selectedDwellingSize, dataRent, selectedRents, selectedDensity, predictorsAmenitiesMatrix, tableRegression, variables_regression, CalculateDwellingSize, ComputeLogLikelihood, optionRegression, options)
Estimate the total likelihood of the model given the parameters.


### calibration.sub.loglikelihood.utilityFromRents(Ro, income, basic_q, beta)
Return utility / amenity index ratio.

## Outputs

### Process and display general values

Created on Mon Nov  2 11:32:48 2020.

@author: Charlotte Liotta


### outputs.export_outputs.export_households(initial_state_households, households_per_income_and_housing, legend1, legend2, path_plots, path_tables)
Bar plot for equilibrium output across housing and income groups.


### outputs.export_outputs.export_housing_types(housing_type_1, housing_type_2, legend1, legend2, path_plots, path_tables)
Bar plot for equilibrium output across housing types.


### outputs.export_outputs.export_map(value, grid, geo_grid, path_plots, export_name, title, path_tables, ubnd, lbnd=0, cmap='Reds')
Generate 2D heat maps of any spatial input.


### outputs.export_outputs.from_df_to_gdf(array, geo_grid)
Convert map array/series inputs into grid-level GeoDataFrames.


### outputs.export_outputs.import_employment_geodata(households_per_income_class, param, path_data)
Import number of jobs per selected employment center.


### outputs.export_outputs.plot_average_income(grid, average_income, path_plots, path_tables)
Plot average income across 1D-space.


### outputs.export_outputs.plot_housing_demand(grid, center, initial_state_dwelling_size, initial_state_households_housing_types, housing_types_sp, data_sp, path_plots, path_tables)
Plot average dwelling size in private formal across space.


### outputs.export_outputs.plot_housing_supply(grid, initial_state_housing_supply, path_plots, path_tables)
Line plot of avg housing supply per type and unit of available land.


### outputs.export_outputs.plot_housing_supply_noland(grid, housing_supply, path_plots, path_tables)
Line plot of total housing supply per type across 1D-space.


### outputs.export_outputs.plot_income_net_of_commuting_costs(grid, income_net_of_commuting_costs, path_plots, path_tables)
Plot avg income net of commuting costs across 1D-space.


### outputs.export_outputs.retrieve_name(var, depth)
Retrieve name of a variable.


### outputs.export_outputs.simul_housing_demand(grid, center, initial_state_dwelling_size, initial_state_households_housing_types, path_plots, path_tables)
Plot average dwelling size in private formal across space.


### outputs.export_outputs.simulation_density_housing_types(grid, initial_state_households_housing_types, path_plots, path_tables)
Line plot for number of households per housing type across 1D-space.


### outputs.export_outputs.simulation_density_income_groups(grid, initial_state_household_centers, path_plots, path_tables)
Line plot for number of households per income group across 1D-space.


### outputs.export_outputs.simulation_housing_price(grid, initial_state_rent, interest_rate, param, center, housing_types_sp, path_plots, path_tables, land_price)
Plot land price per housing type across space.


### outputs.export_outputs.validate_average_income(grid, overall_avg_income, data_avg_income, path_plots, path_tables)
Validate overall average income across 1D-space.


### outputs.export_outputs.validation_density(grid, initial_state_households_housing_types, housing_types, path_plots, path_tables)
Line plot for household density across space in 1D.


### outputs.export_outputs.validation_density_housing_and_income_groups(grid, initial_state_households, path_plots, path_tables)
Plot number of HHs per housing and income group across 1D-space.


### outputs.export_outputs.validation_density_housing_types(grid, initial_state_households_housing_types, housing_types, path_plots, path_tables)
Line plot for number of households per housing type across 1D-space.


### outputs.export_outputs.validation_density_income_groups(grid, initial_state_household_centers, income_distribution_grid, path_plots, path_tables)
Line plot for number of households per income group across 1D-space.


### outputs.export_outputs.validation_housing_price(grid, initial_state_rent, interest_rate, param, center, housing_types_sp, data_sp, path_plots, path_tables, land_price)
Plot land price per housing type across space.


### outputs.export_outputs.validation_housing_price_test(grid, initial_state_rent, initial_state_households_housing_types, interest_rate, param, center, housing_types_sp, data_sp, path_plots, path_tables, land_price)
Plot land price per housing type across space.

### Process values related to floods

Created on Fri Nov  6 17:00:06 2020.

@author: Charlotte Liotta


### outputs.flood_outputs.annualize_damages(array_init, type_flood, housing_type, options)
Annualize damages from floods.


### outputs.flood_outputs.compute_content_cost(initial_state_household_centers, initial_state_housing_supply, income_net_of_commuting_costs, param, fraction_capital_destroyed, initial_state_rent, initial_state_dwelling_size, interest_rate)
Compute value of damaged composite good.


### outputs.flood_outputs.compute_damages(floods, path_data, param, content_cost, nb_households_formal, nb_households_subsidized, nb_households_informal, nb_households_backyard, dwelling_size, formal_structure_cost, content_damages, structural_damages_type4b, structural_damages_type4a, structural_damages_type2, structural_damages_type3a, options, spline_inflation, year_temp, path_tables, flood_categ)
Summarize flood damages per housing and flood type.


### outputs.flood_outputs.compute_damages_2d(floods, path_data, param, content_cost, nb_households_formal, nb_households_subsidized, nb_households_informal, nb_households_backyard, dwelling_size, formal_structure_cost, content_damages, structural_damages_type4b, structural_damages_type4a, structural_damages_type2, structural_damages_type3a, options, spline_inflation, year_temp, path_tables, flood_categ)
Compute full flood damages per housing and flood type.


### outputs.flood_outputs.compute_formal_structure_cost_method2(initial_state_rent, param, interest_rate, coeff_land, initial_state_households_housing_types, construction_coeff)
Compute value of damaged formal structure capital.


### outputs.flood_outputs.compute_stats_per_housing_type(floods, path_floods, nb_households_formal, nb_households_subsidized, nb_households_informal, nb_households_backyard, path_tables, flood_categ, threshold=0.1)
Summarize flood-risk area and flood depth per housing and flood type.


### outputs.flood_outputs.compute_stats_per_income_group(floods, path_floods, nb_households_rich, nb_households_midrich, nb_households_midpoor, nb_households_poor, path_tables, flood_categ, threshold=0.1)
Summarize flood-risk area and flood depth per income and flood type.


### outputs.flood_outputs.create_flood_dict(flood_type, path_floods, path_tables, sim_nb_households_poor, sim_nb_households_midpoor, sim_nb_households_midrich, sim_nb_households_rich)
Create dictionary for household distribution in each flood maps.

### Display values related to floods

Created on Fri Nov  6 17:38:21 2020.

@author: Charlotte Liotta


### outputs.export_outputs_floods.plot_damages(damages1, damages2, path_plots, flood_categ, options)
Plot aggregate annualized damages per housing type.


### outputs.export_outputs_floods.plot_flood_severity_distrib(barWidth, transparency, dictio, flood_type, path_plots, ylim)
Plot distribution of flood severity across income groups for some RP.


### outputs.export_outputs_floods.round_nearest(x, a)
Round to nearest decimal number.


### outputs.export_outputs_floods.simul_damages(damages, path_plots, flood_categ, options)
Plot aggregate annualized damages per housing type.


### outputs.export_outputs_floods.simul_damages_time(list_damages, path_plots, path_tables, flood_categ, options)
Plot aggregate annualized damages per housing type.


### outputs.export_outputs_floods.validation_flood(stats1, stats2, legend1, legend2, type_flood, path_plots)
Bar plot flood depth and area across some RPs per housing types.


### outputs.export_outputs_floods.validation_flood_coastal(stats1, stats2, legend1, legend2, type_flood, path_plots)
Bar plot flood depth and area across some RPs per housing types.
