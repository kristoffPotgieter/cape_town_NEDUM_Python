---
date: '2022-09-23T09:22:05.479Z'
docname: technical_doc
images: {}
path: /technical-doc
title: Technical documentation
---

# Technical documentation

## Code walk-through

The project repository is composed of a `main` script and `plots` scripts, calling on several secondary scripts and user-defined packages. Those packages in turn include modules that are used at different steps of the code:


* `inputs`: contains the modules used to import default parameters and options, as well as pre-treated data


* `equilibrium`: contain the modules used to compute the static equilibrium allocation, as well as the dynamic simulations


* `outputs`: contain the modules used to visualize results and data.


* `calibration`: contains the modules used to re-run the calibration of parameters if necessary

As for the `main` script and `plots` scripts, they start with a preamble that imports (python and user-defined) packages to be used, and defines the file paths that integrate the repository to the local system of the user. They then go on calling the packages described above. Let us start with the `main` script.

## Inputs

### Parameters and options

In this part, the code calls on the `parameters_and_options.py` module that imports the default parameters and options, then sets the timeline for the simulations, and overwrites the default parameters and options according to the specific scenario one is interested in. The detail of key parameters and options is given in [Input tables]() (there are other technical parameters and options that should be of no interest for the end user) . Typically, one may set options allowing or not agents to anticipate floods, or new informal settlements to be built.

Note that, on top of the `import_options` and `import_param` functions, the code later calls on a `import_construction_parameters` function that makes use of imported data to define additional parameter variables:

The key ones are the `mini_lot_size` parameter and the `agricultural_rent` variable (defined through `compute_agricultural_rent` as a function of other parameters). Here is how to interpret the value of agricultural rent. We assume that the price of agricultural land is fixed (corresponds to landlords’ outside options / opportunity cost, here `agricultural_rent_2011` parameter) and that agricultural land is undeveloped. Since we assume that developers make zero profit in equilibrium due to pure and perfect competition, this gives us a relation to obtain the minimum rent developers would be willing to accept if this land were urbanized :

R_A = \\frac{(\\delta P_A)^a (\\rho + \\delta)^{1-a}}{\\kappa a^a (1-a)^{1-a}}In the `compute_agricultural_rent` function, this corresponds to:

Below this rent, it is therefore never profitable to build housing. Agricultural rent defines a floor on equilibrium rent values as well as an endogenous city edge.

To store the output in a dedicated folder, we create a name associated to the parameters and options used:

This name is a code associated to the options, parameters, and scenarios that we changed across our main simulation runs. They can safely be updated by the end user.

### Assumptions

Note that, in our model, we consider three endogenous housing markets (whose allocation is computed in equilibrium) - namely, formal private housing, informal backyards (erected in the backyard of subsidized housing dwelling units), and informal settlements (in predetermined locations)  - one exogenous housing market (whose allocation is directly taken from the data) - the RDP (Reconstruction and Development Programme) formal subsidized housing  - and four income groups.

By default, only agents from the poorest income group have access to RDP housing units. Without loss of generality, we assume that the price of such dwellings is zero, and that they are allocated randomly to the part of poorest agents they can host, the rest being rationed out of the formal subsidized housing market. Then, the two poorest income groups sort across formal private housing, informal backyards, and informal settlements; whereas the two richest only choose to live in the formal private housing units. Those assumptions are passed into the code through the `income_class_by_housing_type` parameter. We believe that this is a good approximation of reality. Also note that, although the two richest income groups are identical in terms of housing options, we distinguish between them to better account for income heterogeneity and spatial sorting along income lines in our simulations, while keeping the model sufficiently simple to be solved numerically.

### Data

Then, the code calls on the `data.py` module. This allows to import basic geographic data, macro data, households and income data, land use projections, flood data, scenarios for time-dependent variables, and transport data. The databases used are presented in [Data bases](). A more detailed view on the specific data sets called in the code, and their position in the folder architecture, is given in [Data sets]() .

It should be noted that, by default, housing type data (used for validation) has already been converted from the SAL (Small Area Level) dimension to the grid dimension (500x500m pixels) with the `import_sal_data` and `gen_small_areas_to_grid` functions, and that income net of commuting costs for each income group and each grid cell (from transport data) has already been computed from the calibrated incomes for each income group and each job center with the `import_transport_data` function. If one wants to run the process again, one needs to overwrite the associated default options (please note that this may take some time to run). The import of flood data (through the `import_full_floods_data`) is also a little time-consuming when agents are set to anticipate floods.

A bit counter-intuitively, the scenario for interest rate does not only serve to define future values of the variable (more on that in Subsequent periods), but also its value at the initial state (as the raw data contains both past and future values). As part of the `import_macro_data` function, the `interest_rate` variable is defined as the average of past values over the past three years, through the `interpolate_interest_rate` function defined in the `functions_dynamic.py` module (included in the `equilibrium` package). This allows to smooth outcomes around a supposed equilibrium value:

```python
    # Interest rate
    #  Import interest_rate history + scenario until 2040
    scenario_interest_rate = pd.read_csv(
        path_scenarios + 'Scenario_interest_rate_1.csv', sep=';')
    #  Fit linear regression spline centered around baseline year
    spline_interest_rate = interp1d(
        scenario_interest_rate.Year_interest_rate[
            ~np.isnan(scenario_interest_rate.real_interest_rate)]
        - param["baseline_year"],
        scenario_interest_rate.real_interest_rate[
            ~np.isnan(scenario_interest_rate.real_interest_rate)],
        'linear'
        )
    #  Get interest rate as the mean (in %) over x last years
    interest_rate = eqdyn.interpolate_interest_rate(spline_interest_rate, 0)

    # Aggregate population: this come from SAL data and serves as a benchmark
```

```python
    nb_years = 3
    interest_rate_n_years = spline_interest_rate(np.arange(t - nb_years, t))
    interest_rate_n_years[interest_rate_n_years < 0] = np.nan
    return np.nanmean(interest_rate_n_years)/100


```

The imported files can be modified directly in the repository, to account for changes in scenarios for instance, as long as the format used remains the same (see [API reference]() for more details on each function requirements). Before going to the next step of the code, we would like to give more details on the imports of land use, flood, and transport data, which we think are not as transparent as the rest of the imports.

#### Land use data

The `import_land_use` and `import_coeff_land` functions help define exogenous land availability $L^h(x)$ for each housing type $h$ and each grid cell $x$. Indeed, we assume that different kinds of housing do not get built in the same places to account for insecure property rights (in the case of informal settlements vs. formal private housing) and housing specificities (in the case of non-market formal subsidized housing, and informal backyards located in the same preccints) . Furthermore, $L$ varies with $x$ to account for both natural, regulatory constraints, infrastructure and other non-residential uses.

As $L^h(x)$ is going to vary across time periods, the first part of the `import_land_use` function imports lacking data estimates for historical and projected land uses. It then defines linear regression splines over time for a set of variables, the key ones being the share of pixel area available for formal subsidized housing, informal backyards, informal settlements, and unconstrained development. Note that, in our benchmark (allowing for informal settlement expansion), land availability for informal settlements is defined over time by the intersection between the timing map below and the high and very high probability areas (also defined below).

The `import_coeff_land` function then takes those outputs as arguments and reweight them by a housing-specific maximum land use parameter. This parameter allows to reduce the development potential of each area to its housing component (accounting for roads, for instance). The share of pixel area available for formal private housing (in a given period) is simply defined as the share of pixel area available for unconstrained development, minus the shares dedicated to the other housing types, times its own maximum land use parameter:

```python
    coeff_land_private[coeff_land_private < 0] = 0
    coeff_land_backyard = (spline_land_backyard(t)
                           * param["max_land_use_backyard"])
    coeff_land_RDP = spline_land_RDP(t)  # * param["max_land_use"]
    coeff_land_settlement = (spline_land_informal(t)
                             * param["max_land_use_settlement"])
    coeff_land = np.array([coeff_land_private, coeff_land_backyard,
                           coeff_land_settlement, coeff_land_RDP])

    return coeff_land


```

The outputs are stored in a `coeff_land` matrix that yields the values of $L^h(x)$ for each time period when multiplied by the area of a pixel. Results are shown in the figure below.

#### Flood data

Flood data is processed through the `import_init_floods_data`, `compute_fraction_capital_destroyed`, and `import_full_floods_data` functions.

The `import_init_floods_data` function imports the pre-processed flood maps from FATHOM (fluvial and pluvial), and DELTARES (coastal). Those maps yield for each grid cell an estimate of the pixel share that is exposed to a flood of some maximum depth level, reached in a given year with a probability equal to the inverse of their return period. For instance, a flood map corresponding to a return period of 100 years considers events that have a 1/100 chance of ocurring in a given year.

Note that the considered return periods are not the same for FATHOM and DELTARES data, and that different flood maps are available depending on the digital elevation model (DEM) considered, whether we account for sea-level rise, and whether we account for defensive infrastructure with respect to fluvial floods .

The function then imports depth-damage functions from the existing literature. Those functions, used in the insurance market, link a level of maximum flood depth to a fraction of capital destroyed, depending on the materials affected by floods . More specifically, we use estimates from Englhardt *et al.* [[2019](#id16)] and Hallegatte *et al.* [[2013](#id24)] for damages to housing structures (depending on housing type), and from de Villiers *et al.* [[2007](#id17)] for damages to housing contents.

On this basis, the `compute_fraction_capital_destroyed` function integrates flood damages over the full range of return periods possible, for each flood type and damage function . This yields an annualized fraction of capital destroyed, corresponding to its expected value in a given year .

Finally, the `import_full_floods_data` function uses those outputs to define the depreciation term $\rho_{h}^{d}(x)$ that is specific to housing type $h$, damage type (structures or contents) $d$, and location $x$ (stored in the `fraction_capital_destroyed` matrix), by taking the maximum of fractions of capital destroyed for all flood types considered. When multiplied by some capital value, this term yields the expected economic cost from flood risks that is considered by anticipating agents when solving for equilibrium. It is also equal to the risk premium in the case of a perfect insurance market (leading to full insurance with actuarially fair prices) .

#### Transport data

Transport data is processed through the `import_transport_data` function. It imports monetary and time transport costs and pre-calibrated incomes (`income_centers_init` parameter) per income group and job center (more on that in Calibration), for a given dimension (grid-cell level or small-place level) and a given simulation year (zero at initial state):

```python
    FATHOM_100yr['pop_flood_prone'] = (
        FATHOM_100yr.prop_flood_prone
        * (housing_types.informal_grid
           + housing_types.formal_grid
           + housing_types.backyard_formal_grid
           + housing_types.backyard_informal_grid)
        )
```

```python
                          path_precalc_inp, path_precalc_transp, dim, options):
    """Compute job center distribution, commuting and net income."""
    # Import (monetary and time) transport costs
    (timeOutput, distanceOutput, monetaryCost, costTime
     ) = calcmp.import_transport_costs(
         grid, param, yearTraffic, households_per_income_class,
         spline_inflation, spline_fuel, spline_population_income_distribution,
         spline_income_distribution, path_precalc_inp, path_precalc_transp,
         dim, options)

    param_lambda = param["lambda"].squeeze()

    incomeNetOfCommuting = np.zeros(
        (param["nb_of_income_classes"],
         timeOutput[:, :, 0].shape[1])
        )
    averageIncome = np.zeros(
```

From there, it computes several outputs, of which the key variable is the expected income net of commuting costs $\tilde{y}_i(x)$, earned by a household of income group $i$ choosing to live in $x$ matrix). It is obtained by recursively solving for the optimal transport mode choice and job center choice of households characterized by $i$ and $x$ (see [Math appendix]() for more details).

## Calibration

This whole part is optional. The reason is that all necessary parameters have been pre-calibrated, and it is only useful to run this part again if the underlying data used for calibration has changed. Also note that it takes time to run. If needed, it has to be run before solving the equilibrium, which is why it was included at this stage of the script. However, it relies on the estimation of partial relations derived from the general equilibrium itself . For better understanding, we therefore advise you to get back to this section after reading the Equilibrium section.

The preamble imports more useful data, under some technical options. Note that, whereas the same data might be used for calibration of parameters and validation of results, it is never an input of the model per se. Typically, this is data at the SAL (Small Area Level) or SP (Small Place) level, less granular than our counterfactual grid-level estimates.

### Construction function parameters

Then, we calibrate construction function parameters using the `estim_construct_func_param` function from the `calib_main_func.py` module:

From the equilibrium formulas for housing supply and number of households, we obtain the following relation:

\\frac{N^{FP}(x)Q_{FP}(x)}{L_{FP}(x)} = \\kappa^{\\frac{1}{a}} (\\frac{(1-a)R_{FP}(x)}{\\rho + \\textcolor{green}{\\rho_{FP}^{struct}(x)} + \\delta})^{\\frac{1-a}{a}}Log-linearizing it (and using the relation between price and rent that we already used to define agricultural rent) allows us to identify the following linear regression, which we estimate with data at the SP level $s$ (instead of the grid-cell level $x$, due to data availability constraints):

log(N_s^{FP}) = \\gamma_1 + \\gamma_2 log(P_s) + \\gamma_3 log(Q_s) + \\gamma_4 log(L_s^{FP}) + \\epsilon_s
* $\gamma_1 = log(\kappa \frac{1-a}{a}^{1-a})$


* $\gamma_2 = 1-a$


* $\gamma_3 = -1$


* $\gamma_4 = 1$

We therefore have $a = 1 - _gamma_2$ and $\kappa = \frac{a}{1-a}^{1-a} exp(\gamma_1)$.

In the code, this translates into:

```python
    y = np.log(data_number_formal[selected_density])
    # Note that we use data_sp["unconstrained_area"] (which is accurate data at
    # SP level) rather than coeff_land (which is an estimate at grid level)
    X = np.transpose(
        np.array([np.ones(len(data_sp["price"][selected_density])),
                  np.log(data_sp["price"][selected_density]),
                  np.log(data_sp["dwelling_size"][selected_density]),
                  np.log(param["max_land_use"]
                         * data_sp["unconstrained_area"][selected_density])])
        )
    # NB: Our data set for dwelling sizes only provides the average (not
    # median) dwelling size at the Sub-Place level, aggregating formal and
    # informal housing

    # model_construction = LinearRegression().fit(X, y)

    modelSpecification = sm.OLS(y, X, missing='drop')
    model_construction = modelSpecification.fit()
    print(model_construction.summary())
    parametersConstruction = model_construction.params

    # We export outputs of the model
    # coeff_b = model_construction.coef_[0]
    coeff_b = parametersConstruction["x1"]
    coeff_a = 1 - coeff_b
    # Comes from zero profit condition combined with footnote 16 from
    # optimization (typo in original paper)
    if options["correct_kappa"] == 1:
        # coeffKappa = ((1 / (coeff_b / coeff_a) ** coeff_b)
        #               * np.exp(model_construction.intercept_))
        coeffKappa = ((1 / (coeff_b / coeff_a) ** coeff_b)
                      * np.exp(parametersConstruction["const"]))
    elif options["correct_kappa"] == 0:
        # coeffKappa = ((1 / (coeff_b) ** coeff_b)
        #               * np.exp(model_construction.intercept_))
        coeffKappa = ((1 / (coeff_b) ** coeff_b)
                      * np.exp(parametersConstruction["const"]))

```

Note that we run our regression on a restricted sample defined by the `selected_density` variable:

As the relation only holds in the formal private sector, we exclude SPs in the bottom quintile of property prices and for which more than 5% of households are reported to live in informal housing. We also exclude rural SPs (i.e., those that are large, with a small share that can effectively be urbanized), the poorest income group (which in effect is crowded out of the formal private sector), as well as Mitchell’s Plain neighbourhood (whose housing market is very specific), and far-away land (where SPs are too large to produce any representative values).

### Estimations of incomes and gravity parameter

We update the parameter vector with the newly calculated values and go on with the calibration of incomes $y_{ic}$ and the gravity parameter $\lambda$ (see [Math appendix]() for more details and definitions).

In practice, we scan a set of predefined values for the parameter $\lambda$ over which we determine the value of $y_{ic}$. We then aggregate the total distribution of residence-workplace distances, and compare it with the data aggregated from Cape Town’s Transport Survey 2013. We select the value of $\lambda$, and the associated $y_{ic}$, that minimizes the total distance between the calculated distribution of commuting distances and aggregates from the data.

In the `main.py` script, this translates into:

The `estim_incomes_and_gravity` function (from the `calib_main_func.py` module) proceeds as follows. It first imports the number of workers per income group in each selected job center (at the TAZ level) through the `import_employment_data` function (from the module of the same name).

#### Transport costs

It then imports transport costs through the `import_transport_costs` function (from the `compute_income.py` module).

The first part of this function imports the data on transportation times and distances, and estimates of public transportation fixed (over one month) and variable costs, based upon linear regressions using data from Roux [[2013](#id26)] (table 4.15). It also imports fuel prices for the variable component of private transportation costs, and 400 rands as the monthly depreciation cost of a vehicle for its fixed component.

In a second part, it computes the total yearly monetary cost of commuting for one household over round trips:

```python
    # Length (in km) using each mode
    if options["correct_round_trip"] == 1:
        multiplierPrice = np.empty((timeOutput.shape))
        multiplierPrice[:] = np.nan
        multiplierPrice[:, :, 0] = np.zeros((timeOutput[:, :, 0].shape))
        multiplierPrice[:, :, 1] = transport_times["distanceCar"] * 2
        multiplierPrice[:, :, 2] = transport_times["distanceCar"] * 2
        multiplierPrice[:, :, 3] = transport_times["distanceCar"] * 2
        multiplierPrice[:, :, 4] = transport_times["distanceCar"] * 2
    elif options["correct_round_trip"] == 0:
        multiplierPrice = np.empty((timeOutput.shape))
        multiplierPrice[:] = np.nan
        multiplierPrice[:, :, 0] = np.zeros((timeOutput[:, :, 0].shape))
        multiplierPrice[:, :, 1] = transport_times["distanceCar"]
        multiplierPrice[:, :, 2] = transport_times["distanceCar"]
        multiplierPrice[:, :, 3] = transport_times["distanceCar"]
        multiplierPrice[:, :, 4] = transport_times["distanceCar"]

    # Multiplying by 235 (nb of working days per year)
    pricePerKM = np.empty(5)
    pricePerKM[:] = np.nan
    pricePerKM[0] = np.zeros(1)
    pricePerKM[1] = priceTrainPerKMMonth*numberDaysPerYear
    pricePerKM[2] = priceFuelPerKMMonth*numberDaysPerYear
    pricePerKM[3] = priceTaxiPerKMMonth*numberDaysPerYear
    pricePerKM[4] = priceBusPerKMMonth*numberDaysPerYear
```

```python
    # Monetary price per year (for each employment center)
    monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
    # trans_monetaryCost = np.zeros((185, timeOutput.shape[1], 5))
    for index2 in range(0, 5):
        monetaryCost[:, :, index2] = (pricePerKM[index2]
                                      * multiplierPrice[:, :, index2])

    #  Train (monthly fare)
    monetaryCost[:, :, 1] = monetaryCost[:, :, 1] + priceTrainFixedMonth * 12
    #  Private car
    monetaryCost[:, :, 2] = (monetaryCost[:, :, 2] + priceFixedVehiculeMonth
                             * 12)
    #  Minibus/taxi
    monetaryCost[:, :, 3] = monetaryCost[:, :, 3] + priceTaxiFixedMonth * 12
    #  Bus
    monetaryCost[:, :, 4] = monetaryCost[:, :, 4] + priceBusFixedMonth * 12
```

In a third and last part, it computes the time opportunity cost parameter (i.e., the fraction of working time spent commuting). Here, the values are given in minutes:

```python
    costTime = (timeOutput * param["time_cost"]
                / (60 * numberHourWorkedPerDay))
```

#### Income calculation

The `estim_incomes_and_gravity` function then goes on with the calibration of incomes per se. To do so, it calls on the `EstimateIncome` function (also from the `compute_income.py` module):

```python
     ) = calcmp.EstimateIncome(
        param, timeOutput, distanceOutput[:, :, 0], monetaryCost, costTime,
        job_centers, average_income, income_distribution, list_lambda, options)

```

Essentially, this function relies on the following equation (that can be solved from [Math appendix]()):

W_{ic} = \\chi_i \\sum_{s} \\pi_{c|is} N_{is}
* $W_{ic}$ is the number of households of income group $i$ who work in job center $c$.

…

Finally

## Equilibrium

This part of the main script simply calls on two functions that return the key outputs of the model: `compute_equilibrium` solves the static equilibrium for baseline year, and `run_simulation` solves its dynamic version for all pre-defined subsequent years.

### Initial state

Let us first dig into the `compute_equilibrium` function. Our main input is the total population per income group in the city at baseline year. Since we took the non-employed (earning no income over the year) out to define our four income groups, we need to reweight it to account for the overall population:

```python
    # General reweighting using SAL data (no formal backyards)
    if options["unempl_reweight"] == 0:
        ratio = population / sum(households_per_income_class)
        households_per_income_class = households_per_income_class * ratio
```

Results are given in the table below.

Then, considering that all formal subsidized housing belongs to the poorest income group, we substract the corresponding number of households from this class to keep only the ones whose allocation in the housing market is going to be determined endogenously:

```python
        households_per_income_class[0] - total_RDP, 0)

```

Finally, we shorten the grid to consider only habitable pixels according to land availability and expected income net of commuting costs to alleviate numeric computations and initialize a few key variables before starting the optimization per se:

```python
    #  and a positive maximum income net of commuting costs across classes
    #  NB: we will have no output in other pixels (numeric simplification)
    selected_pixels = (
        (np.sum(coeff_land, 0) > 0.01).squeeze()
        & (np.nanmax(income_net_of_commuting_costs, 0) > 0)
        )
    coeff_land_full = copy.deepcopy(coeff_land)
```

#### Solving for equilibrium

Note that, among those variables, we define arbitrary values for utility levels across income groups:

```python
    #  We take arbitrary utility levels, not too far from what we would expect,
    #  to make computation quicker
    utility[0, :] = np.array([1501, 4819, 16947, 79809])
    index_iteration = 0
```

This relates to the way this class of models is solved: as a closed-city equilibrium model, **NEDUM-2D** takes total population (across income groups) as exogenous, and utility levels (across income groups) as endogenous .

It is then solved iteratively in four steps:


* We first derive housing demand for each housing type


* We deduce rents for each housing type


* We then compute housing supply for each housing type


* From there, we obtain population in all locations for all housing types.

We update initial utility levels depending on the gap between the objective and simulated population, and re-iterate the process until we reach a satisfying error level , according to the values of the parameters `precision` and `max_iter`. Of course, the closer the initial guess is to the final values, the quicker the algorithm converges. We will see below how each of the intermediate steps is computed.

A last word on the choice of a closed vs. open-city model: within a static framework, it is generally considered that closed-city models are a better representation of short-term outcomes and open-city models of long-term outcomes, as population takes time to adjust through migration. Here, we rely on scenarios from the CoCT (informed by more macro parameters than open-city models usually are) to adjust total population across time periods. Sticking to the closed-city model in those circumstances allows us to make welfare assessments based on utility changes without renouncing to the possibility that migrations occur.

#### Functional assumptions

Then, the `compute_equilibrium` function calls on the `compute_outputs` function for each endogenous housing type, which in turn calls on functions defined as part of the `functions_solver.py` module. This module applies formulas derived from the optimization process described in Pfeiffer *et al.* [[2019](#id6)].

Let us just recall the main assumptions here (refer to the paper for a discussion on those assumptions ). The parameters highlighted in green are specific to a model with agents that perfectly anticipate flood risks.

One the one hand, households maximize Stone-Geary preferences:

U(z,q,x,h) = z^\\alpha (q-q_0)^{1-\\alpha}A(x)B^h(x)
* $z$ is the quantity of composite good consumed (in monetary terms) and $q$ is the quantity of housing consumed (in m²). They are the choice variables .


* $x$ is the location where the household lives and $h$ is the housing type considered. They are the state variables.


* $\alpha$ is the composite good elasticity (`alpha` parameter), $1 - \alpha$ is the surplus housing elasticity, and $q_0$ (`q0` parameter) is the basic need in housing.


* $A(x)$ is a (centered around one) location-specific amenity index (`precalculated_amenities` parameter) and $B^h(x)$ is a (positive) location-specific disamenity index equal to one in formal sectors and smaller than one in informal sectors (`pockets` and `backyard_pockets` parameters), to account for the negative externalities associated with living in such housing (such as eviction probability, etc.) . They are (calibrated) parameters.

They are facing the following the following budget constraint (optimization under constraint):

\\tilde{y}_i(x) + \\mathbb{1}_{h=FS} \\mu(x)YR_{IB}(x) = (1 + \\textcolor{green}{\\gamma \\rho_{h}^{contents}(x)})z + q_{h}R_{h}(x) + \\mathbb{1}_{h \\neq FP} (\\rho + \\textcolor{green}{\\rho_{h}^{struct}(x)} + \\mathbb{1}_{h \\neq FS} \\delta)v_{h}
* $\tilde{y}_i(x)$ is the expected income net of commuting costs for a household of income group $i$ living in location $x$.


* $\mathbb{1}_{h=FS}$ is an indicator variable for living in the formal subsidized housing sector. Indeed, such households have the possibility to rent out an endogenous fraction $\mu(x)$ of their backyard of fixed size $Y$ (`backyard_size` parameter) at the endogenous housing rent $R_{IB}(x)$.


* $\textcolor{green}{\gamma}$ is the fraction of composite good exposed to floods (`fraction_z_dwellings` parameter) and $\textcolor{green}{\rho_{h}^{contents}(x)}$ is the fraction of contents capital destroyed (that is location and housing-type specific): households need to pay for damages to their belongings.


* $q_{h}$ is the (housing-type specific) amount of housing consumed, at the endogenous annual rent $R_{h}(x)$.


* $\mathbb{1}_{h \neq FP}$ is an indicator variable for living in a sector different from the formal private one. Indeed, such households (assimilated to owner-occupiers, more on that below) need to pay for the maintenance of their housing of capital value $v_{h}$ (`subsidized_structure_value` and `informal_structure_value` parameters) that depreciate at rate $\rho + \textcolor{green}{\rho_{h}^{struct}(x)}$ (`depreciation_rate` parameter + `fraction_capital_destroyed` matrix).


* $\mathbb{1}_{h \neq FS}$ is an indicator variable for living in a sector different from the formal subidized one. Indeed, among the set of households described above, those who live in the informal sector (either backyards or settlements) also need to pay for the construction of their “shack” of capital value $v_{h}$. To do so, they pay a fraction of capital value $\delta$ (`interest_rate` variable) of this price each year, which corresponds to the interest paid on an infinite debt (which is a good-enough approximation in a static setting).

On the other hand, formal private developers have a Cobb-Douglas housing production function expressed as:

s_{FP}(k) = \\kappa k^{1-a}
* $s_{FP}$ is the housing supply per unit of available land (in m² per km²).


* $\kappa$ is a (calibrated) scale factor (`coeff_A` parameter).


* $k = \frac{K}{L_{FP}}$ is the (endogenous) amount of capital per unit of available land (in monetary terms).


* $a$ is the (calibrated) land elasticity of housing production (`coeff_a` parameter).

They therefore maximize a profit function (per unit of available land) defined as:

\\Pi(x,k) = R_{FP}(x)s_{FP}(k) - (\\rho + \\textcolor{green}{\\rho_{FP}^{struct}(x)} + \\delta)k - \\delta P(x)
* $R^{FP}(x)$ is the (endogenous) market rent for formal private housing.


* $P(x)$ is the (endogenous) price of land.


* $k$ is the choice variable and $x$ is the state variable.

Underlying those functional forms are structural assumptions about the maximization objective of agents in each housing submarket:


* In the formal private sector, developers buy land from absentee landlords and buy capital to build housing on this land . They then rent out the housing at the equilibrium rent over an infinite horizon . They therefore internalize the costs associated with capital depreciation (both general and from structural flood damages) and interest rate (at which future flows of money are discounted). Households just have to pay for damages done to the content of their home .


* In the formal subsidized sector, (poor) households rent housing for free from the state (same as buying). They only pay for overall capital depreciation (general and from structural and content damages), and may rent out a fraction of their backyard.


* In the informal backyard sector, household rent a fraction of backyard (not housing) owned by formal subsidized housing residents and are responsible for building their own “shack” (owner-occupiers). Then, they pay for overall capital depreciation (general and from structural and content damages) and interest over construction costs too.


* In the informal settlement sector, households rent land from absentee landlords (not housing) and are responsible for building their own “shack” (owner-occupiers). Then, they pay for overall capital depreciation (general and from structural and content damages) and interest over construction costs too.

Note that both optimization programmes are concave, hence they can be maximized using first-order optimality conditions (setting the partial derivatives to zero).

#### Equilibrium dwelling size

As described in the Solving for equilibrium subsection, the `compute_outputs` functions starts by computing housing demand / dwelling size (in m²) for each housing type through the `compute_dwelling_size_formal` function. Optimization over the households’ programme described before implicitly defines this quantity in the formal private sector:

u = (\\frac{\\alpha \\tilde{y}_i(x)}{1 + \\textcolor{green}{\\gamma \\rho_{FP}^{contents}(x)}})^\\alpha \\frac{Q^\*-q_0}{(Q^\* - \\alpha q_0)^\\alpha} A(x)
* $u$ is the fixed utility level


* $Q*$ is the equilibrium dwelling size

By reorganizing the above equation, we obtain the following left and right sides in the code:

```python
    # According to WP, corresponds to [(Q*-q_0)/(Q*-alpha x q_0)^(alpha)] x B
    # (draft, p.11), see theoretical expression in implicit_qfunc()
    left_side = (
        (np.array(utility)[:, None] / np.array(amenities)[None, :])
        * ((1 + (param["fraction_z_dwellings"]
                 * np.array(fraction_capital_destroyed.contents_formal)[
                     None, :])) ** (param["alpha"]))
        / ((param["alpha"] * income_temp) ** param["alpha"])
        )
```

```python
    # Note that with above x definition, q-alpha*q_0 can be negative

    # Note that numpy returns null when trying to get the fractional power of a
    # negative number (which is fine, because we are not interested in such
    # values), hence we ignore the error
    np.seterr(divide='ignore', invalid='ignore')
    result = (
        (q - q_0)
        / ((q - (alpha * q_0)) ** alpha)
        )

    return result


```

The `compute_dwelling_size_formal` function then recovers the value of $Q*$ (depending on income group) through a linear interpolation, before constraining it to be bigger than the parametrized minimum dwelling size in this sector (`mini_lot_size` parameter):

```python
    # We define dwelling size as q corresponding to true values of
```

Back to the `compute_outputs` function, the dwelling size in the informal backyards and informal settlements sectors is exogenously set as being the standard parametrized size of a “shack” (`shack_size` parameter).

#### Equilibrium rent

Plugging this back into the first-order optimality condition for formal private housing households, and just inverting the households’ programme for informal backyards and settlements, we obtain the following bid rents in each sector, depending on income group:

\\Psi_i^{FP}(x,u) = \\frac{(1-\\alpha)\\tilde{y}_i(x)}{Q_{FP}(x,i,u) - \\alpha q_0}
* $Q_{FP}(x,i,u) = max(q_{min},Q^*)$

\\Psi_i^{IB}(x,u) = \\frac{1}{q_I}[\\tilde{y}_i(x) - (\\rho + \\delta + \\textcolor{green}{\\rho_{IB}^{struct}})v_I - (1 + \\textcolor{green}{\\gamma \\rho_{IB}^{contents}})(\\frac{u}{(q_I-q_0)^{1-\\alpha}A(x)B_{IB}(x)})^\\frac{1}{\\alpha}]\\Psi_i^{IS}(x,u) = \\frac{1}{q_I}[\\tilde{y}_i(x) - (\\rho + \\delta + \\textcolor{green}{\\rho_{IS}^{struct}})v_I - (1 + \\textcolor{green}{\\gamma \\rho_{IS}^{contents}})(\\frac{u}{(q_I-q_0)^{1-\\alpha}A(x)B_{IS}(x)})^\\frac{1}{\\alpha}]In the `compute_outputs` function, this corresponds to:

```python
        R_mat = (param["beta"] * (income_net_of_commuting_costs)
                 / (dwelling_size - (param["alpha"] * param["q0"])))
```

```python
        elif options["actual_backyards"] == 0:
            R_mat = (
                (1 / param["shack_size"])
                * (income_net_of_commuting_costs
                    - ((1 + np.array(
                        fraction_capital_destroyed.contents_backyard)[None, :]
                        * param["fraction_z_dwellings"])
                        * ((utility[:, None]
                            / (amenities[None, :]
                               * param_backyards_pockets[None, :]
                               * ((dwelling_size - param["q0"])
                                  ** param["beta"])))
                           ** (1 / param["alpha"])))
                    - (param["informal_structure_value"]
                       * (interest_rate + param["depreciation_rate"]))
                    - (np.array(
                        fraction_capital_destroyed.structure_informal_backyards
                        )[None, :] * param["informal_structure_value"]))
                )
```

```python
        R_mat = (
            (1 / param["shack_size"])
            * (income_net_of_commuting_costs
                - ((1 + np.array(fraction_capital_destroyed.contents_informal)[
                    None, :] * param["fraction_z_dwellings"])
                    * ((utility[:, None] / (amenities[None, :]
                                            * param_pockets[None, :]
                                            * ((dwelling_size - param["q0"])
                                               ** param["beta"])))
                       ** (1 / param["alpha"])))
                - (param["informal_structure_value"]
                   * (interest_rate + param["depreciation_rate"]))
                - (np.array(
                    fraction_capital_destroyed.structure_informal_settlements
                    )[None, :] * param["informal_structure_value"]))
            )
```

Bid rents $\psi^i_h(x,u)$ correspond to the maximum amount households of type $i$ are willing to pay for a unit (1 m²) of housing of type $h$ in a certain location $x$ for a given utility level $u$ (over one year). Assuming that households bid their true valuation and that there are no strategic interactions, housing / land  is allocated to the highest bidder. This is why we retain the bid rents from the highest bidding income groups, and the associated dwelling sizes, as the equilibrium output values for rents and dwelling sizes. In the code, this corresponds to:

```python
    # We select highest bidder (income group) in each location
    proba = (R_mat == np.nanmax(R_mat, 0))
    # We correct the matrix if binding budget constraint
    # (and other precautions)
    limit = ((income_net_of_commuting_costs > 0)
             & (proba > 0)
             & (~np.isnan(income_net_of_commuting_costs))
             & (R_mat > 0))
    proba = proba * limit

    # Yields directly the selected income group for each location
    which_group = np.nanargmax(R_mat, 0)

    # Then we recover rent and dwelling size associated with the selected
    # income group in each location
    R = np.empty(len(which_group))
    R[:] = np.nan
    dwelling_size_temp = np.empty(len(which_group))
    dwelling_size_temp[:] = np.nan
    for i in range(0, len(which_group)):
        R[i] = R_mat[int(which_group[i]), i]
        dwelling_size_temp[i] = dwelling_size[int(which_group[i]), i]

    dwelling_size = dwelling_size_temp
```

#### Equilibrium housing supply

Then, it goes on calling the `compute_housing_supply_formal` and `compute_housing_supply_backyard` functions.

In the formal private sector, profit maximization of developers with respect to capital yields:

s_{FP}(x) = \\kappa^{\\frac{1}{a}} (\\frac{(1-a)R_{FP}(x)}{\\rho + \\textcolor{green}{\\rho_{FP}^{struct}(x)} + \\delta})^{\\frac{1-a}{a}}In the code, this corresponds to:

```python
            * (construction_param ** (1/param["coeff_a"]))
            * ((param["coeff_b"]
                / (interest_rate + param["depreciation_rate"]
                   + capital_destroyed))
               ** (param["coeff_b"]/param["coeff_a"]))
            * ((R) ** (param["coeff_b"]/param["coeff_a"]))
            )

        # Outside the agricultural rent, no housing (accounting for a tax)
```

In the informal backyard sector, utility maximization of households living in formal subsidized housing with respect to fraction of backyard rented out yields:

\\mu(x) = \\alpha \\frac{q_{FS}-q_0}{Y} - (1-\\alpha) \\frac{\\tilde{y}_1(x) - (\\rho + \\textcolor{green}{\\rho_{FS}^{struct}(x)}) \\textcolor{green}{h_{FS}}}{YR_{IB}(x)}In the code, this corresponds to:

```python
        - (param["beta"]
           * (income_net_of_commuting_costs[0, :]
              - (capital_destroyed * param["subsidized_structure_value"]))
           / (param["backyard_size"] * R))
    )

    # NB: we convert units to m²
    housing_supply[R == 0] = 0
    housing_supply = np.minimum(housing_supply, 1)
```

Back to the `compute_outputs` function, the housing supply per unit of available land for informal settlements is just set as 1 km²/km². Indeed, as absentee landlords do not have outside use for their land, they face no opportunity cost when renting it out, and therefore rent all of it. We also recall that the informal backyard and settlement dwellings are not capital-intensive, to the extent that they have a fixed size and cover only one floor. The housing supply is therefore equal to the land supply. Again, this is a simplification, but we believe that this is a good enough approximation of reality.

#### Equilibrium number of households

At the end of the `compute_outputs` function, we just divide the housing supply per unit of available land by the dwelling size, and multiply it by the amount of available land to obtain the number of households by housing type in each grid cell. Then, we associate people in each selected pixel to the highest bidding income group (here denoted $i$):

N_i^h(x) = \\frac{s_h(x)L_h(x)}{Q_h(x,i,u)}From there, we are able to recover the total number of households in each income group for a given housing type:

```python
    # Yields population density in each selected pixel
    people_init = housing_supply / dwelling_size * (np.nansum(limit, 0) > 0)
    people_init[np.isnan(people_init)] = 0
    # Yields number of people per pixel, as 0.25 is the area of a pixel
    # (0.5*0.5 km) and coeff_land reduces it to inhabitable area
    people_init_land = people_init * coeff_land * 0.25

    # We associate people in each selected pixel to the highest bidding income
    # group
    people_center = np.array(people_init_land)[None, :] * proba
    people_center[np.isnan(people_center)] = 0
    # Then we sum across pixels and get the number of people in each income
    # group for given housing type
    job_simul = np.nansum(people_center, 1)
```

Also note that we ensure that the housing rent in the formal private sector is bigger than the agricultural rent:

```python
    if housing_type == 'formal':
        R = np.maximum(R, agricultural_rent)
```

#### Iteration and results

Back to the body of the `compute_equilibrium` function, we sum the outputs of the `compute_outputs` function to get the total number of households in each income group. Then, we define an error metric `error_max_abs` comparing this result with values from the data, and an incremental value `diff_utility` (depending on a predetermined convergence factor):

```python

    #  diff_utility will be used to adjust the utility levels
    #  Note that optimization is made to stick to households_per_income_class,
    #  which does not include people living in RDP (as we take this as
    #  exogenous), see equilibrium condition (i)

    #  We compare total population for each income group obtained from
    #  equilibrium condition (total_simulated_jobs) with target population
    #  allocation (households_per_income_class)

    #  We arbitrarily set a strictly positive minimum utility level at 10
    #  (as utility will be adjusted multiplicatively, we do not want to break
    #  the model with zero terms)
    diff_utility[index_iteration, :] = np.log(
        (total_simulated_jobs[index_iteration, :] + 10)
        / (households_per_income_class + 10))
    diff_utility[index_iteration, :] = (
        diff_utility[index_iteration, :] * param["convergence_factor"])
    (diff_utility[index_iteration, diff_utility[index_iteration, :] > 0]
     ) = (diff_utility[index_iteration, diff_utility[index_iteration, :] > 0]
          * 1.1)

    # Difference with reality
    error[index_iteration, :] = (total_simulated_jobs[index_iteration, :]
                                 / households_per_income_class - 1) * 100
    #  This is the parameter of interest for optimization
    error_max_abs[index_iteration] = np.nanmax(
        np.abs(total_simulated_jobs[index_iteration, :]
               / households_per_income_class - 1))
    error_max[index_iteration] = -1
    error_mean[index_iteration] = np.nanmean(
        np.abs(total_simulated_jobs[index_iteration, :]
               / (households_per_income_class + 0.001) - 1))
```

As long as the error metric is above a predetermined precision level (`precision`), and the number of iterations is below a predetermined threshold (`max_iter`), we repeat the process described above after updating the utility levels with the incremental value:

```python
            index_iteration = index_iteration + 1
            # When population is overestimated, we augment utility to reduce
            # population (cf. population constraint from standard model)
            utility[index_iteration, :] = np.exp(
                np.log(utility[index_iteration - 1, :])
                + diff_utility[index_iteration - 1, :])
            # This is a precaution as utility cannot be negative
            utility[index_iteration, utility[index_iteration, :] < 0] = 10

            # We augment the convergence factor at each iteration in propotion
```

Note that we also update the convergence factor to help the algorithm converge.

```python

            # NB: we assume the minimum error is 100 not to break model with
            # zeros
            convergence_factor = (
                param["convergence_factor"] / (
                    1 + 0.5 * np.abs((
                        total_simulated_jobs[index_iteration, :] + 100
                        ) / (households_per_income_class + 100) - 1))
                )

            # At the same time, we also reduce it while time passes, not to
            # demand too much of the algorithm and to help convergence
            convergence_factor = (
                convergence_factor
                * (1 - 0.6 * index_iteration / param["max_iter"])
                )

            # Now, we do the same as in the initalization phase

            # Compute outputs solver - first iteration
```

To complete the process, we concatenate (exogenous) values for formal subsidized housing to our final output vectors by taking care that all values have the same units:

```python
    #  Share of housing (no backyard) in RDP surface (with land in km²)
    construction_RDP = np.matlib.repmat(
        param["RDP_size"] / (param["RDP_size"] + param["backyard_size"]),
        1, len(grid_temp.dist))
    #  RDP dwelling size
    dwelling_size_RDP = np.matlib.repmat(
        param["RDP_size"], 1, len(grid_temp.dist))

    # We fill the vector for each housing type
    simulated_people_with_RDP = np.zeros((4, 4, len(grid_temp.dist)))
    simulated_people_with_RDP[0, :, selected_pixels] = np.transpose(
```

```python
        construction_RDP * dwelling_size_RDP * households_RDP
        / (coeff_land_full[3, :] * 0.25)  # * 1000000
        )
    housing_supply_RDP[np.isnan(housing_supply_RDP)] = 0
    dwelling_size_RDP = dwelling_size_RDP * (coeff_land_full[3, :] > 0)
    # dwelling_size_RDP[dwelling_size_RDP == 0] = np.nan
    initial_state_dwelling_size = np.vstack(
        [dwelling_size_export, dwelling_size_RDP])
    # Note that RDP housing supply per unit of land has nothing to do with
```

We also return other intermediate outputs from the model, such as utility levels, the final error, or capital per unit of available land.

### Subsequent periods

Back to the body of the `main.py` script, we save the outputs for the initial state equilibrium in a dedicated folder. Then, we launch the `run_simulation` function that take them as an argument, and calls on the same modules as before, plus the `functions_dynamic.py` module.

After initializing a few key variables, the function starts a loop over predefined simulation years. The first part of the loop consists in updating the value of all time-moving variables. This is based on exogenous scenarios previously imported as inputs (through the `import_scenarios` function) and taken as an argument of the function. Provided by the CoCT, they provide trajectories for the following set of variables:


* Income distribution


* Inflation rate


* Interest rate


* Total population


* Price of fuel


* Land use

This leads to the update of, among others, number of households per income class, expected income net of commuting costs, capital values of formal subsidized and informal dwelling units:

```python
             ) = eqdyn.compute_average_income(
                spline_population_income_distribution,
                spline_income_distribution, param, year_temp)
            income_net_of_commuting_costs = np.load(
                precalculated_transport + "GRID_incomeNetOfCommuting_"
                + str(int(year_temp)) + ".npy")
            (param["subsidized_structure_value"]
             ) = (param["subsidized_structure_value_ref"]
                  * (spline_inflation(year_temp) / spline_inflation(0)))
            (param["informal_structure_value"]
             ) = (param["informal_structure_value_ref"]
                  * (spline_inflation(year_temp) / spline_inflation(0)))
            mean_income = spline_income(year_temp)
```

Note that we also update the scale factor of the Cobb-Douglas housing production function so as to neutralize the impact that the inflation of incomes would have on housing supply through the rent :

```python
                (mean_income / param["income_year_reference"])
                ** (- param["coeff_b"]) * param["coeff_A"]
            )
            coeff_land = inpdt.import_coeff_land(
```

This is because we build a stable equilibrium model with rational agents. In particular, this requires to remove money illusion, that is the tendency of households to view their wealth and income in nominal terms, rather than in real terms.

Also note that we are updating land availability coefficents, as they evolve through time, and agricultural rent, which also depends on the interest rate and the updated scale factor:

```python
                spline_land_constraints, spline_land_backyard,
                spline_land_informal, spline_land_RDP, param, year_temp)

            agricultural_rent = inpprm.compute_agricultural_rent(
                spline_agricultural_rent(year_temp), construction_param,
                interest_rate, param, options)

```

Then, we proceed in three steps. We first compute a new unconstrained equilibrium with the updated variables. We then compute the targeted difference between the final value of formal private housing supply at $t+1$ and the one at at $t$, through the `evolution_housing_supply` function:

```python
                housing_limit, param, options, years_simulations[index_iter],
                years_simulations[index_iter - 1], tmpi_housing_supply[0, :],
                stat_temp_housing_supply[0, :])
            # We update the initial housing parameter as it will give the
```

This law of motion reflects the fact that formal private housing stock (only) depreciates with time and that developers respond to price incentives with delay as in Viguié and Hallegatte [[2012](#id23)], hence might differ from the one computed as an unconstrained equilibrium. Formally, this corresponds to:

s_{FP}(x|t+1) - s_{FP}(x|t)= \\begin{cases} \\frac{s_{FP}^{eq}(x|t+1) - s_{FP}(x|t)}{\\tau} - (\\rho + \\textcolor{green}{\\rho_{FP}^{struct}(x)}) s_{FP}(x|t) & \\text{if $s_{FP}(x|t) < s_{FP}^{eq}(x|t+1)$} \\\\ -(\\rho + \\textcolor{green}{\\rho_{FP}^{struct}(x)}) s_{FP}(x|t) & \\text{if $s_{FP}(x|t) \\geq s_{FP}^{eq}(x|t+1)$} \\end{cases}In the body of the `evolution_housing_supply` function, it translates into:

```python
        #                 * (t1 - t0) / param["time_invest_housing"]
        #                 - housing_supply_0 * (t1 - t0)
        #                 / param["time_depreciation_buildings"])
        diff_housing = ((housing_supply_1 - housing_supply_0)
                        * (housing_supply_1 > housing_supply_0)
```

Finally, we compute a new equilibrium under this constraint, which yields our final outputs at $t+1$. Concretely, we set the value of housing supply at $t+1$ as being the sum of the housing supply at $t$ plus the difference we just computed, and store it under the `housing_in` parameter (whose default value does not matter). In the body of the `run_simulations` function, it looks like:

```python
            + deriv_housing_temp

```

Then, we run the `compute_equilibrium` function with the modified option that developers do not adjust their housing supply anymore (so that `housing_supply = housing_in` in the body of the `compute_housing_supply_formal` function). All outputs will then be re-computed to correspond to this fixed target housing supply.

Back to the body of the `main.py` script, we save the simulation outputs in a dedicated folder, that will be used for result visualization.

## Outputs

All the modules of this package are used as part of the plots scripts. Those scripts can be run independently. The `plots_inputs.py` script plots input data for descriptive statistics. The `plots_equil.py` plots outputs specific to the initial state equilibrium, notably with respect to result validation. The `plots_simul.py` plots outputs for all simulation years, and some evolution of variables across time. Only the two latter require to run the main script at least once to save the associated numeric outputs. To call on a specific simulation, one just has to change the path name at the beginning of the scripts to use the dedicated folder. All scripts save the associated plots as images in a dedicated folder.

It should be noted that the resulting visuals are created with the `matplotlib` library and are pretty basic. Indeed, we thought of those plots not as a final deliverable, but rather as a way to quickly visualize results for developer use. Therefore, the plots scripts also save tables associated with values plotted, for use as part of a more user-friendly interface. This interface is developed jointly with the CoCT within the `streamlit` framework and uses `plotly` for graphical representations. It is accessible through this [link](https://kristoffpotgieter-nedumstreamlit-01--help-hozyn1.streamlitapp.com/). It notably simplifies the integration of visualization parameters (absolute vs. relative values, etc.) and the comparison between maps or graphs for different scenarios (or validation).

We rely on the interface for comments about the interpretation of results. As there exists a multiplicity of ways to visualize data, we do not see the plots scripts as a definite code for what can be plotted, but rather as a quick overview of the most important variables in the model. We believe the code to be self-explaining and redirect you to the [API reference]() section for function documentation. Suffices to say here that the `export_outputs.py` module is for processing and displaying the standard urban variables of the model (already present in Pfeiffer *et al.* [[2019](#id6)]), that the `flood_outputs.py` module is for processing values relative to floods, and that the `export_outputs_floods.py` module is for displaying them.

### Footnotes
