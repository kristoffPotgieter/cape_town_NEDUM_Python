---
date: '2022-09-23T09:22:05.479Z'
docname: input_tables
images: {}
path: /input-tables
title: Input tables
---

# Input tables

<!-- Need to modifiy the code accordingly + ask Claus and Basile to fill a reference column -->
## Key default parameters

> | Parameters

>  | Type

>  | Value

>  | Description

>  | Reference

>  | Modif

>  |
> | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ----------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |  |  |  |  |  |  |  |  |  |
> | agricultural_rent_2011

>                                                    | user

>                                                                                                                                             | 807.2

>                       | Value (in rands) of the agricultural land price in 2011

>  | Pfeiffer et al. (2019)

>  | 1

>                                                                                                                                                        |
> | alpha

>                                                                     | calib

>                                                                                                                                            | 0.75

>                        | Composite good elasticity in utility function

>            | /

>                       | 0

>                                                                                                                                                        |
> | backyard_pockets

>                                                          | calib

>                                                                                                                                            | /

>                           | Disamenity index (real value from 0 to 1) for informal settlements in each grid cell

>  | /

>                       | 0

>                                                                                                                                                        |
> | backyard_size

>                                                             | user

>                                                                                                                                             | 70

>                          | Size (in m²) of the backyard part in a RDP housing unit

>                               | Pfeiffer et al. (2019)

>  | 1

>                                                                                                                                                        |
> | baseline_year

>                                                             | user

>                                                                                                                                             | 2011

>                        | Baseline year to be used in the simulations

>                                           | /

>                       | 0

>                                                                                                                                                        |
> | beta

>                                                                      | calib

>                                                                                                                                            | 0.25

>                        | Surplus housing elasticity in utility function

>                                        | /

>                       | 0

>                                                                                                                                                        |
> | coeff_a

>                                                                   | calib

>                                                                                                                                            | 0.76

>                        | Land elasticity in housing production function

>                                        | /

>                       | 0

>                                                                                                                                                        |
> | coeff_A

>                                                                   | calib

>                                                                                                                                            | 0.03

>                        | Scale parameter in housing production function

>                                        | /

>                       | 0

>                                                                                                                                                        |
> | coeff_b

>                                                                   | calib

>                                                                                                                                            | 0.24

>                        | Capital elasticity in housing production function

>                                     | /

>                       | 0

>                                                                                                                                                        |
> | depreciation_rate

>                                                         | user

>                                                                                                                                             | 0.025

>                       | Depreciation rate of housing capital

>                                                  | Viguié et al. (2014)

>    | 1

>                                                                                                                                                        |
> | fraction_z_dwellings

>                                                      | user

>                                                                                                                                             | 0.53

>                        | Fraction of the composite good that is kept inside the house and that can possibly be destroyed by floods (food, furniture, etc.)

>  | Quantec, RIES 2011

>      | 1

>                                                                                                                                                        |
> | future_rate_public_housing

>                                                | user

>                                                                                                                                             | 1000

>                        | Number of formal subsidized housing units built per year in future simulations

>                                                     | Expert estimate

>         | 1

>                                                                                                                                                        |
> | historic_radius

>                                                           | user

>                                                                                                                                             | 6

>                           | Length (in km) of the radius defining the historic town from the city centre

>                                                       | Expert estimate

>         | 1

>                                                                                                                                                        |
> | household_size

>                                                            | user

>                                                                                                                                             | /

>                           | Average number of workers in a household (with unemployment) for each income group

>                                                 | Pfeiffer et al. (2019)

>  | 0

>                                                                                                                                                        |
> | income_centers_init

>                                                       | calib

>                                                                                                                                            | /

>                           | Value (in rands) of annual household income per income group and employment centre

>                                                 | /

>                       | 0

>                                                                                                                                                        |
> | income_class_by_housing_type

>                                              | user

>                                                                                                                                             | /

>                           | Multidimensional array of dummies setting income group eligibility to each housing type

>                                            | /

>                       | 0

>                                                                                                                                                        |
> | income_distribution

>                                                       | user

>                                                                                                                                             | /

>                           | List allocating each income group in the data (12) to one income group in the model (4)

>                                            | /

>                       | 0

>                                                                                                                                                        |
> | informal_structure_value

>                                                  | user

>                                                                                                                                             | 3000

>                        | Cost of inputs (in rands) for building an informal dwelling unit

>                                                                   | Expert estimate

>         | 1

>                                                                                                                                                        |
> | job_center_threshold

>                                                      | user

>                                                                                                                                             | 2500

>                        | Number of jobs above wich we retain a transport zone (TAZ) as an employment centre

>                                                 | /

>                       | 0

>                                                                                                                                                        |
> | lambda

>                                                                    | calib

>                                                                                                                                            | 2.68

>                        | Gravity parameter in expected income (net of commuting costs) calculation

>                                                          | /

>                       | 0

>                                                                                                                                                        |
> | limit_height_center

>                                                       | user

>                                                                                                                                             | 80

>                          | Maximum legal height (in m) for buildings in the historic town

>                                                                     | Expert estimate

>         | 1

>                                                                                                                                                        |
> | limit_height_out

>                                                          | user

>                                                                                                                                             | 10

>                          | Maximum legal height (in m) for buildings out of the historic town

>                                                                 | Expert estimate

>         | 1

>                                                                                                                                                        |
> | max_iter

>                                                                  | user

>                                                                                                                                             | 2000

>                        | Maximum number of iterations for equilibrium solver algorithm

>                                                                      | /

>                       | 0

>                                                                                                                                                        |
> | max_land_use

>                                                              | user

>                                                                                                                                             | 0.7

>                         | Maximum share of a grid cell land area available for formal private housing that can actually be built

>                             | Viguié et al. (2014)

>    | 1

>                                                                                                                                                        |
> | max_land_use_backyard

>                                                     | user

>                                                                                                                                             | 0.45

>                        | Maximum share of a grid cell land area available for informal backyard housing that can actually be built

>                          | Expert estimate

>         | 1

>                                                                                                                                                        |
> | max_land_use_settlement

>                                                   | user

>                                                                                                                                             | 0.4

>                         | Maximum share of a grid cell land area available for informal settlement housing that can actually be built

>                        | Expert estimate

>         | 1

>                                                                                                                                                        |
> | mini_lot_size

>                                                             | user

>                                                                                                                                             | 31.64

>                       | Minimum size (in m²) constraint for formal private housing units

>                                                                   | Pfeiffer et al. (2019)

>  | 1

>                                                                                                                                                        |
> | nb_of_income_classes

>                                                      | user

>                                                                                                                                             | 4

>                           | Number of income groups used in the model

>                                                                                          | /

>                       | 0

>                                                                                                                                                        |
> | pockets

>                                                                   | calib

>                                                                                                                                            | /

>                           | Disamenity index (real value from 0 to 1) for informal backyards in each grid cell

>                                                 | /

>                       | 0

>                                                                                                                                                        |
> | precalculated_amenities

>                                                   | calib

>                                                                                                                                            | /

>                           | Amenity index (real value from 0 to 1) in each grid cell

>                                                                           | /

>                       | 0

>                                                                                                                                                        |
> | precision

>                                                                 | user

>                                                                                                                                             | 0.02

>                        | Maximum error allowed for simulated number of households per income group (in %)

>                                                   | /

>                       | 0

>                                                                                                                                                        |
> | priceBusFixed

>                                                             | user

>                                                                                                                                             | 4.32

>                        | Fixed cost (in rands) for a one-way bus trip

>                                                                                       | Roux (2013)

>             | 0

>                                                                                                                                                        |
> | priceBusPerKM

>                                                             | user

>                                                                                                                                             | 0.785

>                       | Variable cost (in rands) for 1km of bus commuting

>                                                                                  | Roux (2013)

>             | 0

>                                                                                                                                                        |
> | priceFixedVehiculeMonth

>                                                   | user

>                                                                                                                                             | 400

>                         | Fixed cost (in rands) for one month of private car commuting

>                                                                       | Roux (2013)

>             | 0

>                                                                                                                                                        |
> | priceTaxiFixed

>                                                            | user

>                                                                                                                                             | 6.24

>                        | Fixed cost (in rands) for a one-way minibus/taxi trip

>                                                                              | Roux (2013)

>             | 0

>                                                                                                                                                        |
> | priceTaxiPerKM

>                                                            | user

>                                                                                                                                             | 0.522

>                       | Variable cost (in rands) for 1km of minibus/taxi commuting

>                                                                         | Roux (2013)

>             | 0

>                                                                                                                                                        |
> | priceTrainFixed

>                                                           | user

>                                                                                                                                             | 4.48

>                        | Fixed cost (in rands) for a one-way train trip

>                                                                                     | Roux (2013)

>             | 0

>                                                                                                                                                        |
> | priceTrainPerKM

>                                                           | user

>                                                                                                                                             | 0.164

>                       | Variable cost (in rands) for 1km of train commuting

>                                                                                | Roux (2013)

>             | 0

>                                                                                                                                                        |
> | q0

>                                                                        | calib

>                                                                                                                                            | 3.97

>                        | Basic need in housing (in m²) in utility function

>                                                                                  | /

>                       | 0

>                                                                                                                                                        |
> | RDP_size

>                                                                  | user

>                                                                                                                                             | 40

>                          | Size (in m²) of the dwelling part in a RDP housing unit

>                                                                            | Pfeiffer et al. (2019)

>  | 1

>                                                                                                                                                        |
> | shack_size

>                                                                | user

>                                                                                                                                             | 14

>                          | Size (in m²) of an informal settlement dwelling unit

>                                                                               | Expert estimate

>         | 1

>                                                                                                                                                        |
> | subsidized_structure_value

>                                                | user

>                                                                                                                                             | 127000

>                      | Cost of inputs (in rands) for building a formal subsidized dwelling unit

>                                                           | Expert estimate

>         | 1

>                                                                                                                                                        |
> | time_cost

>                                                                 | user

>                                                                                                                                             | 1

>                           | Multiplier associating the value of time lost in commuting to foregone revenues

>                                                    | /

>                       | 0

>                                                                                                                                                        |
> | time_depreciation_buildings

>                                               | user

>                                                                                                                                             | 100

>                         | Time (in years) for the full depreciation of a formal private housing unit

>                                                         | Viguié et al. (2014)

>    | 0

>                                                                                                                                                        |
> | time_invest_housing

>                                                       | user

>                                                                                                                                             | 3

>                           | Lag (in years) for formal private housing building in simulations

>                                                                  | Viguié et al. (2014)

>    | 0

>                                                                                                                                                        |
> | walking_speed

>                                                             | user

>                                                                                                                                             | 4

>                           | Speed (in km/h) at which representative households walk to work

>                                                                    | /

>                       | 0

>                                                                                                                                                        |
## Key default options

> | Options

>                                                                   | Value

>                                                                                                                                            | Description

>                 |
> | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
> | agents_anticipate_floods

>                                                  | 1

>                                                                                                                                                | Dummy for having agents perfectly anticipate flood risks

>  |
> | coastal

>                                                                   | 1

>                                                                                                                                                | Dummy for taking coastal floods into account (on top of fluvial floods)

>  |
> | correct_pluvial

>                                                           | 1

>                                                                                                                                                | Dummy for reducing pluvial risk for (better protected) formal structures

>  |
> | defended

>                                                                  | 0

>                                                                                                                                                | Dummy variable for considering defended (vs. undefended) fluvial flood maps from FATHOM

>  |
> | dem

>                                                                       | /

>                                                                                                                                                | Digital elevation model (DEM) to be used with coastal flood data (MERITDEM or NASADEM)

>   |
> | fuel_price_scenario

>                                                       | 2

>                                                                                                                                                | Categorical variable to select scenario for fuel price (should be set to 1/2/3 for low/medium/high price growth)

>  |
> | inc_ineq_scenario

>                                                         | 2

>                                                                                                                                                | Categorical variable to select scenario for income distribution (should be set to 1/2/3 for low/medium/high income inequality)

>  |
> | inflation_scenario

>                                                        | 2

>                                                                                                                                                | Categorical variable to select scenario for inflation rate (should be set to 1/2/3 for low/medium/high growth rate)

>             |
> | informal_land_constrained

>                                                 | 0

>                                                                                                                                                | Dummy for forbidding new informal housing construction

>                                                                          |
> | interest_rate_scenario

>                                                    | 2

>                                                                                                                                                | Categorical variable to select scenario for interest rate (should be set to 1/2/3 for low/medium/high growth rate)

>              |
> | pluvial

>                                                                   | 1

>                                                                                                                                                | Dummy for considering pluvial floods on top of fluvial

>                                                                          |
> | pop_growth_scenario

>                                                       | 3

>                                                                                                                                                | Categorical variable to select scenario for total population (should be set to 1/2/3 for low/medium/high population growth)

>     |
> | slr

>                                                                       | 1

>                                                                                                                                                | Dummy for taking sea-level rise into account in coastal flood data

>                                                              |
