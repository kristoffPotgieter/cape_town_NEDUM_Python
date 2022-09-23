---
date: '2022-09-23T09:22:05.479Z'
docname: main_nb
images: {}
path: /main-nb
title: 'Notebook: run model'
---

# Notebook: run model

## Preamble

### Import packages

### Define file paths

This corresponds to the architecture described in the README file (introduction tab of the documentation): the data folder is not hosted on the Github repository and should be placed in the root folder enclosing the repo.

### Create associated directories if needed

### Set timeline for simulations

## Import parameters and options

### We import default parameter and options

### We also set custom options for this simulation

#### We first set options regarding structural assumptions used in the model

#### Then we set options regarding flood data used

#### We also set options for scenarios on time-moving exogenous variables

#### Finally, we set options regarding data processing

Default is set at zero to save computing time (data is simply loaded in the model).

NB: this is only needed to create the data for the first time, or when the source is changed, so that pre-processed data is updated.

## Give name to simulation to export the results

## Load data

### Basic geographic data

### Macro data

### Households and income data

### Land use projections

#### For reference, let us visualize the informal settlement risks considered



![image](inc_group_distrib.png)



![image](inc_group_distrib.png)


#### Let us visualize land availaibility ay baseline year (2011)

### Import flood data (takes some time when agents anticipate floods)

#### Let us visualize flood data

We will first show some flood maps for visual reference, then the associated fractions of capital destroyed computed through damage functions (final damages depend on spatial sorting).

NB: all maps are undefended (do not take protective infrastructure into account), and a return period of 100 years corresponds to a 1% chance of occurrence in a given year.

### Import scenarios (for time-moving variables)

### Import expected income net of commuting costs (for all time periods)

#### Let us visualize income net of commuting costs at baseline year

Note that this variable is computed through our commuting choice model, based on calibrated incomes per income group and job center.

## Compute initial state equilibrium

Reminder: income groups are ranked from poorer to richer, and housing types follow the following order: formal-backyard-informal-RDP

Note on outputs (with dimensions in same order as axes):

initial_state_utility = utility for each income group (no RDP) after optimization

initial_state_error = value of error term for each group after optimization

initial_state_simulated_jobs = total number of households per housing type (no RDP) and income group

initial_state_households_housing_types = number of households per housing type (with RDP) per pixel

initial_state_household_centers = number of households per income group per pixel

initial_state_households = number of households in each housing type and income group per pixel

initial_state_dwelling_size = dwelling size (in m²) for each housing type per pixel

initial_state_housing_supply = housing surface built (in m²) per unit of available land (in km²) for each housing type in each pixel

initial_state_rent = average rent (in rands/m²) for each housing type in each pixel

initial_state_rent_matrix = average willingness to pay (in rands) for each housing type (no RDP) and each income group in each pixel

initial_state_capital_land = value of the (housing construction sector) capital stock (in monetary units) per unit of available land (in km²) in each housing type (no RDP) and each selected pixel

initial_state_average_income = average income per income group (not an output of the model)

initial_state_limit_city = indicator dummy for having strictly more than one household per housing type and income group in each pixel

### Let us visualize key equilibrium outputs

#### Let us start with population distribution

We first look at sorting across housing types.

Here are a few caveats on how to interpret those results:


* For a given housing type, residential locations only vary a priori according to their (dis)amenity index, income net of commuting costs, and exposure to flood risks. We do not account for other location-specific exogenous factors, which explains the overall smooth aspect (compared to reality) of our spatial sorting. Besides, land availability is defined negatively by the share of land not available for other housing types, but in reality, this land may also be allocated to other uses, such as
commercial real estate. Therefore, even though we do simulate the model at the grid-cell level, it makes more sense to interpret results at the scale of the neighbourhood.


* The fact that we are not able to replicate some stylized facts for the CoCT should be interpreted in this regard. For instance, we are not able to reproduce the high density on the Atlantic Seaboard, as its amenities do not appear sufficient to offset its distance to the CBD. Likewise, the higher disamenity or higher uncertainty in income calibration for specific areas (such as Khayelitsha or Mitchell’s Plain) could explain why we are not able to reproduce the (formal) density in those areas.

Remember that backyarding essentially occurs within (exogenous) formal subsidized housing preccints, hence the observed spatial distribution.

Contrary to formal private housing, we do observe here a granular spatial distribution more in line with the data, that accounts for the fact that informal settlement locations are exogenously set.

Finally, the spatial distribution for formal subsidized housing is just taken from the data (not simulated).

We then look at sorting across income groups.

Overall, the distribution of the two poorest income groups is in line with what we could expect given their opportunities on the housing market: being (parly or fully) crowded out of the formal private segment, they redirect themselves to the informal segments and the formal subsidized segment (for those eligible).

The second richest group, which makes up the most part of formal private housing dwellers, illustrates the standard urban economics trade-off between job accessibility and high rents / small dwelling sizes, with a peak at mid-distance (abstracting from location specifics).

The richest income group, who has a higher opportunity cost of time (and better job opportunities), crowds out the second richest group near the CBD, but also does locate in more peripheral high-amenity areas where they overbid the second richest group.

#### We may also look at housing supply (in m²)

We do observe that informal settlements are somewhat denser than other housing types. This seems to indicate that horizontal densification dominates vertical densification within the context of the CoCT. Indeed, even though formal private housing can be built high, it is less constrained in terms of land availability and can spread out more, leaving some open space. On the contrary, as informal settlements are more constrained and can only expand horizontally, they end up using most of the
available land.

NB: Note that our model does not allow us to disentangle between high structures with small dwelling units and low structures with big dwelling units within the formal private housing sector.

#### Now, let us look at land prices (in rands / m²)

Our results conform to the standard urban economics predictions about the overall shape of the housing/land rent/price gradient.

Note that, although we also simulate the average annual rents for informal backyards and settlements, it is not absolutely rigorous to apply the above formula to recover land prices in those areas, as they are considered unfit for development. We still include the plots for reference, that can be interpreted as land prices should such areas become fit for development, keeping constant housing supply and demand (not very realistic).

Note that we cannot estimate land rents for formal subsidized parcels since such housing is exogenous in our model.

#### Finally, let us look at flood damages (in rands)

In the interest of space and for illustrative purposes, we only show results for the formal private sector structures (which are also the biggest in absolute terms).

We redirect the reader to the interface for a more detailed view on other housing submarkets or content damages, with results given as a share of income and distributional impacts, for instance.

Then, we get to plot the annualized value of some of those damages

As could have been expected from the flood maps, fluvial damages are more acute but also more localized than pluvial damages. Households seem to avoid the worst-affected areas, but are willing to trade off some flood exposure for good locations nearby the CBD (especially regarding fluvial risks). It is worth noting that the biggest pluvial damages occur in such places where flood exposure is not the highest: it is rather the increase in capital value (driven by a relaxed trade-off) that causes
the impact.

The same mechanism seems at play regarding coastal damages, but the estimated damages are well above standard estimates from the literature. This reflects the fact that we use sea-level rise projections based upon the (pessimistic) RCP 8.5 climate change scenario, but also the methodology we use: capital values are determined endogenously through the housing market, and not calibrated to reflect some maximum share of exposed capital at the city-level. As values are typically high near the CBD,
so are damages when floods hit such areas.

## Run simulations for subsequent periods (time depends on timeline length)

### Output visualization

All the above outputs are available at each period. For reference, we include here some aggregates that evolve over time.

We redirect the reader to the interface for a more detailed view of other aggregates, of which the evolution of flood damages for instance. Note that since flood risks do not evolve through time, the evolution in flood damages is just a function of population growth and spatial sorting.

#### Evolution of utility levels over the years

#### Evolution of population sorting across housing types
