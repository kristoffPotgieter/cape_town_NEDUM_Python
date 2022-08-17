=======================
Technical documentation
=======================


-----------------
Code walk-through
-----------------

The project repository is composed of a ``main`` script and ``plots`` scripts, calling on several secondary scripts and user-defined packages. Those packages in turn include modules that are used at different steps of the code:

* ``inputs``: contains the modules used to import default parameters and options, as well as pre-treated data
* ``equilibrium``: contain the modules used to compute the static equilibrium allocation, as well as the dynamic simulations
* ``outputs``: contain the modules used to visualize results and data.
* ``calibration``: contains the modules used to re-run the calibration of parameters if necessary

As for the ``main`` script and ``plots`` scripts, they start with a preamble that imports (python and user-defined) packages to be used, and defines the file paths that integrate the repository to the local system of the user. They then go on calling the packages described above. Let us start with the ``main`` script.

|

------
Inputs
------

^^^^^^^^^^^^^^^^^^^^^^
Parameters and options
^^^^^^^^^^^^^^^^^^^^^^

In this part, the code calls on the ``parameters_and_options.py`` module that imports the default parameters and options, then sets the timeline for the simulations, and overwrites the default parameters and options according to the specific scenario one is interested in. The detail of key parameters and options is given in :doc:`../input_tables` [#f1]_. Typically, one may set options allowing or not agents to anticipate floods, or new informal settlements to be built. To store the output in a dedicated folder, we create a name associated to the parameters and options used:

.. literalinclude:: ../main.py
   :language: python
   :lines: 111-118
   :lineno-start: 111

This name is a code associated to the options, parameters, and scenarios that we changed across our main simulation runs. They can safely be updated by the end user.


^^^^^^^^^^^
Assumptions
^^^^^^^^^^^

Note that, in our model, we consider three endogenous housing markets (whose allocation is computed in equilibrium) - namely, formal private housing, informal backyards (erected in the backyard of subsidized housing dwelling units), and informal settlements (in predetermined locations) [#f2]_ - one exogenous housing market (whose allocation is directly taken from the data) - the RDP (Reconstruction and Development Programme) formal subsidized housing [#f3]_ - and four income groups.

By default, only agents from the poorest income group have access to RDP housing units. Without loss of generality, we assume that the price of such dwellings is zero, and that they are allocated randomly to the part of poorest agents they can host, the rest being rationed out of the formal subsidized housing market. Then, the two poorest income groups sort across formal private housing, informal backyards, and informal settlements; whereas the two richest only choose to live in the formal private housing units. Those assumptions are passed into the code through the ``income_class_by_housing_type`` parameter. We believe that this is a good approximation of reality. Also note that, although the two richest income groups are identical in terms of housing options, we distinguish between them to better account for income heterogeneity and spatial sorting along income lines in our simulations, while keeping the model sufficiently simple to be solved numerically.

.. figure:: images/model_assumpt.png 
   :scale: 60% 
   :align: center
   :alt: summary table of the modeling assumptions regarding housing

   Modeling assumptions regarding housing (*Source*: :cite:t:`pfeiffer`)


^^^^
Data
^^^^

Then, the code calls on the ``data.py`` module. This allows to import basic geographic data, macro data, households and income data, land use projections, flood data, scenarios for time-dependent variables, and transport data. The databases used are presented in :doc:`../data_bases`. A more detailed view on the specific data sets called in the code, and their position in the folder architecture, is given in :doc:`../data_sets` [#fdata]_.

It should be noted that, by default, housing type data has already been converted from the SAL (Small Area Level) dimension to the grid dimension (500x500m pixels), and that income net of commuting costs for each income group and each grid cell (from transport data) has already been computed from the calibrated incomes for each income group and each job center. If one wants to run the process again, one needs to overwrite the associated default options (please note that this may take some time to run). The import of flood data is also a little time-consuming when agents are set to anticipate floods.

The imported files can be modified directly in the repository, to account for changes in scenarios for instance, as long as the format used remains the same (see :doc:`../api_ref` for more details on each function requirements). Before going to the next step of the code, we would like to give more details on the imports of land use, flood, and transport data, which we think are not as transparent as the rest of the imports.

.. _land_avail_desc:

"""""""""""""
Land use data
"""""""""""""

The ``import_land_use`` and ``import_coeff_land`` functions help define exogenous land availability :math:`L^h(x)` for each housing type :math:`h` and each grid cell :math:`x`. Indeed, we assume that different kinds of housing do not get built in the same places to account for insecure property rights (in the case of informal settlements vs. formal private housing) and housing specificities (in the case of non-market formal subsidized housing, and informal backyards located in the same preccints) [#fmixed]_. Furthermore, :math:`L` varies with :math:`x` to account for both natural, regulatory constraints, infrastructure and other non-residential uses.

As :math:`L^h(x)` is going to vary across time periods, the first part of the ``import_land_use`` function imports lacking data estimates for historical and projected land uses. It then defines linear regression splines over time for a set of variables, the key ones being the share of pixel area available for formal subsidized housing, informal backyards, informal settlements, and unconstrained development. Note that, in our benchmark, land availability for informal settlements is defined over time by the intersection between the timing map below and the high and very high probability areas (also defined below).

.. figure:: images/WBUS2_Land_occupation_timing.png 
   :scale: 15% 
   :align: center
   :alt: map for estimated timing of future informal settlements

   Estimated timing of future new informal settlements

.. figure:: images/WBUS2_Land_occupation_probability.png 
   :scale: 15% 
   :align: center
   :alt: map for estimated probability of future informal settlements

   Estimated probability of future new informal settlements

The ``import_coeff_land`` function then takes those outputs as arguments and reweight them by a housing-specific maximum land use parameter. This parameter allows to reduce the development potential of each area to its housing component (accounting for roads, for instance). The share of pixel area available for formal private housing (in a given period) is simply defined as the share of pixel area available for unconstrained development, minus the shares dedicated to the other housing types, times its own maximum land use parameter:

.. literalinclude:: ../inputs/data.py
   :language: python
   :lines: 670-681
   :lineno-start: 670

The outputs are stored in a ``coeff_land`` vector that yields the values of :math:`L^h(x)` for each time period when multiplied by the area of a pixel. Results are shown in the figure below.

.. figure:: images/land_avail.png 
   :scale: 90% 
   :align: center
   :alt: map of land availability ratios per housing type

   Share of available land for each housing type (*Source*: :cite:t:`pfeiffer`)


""""""""""
Flood data
""""""""""

Flood data is processed through the ``import_init_floods_data``, ``compute_fraction_capital_destroyed``, and ``import_full_floods_data`` functions.

The ``import_init_floods_data`` function imports the pre-processed flood maps from FATHOM (fluvial and pluvial), and DELTARES (coastal). Those maps yield for each grid cell an estimate of the pixel share that is exposed to a flood of some maximum depth level, reached in a given year with a probability equal to the inverse of their return period. For instance, a flood map corresponding to a return period of 100 years considers events that have a 1/100 chance of ocurring in a given year.

.. figure:: ../../4.\ Sorties/input_plots/P_100yr_map_depth.png
   :scale: 70% 
   :align: center
   :alt: map of land availability ratios per housing type

   Maximum pluvial flood depth (in meters) for a 100-year return period (*Source*: FATHOM)

Note that the considered return periods are not the same for FATHOM and DELTARES data, and that different flood maps are available depending on the digital elevation model (DEM) considered, whether we account for sea-level rise, and whether we account for defensive infrastructure with respect to fluvial floods [#f4]_. 

The function then imports depth-damage functions from the existing literature. Those functions, used in the insurance market, link a level of maximum flood depth to a fraction of capital destroyed, depending on the materials affected by floods [#f5]_. More specifically, we use estimates from :cite:t:`englhardt` and :cite:t:`hallegatte` for damages to housing structures (depending on housing type), and from :cite:t:`villiers` for damages to housing contents.

On this basis, the ``compute_fraction_capital_destroyed`` function integrates flood damages over the full range of return periods possible, for each flood type and damage function [#f6]_. This yields an annualized fraction of capital destroyed, corresponding to its expected value in a given year [#f7]_.

Finally, the ``import_full_floods_data`` function uses those outputs to define the depreciation term :math:`\rho_{h}^{d}(x)` that is specific to housing type :math:`h`, damage type (structures or contents) :math:`d`, and location :math:`x`, by taking the maximum of fractions of capital destroyed for all flood types considered. When multiplied by some capital value, this term yields the expected economic cost from flood risks that is considered by anticipating agents when solving for equilibrium. It is also equal to the risk premium in the case of a perfect insurance market (leading to full insurance with actuarially fair prices) [#f8]_.

.. figure:: ../../4.\ Sorties/input_plots/structure_informal_settlements_fract_K_destroyed.png
   :scale: 70% 
   :align: center
   :alt: map of annualized fraction of capital destroyed in informal settlements

   Annualized fraction of capital destroyed in informal settlements

""""""""""""""
Transport data
""""""""""""""

Transport data is processed through the ``import_transport_data`` function. It imports monetary and time transport costs and pre-calibrated incomes per income group and job center (more on that in :ref:`calibration_process`). From there, it computes several outputs, of which the key variable is the expected income net of commuting costs :math:`\tilde{y}_i(x)`, earned by a household of income group :math:`i` choosing to live in :math:`x`. It is obtained by recursively solving for the optimal transport mode choice and job center choice of households characterized by :math:`i` and :math:`x`. 

.. math::



.. _calibration_process:

-----------
Calibration
-----------

This whole part is optional. The reason is that all necessary parameters have been pre-calibrated, and it is only useful to run this part again if the underlying data used for calibration has changed. Also note that it takes time to run. If needed, it has to be run before solving the equilibrium, which is why it was included at this stage of the script. However, it relies on the estimation of partial relations derived from the general equilibrium itself [#fcalib]_. For better understanding, we therefore advise you to get back to this section after reading the :ref:`equilibrium_desc` section.

The preamble imports more useful data, under some technical options. Note that, whereas the same data might be used for calibration of parameters validation of results, it is never an input of the model per se. Typically, this is data at the SAL or SP (Small Place) level, less granular that our counterfactual grid-level estimates.

Then, we calibrate construction function parameters using the ``estim_construct_func_param`` function.

|

.. _equilibrium_desc:

-----------
Equilibrium
-----------

This part of the main script simply calls on two functions that return the key outputs of the model: ``compute_equilibrium`` solves the static equilibrium for baseline year, and ``run_simulation`` solves its dynamic version for all pre-defined subsequent years.

^^^^^^^^^^^^^
Initial state
^^^^^^^^^^^^^

Let us first dig into the ``compute_equilibrium`` function. Our main input is the total population per income group in the city at baseline year. Since we took the non-employed (earning no income over the year) out to define our four income groups, we need to reweight it to account for the overall population. Results are given in the table below.

.. figure:: images/inc_group_distrib.png 
   :scale: 60% 
   :align: center
   :alt: summary table of income groups characteristics

   Income groups used in the simulation (*Source*: :cite:t:`pfeiffer`)

Then, considering that all formal subsidized housing belongs to the poorest income group, we substract the corresponding number of households from this class to keep only the ones whose allocation in the housing market is going to be determined endogenously. We shorten the grid to consider only habitable pixels according to land availability and expected income net of commuting costs to alleviate numeric computations and initialize a few key variables before starting the optimization per se.

.. _solving_desc:

"""""""""""""""""""""""
Solving for equilibrium
"""""""""""""""""""""""

Note that, among those variables, we define arbitrary values for utility levels across income groups. This relates to the way this class of models is solved: as a closed-city equilibrium model, **NEDUM-2D** takes total population (across income groups) as exogenous, and utility levels (across income groups) as endogenous [#f9]_.

It is then solved iteratively in four steps:
* We first derive housing demand for each housing type
* We deduce rents for each housing type
* We then compute housing supply for each housing type
* From there, we obtain population in all locations for all housing types. We update initial utility levels depending on the gap between the objective and simulated population, and re-iterate the process until we reach a satisfying error level [#f10]_.
Of course, the closer the initial guess is to the final values, the quicker the algorithm converges. We will see below how each of the intermediate steps is computed.

A last word on the choice of a closed vs. open-city model: within a static framework, it is generally considered that closed-city models are a better representation of short-term outcomes and open-city models of long-term outcomes, as population takes time to adjust through migration. Here, we rely on scenarios from the CoCT (informed by more macro parameters than open-city models usually are) to adjust total population across time periods. Sticking to the closed-city model in those circumstances allows us to make welfare assessments based on utility changes without renouncing to the possibility that migrations occur.

""""""""""""""""""""""
Functional assumptions
""""""""""""""""""""""

Then, the ``compute_equilibrium`` function calls on the ``compute_outputs`` function for each endogenous housing type, which in turn calls on functions defined as part of the ``functions_solver.py`` module. This module applies formulas derived from the optimization process described in :cite:t:`pfeiffer`. 

Let us just recall the main assumptions here (refer to the paper for a discussion on those assumptions): households optimize over Stone-Geary preferences described in equation (4), under a budget constraint described in equation (8); and formal private developers have a Cobb-Douglas housing production function and optimize over a profit function (per unit of available land) described before equation (6).

Underlying those functional forms are structural assumptions about the maximization objective of agents in each housing submarket:
* In the formal private sector, developers buy land from absentee landlords and buy capital (directly given in monetary values) to build housing on this land [#fabsentee]_. They then rent out the housing at the equilibrium rent over an infinite horizon [#fconstant]_. They therefore internalize the costs associated with capital depreciation (both general and from structural flood damages) and interest rate (at which future flows of money are discounted). Households just have to pay for damages done to the content of their home [#fequiv]_.
* In the formal subsidized sector, (poor) households rent housing for free from the state. They only pay for overall capital depreciation (general and from structural and content damages), and may rent out a fraction of their backyard.
* In the informal backyard sector, household rent a fraction of backyard owned by formal subsidized housing residents and are responsible for building their own "shack". Then, they also pay for overall capital depreciation (general and from structural and content damages).
* In the informal settlement sector, households rent land from absentee landlords (not the same as in the formal private sector) and are responsible for building their own "shack". Then, they also pay for overall capital depreciation (general and from structural and content damages).

"""""""""""""""""""""""""
Equilibrium dwelling size
"""""""""""""""""""""""""

As described in the :ref:`solving_desc` subsection, the ``compute_outputs`` functions starts by computing housing demand / dwelling size (in m²) for each housing type. Equation (10) from :cite:t:`pfeiffer` implicitly defines this quantity in the formal private sector. The ``compute_dwelling_size_formal`` function then recovers the value (depending on income group) through a linear interpolation, before constraining it to be bigger than the parametrized minimum dwelling size in this sector. The dwelling size in the informal backyards and informal settlements sectors is exogenously set as being the standard parametrized size of a "shack".

""""""""""""""""
Equilibrium rent
""""""""""""""""

Then, we use equations (11), (13), and (14) from :cite:t:`pfeiffer` to recover the bid rents for each income group in the formal private, informal backyard, and informal settlement sectors respectively. Bid rents :math:`\psi^i_h(x,u)` correspond to the maximum amount households of type :math:`i` are willing to pay for a unit (1 m²) of housing of type :math:`h` in a certain location :math:`x` for a given utility level :math:`u` (over one year). Assuming that households bid their true valuation and that there are no strategic interactions, housing / land [#fhland]_ is allocated to the highest bidder. This is why we retain the bid rents from the highest bidding income groups, and the associated dwelling sizes, as the equilibrium output values.

""""""""""""""""""""""
Optimal housing supply
""""""""""""""""""""""

In the formal private sector, the ``compute_housing_supply_formal`` function applies formula (6) from :cite:t:`pfeiffer` to get the housing supply per unit of available land (in m² per km²), after selecting the appropriate damage function [#fland]_. The ``compute_housing_supply_backyard`` function does the same for informal backyards with formula (7). For informal settlements, the housing supply per unit of available land is just set as 1 km²/km². Indeed, as absentee landlords do not have outside use for their land, they face no opportunity cost when renting it out, and therefore rent all of it. We also recall that the informal backyard and settlement dwellings are not capital-intensive, to the extent that they have a fixed size and cover only one floor. The housing supply is therefore equal to the land supply. Again, this is a simplification, but we believe that this is a good enough approximation of reality.

""""""""""""""""""""""""""""""""
Equilibrium number of households
""""""""""""""""""""""""""""""""

At the end of the ``compute_outputs`` function, we just divide the housing supply per unit of available land by the dwelling size, and multiply it by the amount of available land to obtain the number of households by housing type in each grid cell. Then, we associate people in each selected pixel to the highest bidding income group. From there, we are able to recover the total number of households in each income group for a given housing type.

"""""""""""""""""""""
Iteration and results
"""""""""""""""""""""

Back to the body of the ``compute_equilibrium`` function, we sum the outputs of the ``compute_outputs`` function to get the total number of households in each income group. Then, we define an error metric ``error_max_abs`` comparing this result with values from the data, and an incremental value ``diff_utility`` (depending on a predetermined convergence factor). As long as the error metric is above a predetermined precision level, and the number of iterations is below a predetermined threshold, we repeat the process described above after updating the utility levels with the incremental value. Note that we also update the convergence factor to help the algorithm converge.

To complete the process, we concatenate (exogenous) values for formal subsidized housing to our final output vectors by taking care that all values have the same units. We also return other intermediate outputs from the model, such as utility levels, the final error, or capital per unit of available land.

^^^^^^^^^^^^^^^^^^
Subsequent periods
^^^^^^^^^^^^^^^^^^

Back to the body of the ``main.py`` script, we save the outputs for the initial state equilibrium in a dedicated folder.
Then, we launch the ``run_simulation`` function that take them as an argument, and calls on the same modules as before, plus  the ``functions_dynamic.py`` module.

After initializing a few key variables, the function starts a loop over predefined simulation years. The first part of the loop consists in updating the value of all time-moving variables. This is based on exogenous scenarios previously imported as inputs (through the ``import_scenarios`` function) and taken as an argument of the function. Provided by the CoCT, they provide trajectories for the following set of variables:
* Income distribution
* Inflation rate
* Interest rate
* Total population
* Price of fuel
* Land use

This leads to the update of, among others, number of households per income class, expected income net of commuting costs, capital values of formal subsidized and informal dwelling units. Note that we also update the scale factor of the Cobb-Douglas housing production function so as to neutralize the impact that the inflation of incomes would have on housing supply through the rent (see equation (6) from :cite:t:`pfeiffer`). This is because we build a stable equilibrium model with rational agents. In particular, this requires to remove money illusion, that is the tendency of households to view their wealth and income in nominal terms, rather than in real terms. Also note that we are updating land availability coefficents, as they evolve through time, and agricultural rent, which also depends on the interest rate and the updated scale factor [#fagri]_.

Then, we proceed in three steps. We first compute a new unconstrained equilibrium with the updated variables. We then compute the targeted difference between the final value of housing supply at :math:`t+1` and the one at at :math:`t` (according to equation (15) in :cite:t:`pfeiffer`), through the ``evolution_housing_supply`` function. This law of motion reflects the fact that formal private housing stock (only) depreciates with time and that developers respond to price incentives with delay as in :cite:t:`vhallegatte`, hence might differ from the one computed as an unconstrained equilibrium. Finally, we compute a new equilibrium under this constraint, which yield our final outputs at :math:`t+1`. Concretely, we set the value of housing supply at :math:`t+1` as the sum of the housing supply at :math:`t` plus the difference we just computed, and run the ``compute_equilibrium`` function with the modified option that developers do not adjust their housing supply anymore. All outputs will then be re-computed to correspond to this fixed target housing supply.

Back to the body of the ``main.py`` script, we save the simulation outputs in a dedicated folder, that will be used for result visualization.

|

-------
Outputs
-------

All the modules of this package are used as part of the plots scripts. Those scripts can be run independently. The ``plots_inputs.py`` script plots input data for descriptive statistics. The ``plots_equil.py`` plots outputs specific to the initial state equilibrium, notably with respect to result validation. The ``plots_simul.py`` plots outputs for all simulation years, and some evolution of variables across time. Only the two latter require to run the main script at least once to save the associated numeric outputs. To call on a specific simulation, one just has to change the path name at the beginning of the scripts to use the dedicated folder. All scripts save the associated plots as images in a dedicated folder.

It should be noted that the resulting visuals are created with the ``matplotlib`` library and are pretty basic. Indeed, we thought of those plots not as a final deliverable, but rather as a way to quickly visualize results for developer use. Therefore, the plots scripts also save tables associated with values plotted, for use as part of a more user-friendly interface. This interface is developed jointly with the CoCT within the ``streamlit`` framework and uses ``plotly`` for graphical representations. It is accessible through this `link <https://kristoffpotgieter-nedumstreamlit-01--help-hozyn1.streamlitapp.com/>`_. It notably simplifies the integration of visualization parameters (absolute vs. relative values, etc.) and the comparison between maps or graphs for different scenarios (or validation).

We rely on the interface for comments about the interpretation of results. As there exists a multiplicity of ways to visualize data, we do not see the plots scripts as a definite code for what can be plotted, but rather as a quick overview of the most important variables in the model. We believe the code to be self-explaining and redirect you to the :doc:`../api_ref` section for function documentation. Suffices to say here that the ``export_outputs.py`` module is for processing and displaying the standard urban variables of the model (already present in :cite:t:`pfeiffer`), that the ``flood_outputs.py`` module is for processing values relative to floods, and that the ``export_outputs_floods.py`` module is for displaying them.

|

.. rubric:: Footnotes

.. [#f1] The type of a parameter indicates whether it is plugged as a raw value or is the outcome of a calibration process. In theory, all "user" defined parameters can be overwritten by the end user. In practice, we cannot guarantee that the model runs seamlessly for all possible changes in values. We therefore encourage you to ask for guidance if you would like to change a parameter that was not included in main simulation runs.

.. [#f2] We are aware of the existence of formal (concrete) backyard structures but disregard their existence for the sake of model tractability and as they only represent a minority of all backyard settlements.

.. [#f3] Note that there is no consensus on how to enumerate RDP dwelling units, as they are counted as formal dwelling units in census data. We follow a general validation procedure that draws on several data sources to estimate this number (including council housing), and then allocate the units across the grid using the CoCT's cadastre.

.. [#fdata] Not all the "raw" data sets (as opposed to "user" data sets that result from a calibration process and can be updated by the end user) used are actually raw data. Some of them have been pre-processed, and this explains why there are more files (used as intermediary inputs) in the projet repository than described in :doc:`../data_sets`. Some other files may be related to older versions of the model, or may have been imported in group with other data sets. They are kept as reference and may serve for later implementations.

.. [#fmixed] To some extent, this precludes mixed land use, which we do not see as a major issue given that the granularity of our setting allows us to approximate it finely.

.. [#f4] More details about flood data can be found `here <https://www.fathom.global/product/flood-hazard-data-maps/fathom-global/>`__ and `here <https://microsoft.github.io/AIforEarthDataSets/data/deltares-floods.html>`__. Typically, underlying prediction models consider past and forecasted precipitation and storms, current river levels, as well as soil and terrain conditions.

.. [#f5] We are aware that there are other factors than maximum flood depth (such as duration and intensity) that may affect flood severity. However, partly due to data limitations, we believe that focusing on flood depth is a good first approximation.

.. [#f6] More comments on the integration method used can be found `here <https://storymaps.arcgis.com/stories/7878c89c592e4a78b45f03b4b696ccac>`__.

.. [#f7] Note that we add an option to discount the most likely / less serious pluvial flood risks for formal private housing, then for formal subsidized and informal backyard structures. This is to account for the fact that such (more or less concrete) structures are better protected from run-offs, which is not an information provided by the flood maps.

.. [#f8] For the purpose of this model, it is therefore equivalent to assume complete market insurance or self-insurance. We may actually have an incomplete combination of the two, which we could simulate by putting weights on our two perfect-anticipations and no-anticipations polar cases. Note however that we do not model endogenous self-protection investments.

.. [#fcalib] In the absence of quasi-natural experiments, for such estimated parameters to be properly identified in our simulations, we need to assume that the variables used in the calibration are close to a long-term equilibrium at the baseline year we study (no deviations due to the durability of housing, etc.). A good robustness check would be to see how our estimates would change when running the calibration over previous periods.

.. [#f9] We recall that, according to the spatial indifference hypothesis, all similar households share a common (maximum attainable) utility level in equilibrium. In our model, households only differ a priori in their income class, which is why we have a unique utility level for each income class. Intuitively, the richer the household, the bigger the utility level, as a higher income translates into a bigger choice set. If such utility levels can be compared in ordinal terms, no direct welfare assessment can be derived in cardinal terms: utilities must be converted into income equivalents first. A further remark is that income levels are a proxy for household heterogeneity in the housing market. They could themselves be endogenized (including unemployment) to study interactions with the labor market, although we leave that for future work.

.. [#f10] When we overestimate the population, we increase the utility level, and conversely. Indeed, a higher utility translates into a higher land consumption (all other things equal) for the same land available, hence a lower accommodable number of people.

.. [#fabsentee] The alternative to the absentee landlord assumption is the public ownership assumption: in that case, all housing revenues are equally shared among residents of the city. The choice is supposed to reflect the ownership structure of the city, but in practice it has little impact on the spatial structure of the city :cite:p:`avner`. The main difference is that we do not consider the same notions of welfare in each case.

.. [#fconstant] Note that, in this kind of static models where prices and rents are constant over time, it is equivalent to consider choices to buy or to rent, as price is just the capitalized value of all future rent flows. We will therefore focus on housing rents only, as is standard in the literature.

.. [#fequiv] Note that in the benchmark case with no market failure, tax equivalence results hold that the burden of added maintenance costs (that we assimilate to a tax) is shares across supply and demand, not on the basis of which side the tax applies to, but depending on their relative elasticities :cite:p:`auerbach`. Therefore, the structural assumptions made on the distribution of those costs should not have an impact on overall welfare.

.. [#fhland] Remember that households rent housing in the formal private and subsidized sectors, but direcly rent land in the informal backyard and settlement sectors. 

.. [#fland] Recall that available land has been exogenously defined in the :ref:`land_avail_desc` subsection and is not equal to total land area. We should apply appropriate conversions to visualize the desired outputs.

.. [#fagri] We assume that the price of agricultural land is fixed (corresponds to landlords' outside options / opportunity cost) and that agricultural land is undeveloped. Since we assume that developers make zero profit in equilibrium due to pure and perfect competition, this gives us a relation to obtain the minimum rent developers would be willing to accept if this land were urbanized (see footnote 16 in :cite:t:`pfeiffer`). Below this rent, it is therefore never profitable to build housing. Agricultural rent defines a floor on equilibrium rent values as well as an endogenous city edge.