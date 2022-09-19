===============
User guidelines
===============

-------------------
Elements of context
-------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Introduction to urban economic theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	"*The standard urban model has shown us that the price of land in large cities is similar to the gravity field of large planets that decreases with distance at a predictable rate. Ignoring land prices when designing cities is like ignoring gravity when designing an airplane*."

	\- Alain Bertaud, *Order without design* (2018)

Urban shape may be seen as the result of two sets of forces:

* State decisions, such as land-use constraints, zoning, transport policies, etc.
* Multiple decisions made by firms and inhabitants, through the real-estate and construction markets notably

The standard urban economic model - as envisioned by :cite:t:`alonso`, :cite:t:`muth`, :cite:t:`mills`, and formalized by :cite:t:`wheaton` - aims at analyzing market forces operating under the constraints imposed by state decisions. It showcases three main mechanisms:

* Households trading off between lower transportation costs and shorter commuting time when living close to the city center (where they work), vs. larger dwellings and lower rents in remote areas
* Local amenities (e.g. a nice view) modulating rents locally
* Investors optimizing housing density as a function of rents and construction costs

According to this model, if there is one major city center, rents and population density will be maximum there, and decrease when moving away. To reach this main conclusion (among others), the model makes a number of simplifying assumptions, most of which can be relaxed in more elaborate models to obtain more realistic results (see :cite:t:`duranton`) [#f1]_.

The key hypothesis of this set of models is that, in the long run, land use patterns reach a stationary equilibrium in which (similar) agents share a common utility level. Put in other words, they cannot reach a higher utility level by changing locations, which allows to rationalize the observed land use patterns through optimizing behaviours. It should be noted that, without further assumptions, it is not guaranteed that such spatial equilibrium is indeed optimal in terms of welfare. We therefore need to distinguish between a positive approach (adopted in **NEDUM-2D**) and a normative approach to urban economics.

The interest of such models grounded in economic theory compared to other land use and transport integrated (LUTI) simulation models [#f2]_ lies in their relative tractability and potential for causal interpretation (:cite:t:`arnott`). 

^^^^^^^^^^^^^^^^^^^^^
Previous developments
^^^^^^^^^^^^^^^^^^^^^

**NEDUM-2D**, for instance, which was initially developed for the Paris metropolitan area (:cite:t:`viguie`), belongs to this category of models. It directly applies a discrete two-dimensional version of the standard urban monocentric model for residential land use with several transport modes (see :cite:t:`fujita`) on a grid of pixels, accounting for zoning and land availability constraints defined at the pixel level. This allows for more realistic land use patterns than the basic model with a linear city. 

Furthermore, as indicated by its name, **NEDUM-2D** is a dynamic model that accounts for the fact that urban stationary equilibria may not exist: when population, transport prices, or incomes vary, housing infrastructure cannot adapt rapidly to changing conditions and is always out of equilibrium. Such inertia is imposed as a constraint on the theoretical equilibrium, which remains the basis for explaining observed land use patterns.

From its inception, **NEDUM-2D** was also designed to simulate prospective scenarios within the framework of mitigation and adaptation policies at the local scale: urban sprawl, climate vulnerability, as well as transport-related greenhouse gas emissions were included in the model. 

In subsequent versions, **NEDUM-2D** has been used within urban contexts as diverse as those of Buenos Aires, Toulouse, Wuhan, London...

^^^^^^^^^^^^^^^^^^^^^
Overview of the model
^^^^^^^^^^^^^^^^^^^^^

In its latest implementation, :cite:t:`pfeiffer` further refine the model by making it polycentric, and by introducing heterogeneous income groups, as well as informal housing situations that coexist with market and state-driven formal housing. The model is calibrated for the *City of Cape Town* (CoCT) and indeed allows to account for key features of cities in the developing world [#f3]_.

.. figure:: images/empl_loc.png 
   :scale: 70% 
   :align: center
   :alt: map of employment locations with number of workers per income group

   Employment locations used in the simulation, by income group (*Source*: :cite:t:`pfeiffer`)

More specifically, it considers two types of land and housing informality: informal settlements in predetermined locations (which is akin to squatting as in :cite:t:`brueckner`) and a rental market for backyard structures erected by owners of state-driven subsidized housing as modeled by :cite:t:`brueckner2`. It then integrates these elements within a closed-city model (with exogenous population growth) and simulates developers’ construction decisions as well as the housing and location choices of households from different income groups at a distance from several employment subcenters (while accounting for state-driven subsidized housing programs, natural constraints, amenities, zoning, transport options, and the costs associated with each transport mode).

It has displayed good performance, as shown by the validation plots below:

.. figure:: images/global_valid.png 
   :scale: 70% 
   :align: center
   :alt: line plots comparing population density and housing prices between simulation and data for the year 2011

   Comparison between simulation (green) and data (blue) for the year 2011 (*Source*: :cite:t:`pfeiffer`)

.. figure:: images/housing_valid.png 
   :scale: 70% 
   :align: center
   :alt: line plots comparing total population pet housing type between simulation and data for the year 2011

   Allocation of households to housing types and spatial distributions (*Source*: :cite:t:`pfeiffer`)

Ongoing work at the *World Bank* has been focusing on incorporating vulnerability to flood risks in this version of the model, by distinguishing between fluvial, pluvial, and coastal floods. Typically, fluvial floods are one-off, hard-to-predict water overflows from rivers, whereas pluvial floods designate rather seasonal surface water floods or flash floods, caused by extreme rainfall independently of an overflowing water body. Coastal floods correspond to hard-to-predict storm surges, with the added uncertainty of sea-level rise. The associated risks that we consider include:

* Structural damages: building depreciation caused by flood damages
* Contents damages: destruction of part of households’ belongings due to floods

We believe that those are the main channels through which flood risks affect the city patterns :cite:p:`pharoah` [#fQSE]_. Agents internalize those risks by considering the annualized value of those damages (based on probabilistic flood maps) as an added term in the depreciation of their housing capital and of their quantity of goods consumed (assimilated to a composite good) [#fmyopic]_.

As before, the model allows to simulate how these trade-offs might be affected by future demographic, climate, and policy scenarios.

|

-------------------
Policies assessment
-------------------

^^^^^^^^^^^^^^^^^^
Mechanisms at play
^^^^^^^^^^^^^^^^^^

Observe that in equilibrium, formal and informal housing markets are connected in several ways. 

Firstly, there is a direct connection due to the fact that, with the exception of subsidized housing beneficiaries who receive a transfer from the State, other poor households optimize across formal and informal residential options until their utilities are equalized. 

Secondly, the fact that informal settlements and backyarding locations are exogenously determined does not imply that formal and informal housing developments occur in isolation of one another. In fact, they are linked through the choices of poor households across formal and informal housing options, and because formal developers’ building decisions respond to private formal housing prices, with private formal housing prices partially reflecting the sorting of low-income households across formal and informal housing market segments. 

Finally, there is an externality associated with the use of land for informal settlements and for publicly subsidized housing as these areas are somehow taken away from developable land that would otherwise be available for private formal development. This affects the supply and demand for formal housing by restricting the set of potential locations available for private formal development, while accommodating a potentially large number of urban residents in the informal sector [#f4]_.

The main added mechanism from flood risk anticipation is that the poorest households might trade-off protection from flood risks for cheaper housing and better accessibility. Our preliminary results indeed show a tendency for informal settlements to expand in the near future, thereby increasing the vulnerability of the affected populations in absence of any mitigating investments.


^^^^^^^^^^^^^^^^^^^^^^^^^
Interpretation of results
^^^^^^^^^^^^^^^^^^^^^^^^^

**It should be noted that prospective scenarios only represent conceivable futures that may inform cost-benefit analysis, and have no predictive value per se, as many phenomena are neglected to preserve tractability**. 

As such, **NEDUM-2D** only makes predictions with respect to some simplifying assumptions (exogenous land availability and subsidized housing, etc.) and some economic mechanisms (housing supply and demand) described above [#f5]_. Although it is calibrated to stick closely to reality at present time for validation purposes, the number of parameters fed into the model is restricted to avoid overfitting and extreme sensitivity of the outputs to initial conditions. 

Indeed, the aim of such a model is to provide simulations for the future, with the largest external validity possible in the absence of observable counterfactuals. For them to be informative, they need to display complex direct and indirect effects while keeping tractable the mechanisms that cause them, hence the need to restrict the number of such mechanisms that are interacting in equilibrium. 

Here, **NEDUM-2D** preserves the main market mechanism from the standard urban economic model, while allowing for sorting across different housing submarkets. If one is interested in the impact of other mechanisms on land use patterns, one should probably consider another (non-economic) model. Also note that in its current version, **NEDUM-2D** does not allow to conduct proper welfare evaluations.

Empirically, :cite:t:`liotta` show that the standard urban economic model has a good predictive power in terms of population density and rent variations, but not so much in terms of housing production. However, they also show that high levels of informality, strong regulations and planning, as well as specific amenities are, as expected by the theory, main factors leading to the discrepancies. As we account for those elements, we believe that our model yields relatively good predictions. Still, as is common approach in the literature, we think that our most significant contribution is not to deliver predictions in absolute terms, but rather comparative statics that relate one scenario to another.

|

.. rubric:: Footnotes

.. [#f1] For a broader, less technical review of models used in spatial economics, see :cite:t:`glaeser`.

.. [#f2] See :cite:t:`wray` for a survey of land use modeling in South Africa.

.. [#f3] See :cite:t:`duranton2` for a review of urban economic models within the context of developing countries.

.. [#fQSE] Contrary to the so-called "Quantitative Spatial Economics" literature :cite:p:`rossi-hansberg`, we do not endogenize employment locations, to the extent that we do not allow firms to compete with households for land. There are two main reasons for that. First, the (relative) numerical simplicity of our model allows us to deal with several dimensions of heterogeneity within an extremely granular setting. Second, survey data and expert advice do not lead us to consider flood risks as a major potential shifter for job center distribution across the city. Since this is the focus of the current version, we therefore keep this distribution as fixed (more on that in :doc:`../technical_doc`) to focus on the housing mechanisms described above.

.. [#fmyopic] We still need to assess empirically to what extent those anticipations vary across flood risks, and how this may contribute to myopia in housing markets.

.. [#f4] The net effect on formal housing prices is ambiguous as the restricted supply of formal land should raise formal housing prices in the center, while pushing away population to peripheral areas where prices will be lower. Housing in the informal sector reduces the demand for formal housing, which exerts a downward pressure on formal housing prices.

.. [#f5] See :doc:`../technical_doc` for more details.


