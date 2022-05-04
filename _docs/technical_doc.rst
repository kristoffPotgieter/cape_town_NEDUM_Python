=======================
Technical documentation
=======================

-----------------
Code walk-through
-----------------

The project repository is composed of a main script, calling on several secondary scripts and user-defined packages. Those packages in turn include modules that are used at different steps of the code:

* Inputs: contain the modules used to import default parameters and options, as well as pre-treated data.
* Equilibrium: contain the modules used to compute the static equilibrium allocation, as well as the dynamic simulations.
* Output: contain the modules used to obtain output graphs for validation of the results and interpretation of the scenarios.
* Calibration: contain the modules used to re-run the calibration of parameters if necessary.

As for the main script, it starts with a preamble that imports (python and user-defined) packages to be used, and defines the file paths that integrate the repository to the local system of the user. It then goes on calling on the packages described above.

------
Inputs
------

In this part, the code calls on the ``parameters_and_options.py`` module that imports the default parameters and options, then sets the timeline for the simulations, and overwrites the default parameters and options according to the specific scenarios one is interested in. Typically, one may set options allowing or not agents to anticipate floods, or new informal settlements to be built. One may also choose the parameters associated to a given calibration process (see :ref:`calibration_process` for more details). One may go directly to the ``parameters_and_options.py`` script to see how the default parameters and options are set, and what one may want to change given the underlying hypotheses [#f1]_.
To store the output in a dedicated folder, we create a name associated to the parameters and options used.

Note that, in our model, we consider three endogenous housing markets (whose allocation is computed in equilibrium) - namely, formal private housing, backyard structures (erected in the backyard of subsidized-housing dwelling units), and informal settlements (in predetermined locations) - one exogenous housing market (whose allocation is directly taken from the data [#f2]_) - the RDP (Reconstruction and Development Programme) subsidized housing - and four income groups. By default, only agents from the poorest income group have access to RDP housing units. Without loss of generality, we assume that the price of such dwellings is zero, and that they are allocated randomly to the part of poorest agents they can host, the rest being rationed out of the subsidized housing market. Then, the two poorest income groups sort across formal private housing, backyard structures, and informal settlements, whereas the two richest only choose to live in the formal private housing units. We believe that this is a good approximation of reality. Also note that, although the two richest income groups are identical in terms of housing options, we distinguish between the two to better account in our simulations for income heterogeneity and spatial sorting along income lines, while keeping the model sufficiently simple to be solved numerically.

Then, the code calls on the ``data.py`` module. This allows to import basic geographic data, macro data, households and income data, land use projections, flood data, scenarios for time-dependent variables (see :doc:`../data_list` for more details and the sources and uses of those data sets), and transport data. It should be noted that, by default, housing type data has already been converted from the SAL (Small Area Level) dimension to the grid dimension, and that income net of commuting costs for each income group and each grid cell (from transport data) has already been computed from the calibrated incomes for each income group and each job center. If one wants to run the process again, one needs to overwrite the associated default options (please not that this may take some time to run). The import of flood data is also a bit time-consuming when agents are set to anticipate floods.

Again, the imported files can be modified directly, to account for changes in scenarios for instance, as long as the format used remains the same (see :doc:`../api_ref` for more details on each function requirements). Before going to the next step of the code, we would like to give more details on the imports of land use, flood, and transport data, which we think is not as transparent as the rest of the imports.

-----------
Equilibrium
-----------

------
Output
------

.. calibration_process
-----------
Calibration
-----------

------------
Ongoing work
------------

Formal vs. informal backyard, backyards in formal private units, interest rate


.. rubric:: Footnotes

.. [#f1] Please note that not all options are yet linked dynamically with the rest of the code, so that not all changes will result in coherent results.

.. [#f2] Note that there is no consensus on how to enumerate RDP dwelling units, as they are counted as formal dwelling units in census data. We follow a general validation procedure that draws on several data sources to estimate this number, and then allocate the units across the grid following expert advice...