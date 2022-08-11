========
NEDUM-2D
========

..Define as main README file and integrate with include directive

----------
Disclaimer
----------

**The contents of this repository are all in-progress and should not be expected to be free of errors or to perform any specific functions. Use only with care and caution**.

--------
Overview
--------

**NEDUM-2D** (Non-Equilibrium Dynamic Urban Model) is a tool for simulating land-use patterns across a city in two dimensions. The initial rationale for its development was to make urban simulations more realistic by modelling deviations from theoretical static equilibria as caused by inertia in the adaptive behaviour of households and developers over time. The current version developed for the *City of Cape Town* (CoCT) incorporates several transportation modes, employment centres, income groups, and housing types, to provide even more realistic prospective scenarios, while remaining tractable in terms of causal mechanisms at play. It allows to model the spatial impact of policies with a special interest for cities in developing countries, such as informal housing regulation and localized flood protection investments.

---------
Resources
---------

Documentation is freely available `here <https://domain.invalid/>`_.

A simple user interface is available `here <https://domain.invalid/>`_ to give an overview of **NEDUM-2D** main output.

------------
Installation
------------

**Step 1**: Git clone **NEDUM-2D** folder in your computer

* Use your terminal and go to a location where you want to store the **NEDUM-2D** project
* Type: ``git clone https://github.com/CIRED/cape_town_NEDUM_Python.git``

**Step 2**: Create a conda environment from the *environment.yml* file

* The *environment.yml* file is in the **NEDUM-2D** folder
* Use the terminal and go to the **NEDUM-2D** folder stored on your computer
* Type: ``conda env create -f nedum-2d-env.yml``

**Step 3**: Activate the new environment

* The first line of the *.yml* file sets the new environment’s name
* Type: ``conda activate NEDUM-2D``

**Step 4**: Launch **NEDUM-2D**

* Launch from **NEDUM-2D** root folder: ``python project/main.py``


---------------
Getting started
---------------

Project includes libraries, scripts and notebooks.
``../project`` is the folder containing scripts, notebooks, inputs and outputs.

The standard way to run **NEDUM-2D** is to launch the main script.
The model creates results in a folder in ``project/output``. Folder name is by default ``ddmmyyyy_hhmm`` (launching date and hour). By default, only a selection of the most important results and graphs are available.
For custom results, one needs to change the configuration in ``project/inputs/parameters_and_options.py`` and ``project/inputs/parameters_and_options.py``. For more details, see :ref:`input_config`.

Also note that the model runs on pre-calibrated parameters, as the calibration process is time-consuming. One only needs to re-run the process if the data used for calibration changes. If so, one needs to launch ``python project/precalibration.py`` before launching the main script. For more details, see :ref:`calibration_process`.

In the ``output/ddmmyyyy_hhmm`` folder, you will find the following files:

* ``detailed.csv`` for detailed outputs readable directly with an Excel-like tool
* ``summary_input.csv`` for a summary of main inputs
* ``.png`` graphs for equilibrium validation and simulation vizualisation.

For more refined use cases, launch one of the Jupyter Notebook analysis tools (work in progress).
Notebook templates are stored in ``project/nb_template_analysis``.
**Users should copy and paste the template notebook in another folder to launch it**.

-----------------
About the authors
-----------------

The development of the **NEDUM-2D** model was initiated at *CIRED* in 2014. Coordinated by Vincent Viguié, it involved over the years, in alphabetic order, Paolo Avner, Stéphane Hallegattte, Charlotte Liotta, Thomas Monnier, Basile Pfeiffer, Claus Rabe, Julie Rozenberg, and Harris Selod.

----
Meta
----

If you find **NEDUM-2D** useful, please kindly cite our last paper::

	@techreport{
	  author      = {Pfeiffer, Basile and Rabe, Claus and Selod, Harris and Viguié, Vincent},
	  title       = {Assessing Urban Policies Using a Simulation Model with Formal and Informal Housing:
	  Application to Cape Town, South Africa},
	  year        = {2019},
	  institution = {World Bank},
	  address     = {Washington, DC},
	  series      = {Policy Research Working Paper},
	  type        = {Working Paper},
	  number      = {8921},
	  url         = {https://openknowledge.worldbank.org/handle/10986/31987}
	}

Thomas Monnier – `@TLMonnier <https://twitter.com/TLMonnier>`_ – thomas.monnier@ensae.fr

Distributed under the GNU GENERAL PUBLIC LICENSE. See :doc:`../LICENSE` for more information.

https://github.com/CIRED/cape_town_NEDUM_Python