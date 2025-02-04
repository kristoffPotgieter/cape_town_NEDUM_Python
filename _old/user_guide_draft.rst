Write guidelines about NEDUM philosophy
=======================================

::

   See Res-IRF technical documentation + policies assessment



--------------

Write use cases (coastal floods?)
=================================

::

   See Res-IRF simulation and sensitivity analysis + setting up public
   policies

--------------

Add data (+ format) description
===============================

::
   See Res-IRF inputs
   Should we do a proper dictionary? Or should the format just be precised as part of docstrings 
   (makes sense, especially when variable dimensions to be interpreted)? Maybe we could also change the structure of the repo...

--------------

Where do pre-calculated inputs come from? What about `._` files?
Else, just matrix form for raw numbers or pixel parameters (24,014), or coord for TAZ (1,787) used in `transportMatrices.mat`?
`data.mat` is a structure with 54 fields characterizing SPs (1,046 or 1,013?) or pixels (24,014).
There seems to be 12 income brackets, converted into 4 for the analysis.
What are the 185 income centers (also used in `Transport_times` matrices)?
`model_amenity.mat` is a `GeneralizedLinearModel` with 307x18 geographic observations corresponding to SPs. But what is predicted?
What are the 22 dimensions in `timeCenters.mat`?

In `data_Cape_Town`:
-  Let's not focus on ancient and old files. 
-  `data_maps` contains geographic files treated with R scripts `Map_employment_centers`, `Map_future_projects`, `Map_grid_CT`,
`script_import_data_maps`. Should we update them?
- `Scenarios` contains tables for scenarios regarding different sets of variables (read description): what about tables not self-contained
in description?
- What about `OD_matrix`? Seems to be a square matrix for TAZ...
- Need to explain other tables or include format as part of docstrings?

`FATHOM` contains prospective flood probability tables at the pixel level (24,014), for fluvial and pluvial, defended and undefended:
cf. readme file.

`Flood plains - from Claus` contains a table for informal dwellings (number?) at the pixel level in 2020

`Land occupation` contains tables and maps for informal settlement risk scenarios: explain in more details?

`precalculated_transport` contains numpy arrays for income net of commuting over 20 years? Where does it come from?

Finally, at the folder root, we have a table of housing types (which unit?) per pixel. We also have no RDP, but a distinction
between formal and informal backyarding... What does this mean?

::
   We already have a variable dictionary from Charlotte: reformat it for use?


Code walk-through
=================

Organize files as in ``README.md``, open Spyder, define *code capetown
python* folder as a project, and run ``main.py``:

Preamble
--------

-  Load libraries, user packages, define file paths, and set up timing
   for optimization.

Import parameters and options
-----------------------------

-  Set baseline in ``inputs\parameters_and_options.py`` (**choice to be
   made between default and settings**).
-  Calibration needs to be run before (from ``old_calibration_code.py``)
   if needed to adjust parameters calibrated to location (please note
   that this may take a full day).

Load data
---------

-  Set data format in ``inputs\data.py`` (**make this uniform**). The
   question is: **should we make this implicit** (as for the scenario
   import, defined as part of the ``run simulation`` function) **or make
   all the imports explicit** (including scenarios)?
-  In ``inpdt.import_macro_data``, where does population figures come from? 
-  Get into details of ``inpdt.import_land_use``
-  Is ``inpdt.import_coeff_land`` useful?

Compute initial state
---------------------

-  Set equilibrium modelling in ``equilibrium\compute_equilibrium.py``
   (**describe more extensively in appendix**).

Validation
----------

-  Check how modelled initial state fits the data, using plots as
   defined in ``outputs\export_outputs.py`` and
   ``outputs\flood_outputs.py`` (**discuss what are the plots we want to
   appear**)
-  Should we add key parameter tables to make the reading of plots more
   self-contained?

Scenarios
---------

-  Set the time range of the simulation, then choose counterfactual
   parameters and options: note that those may differ from the ones used
   for setting up the initial state (**shouldn’t we just use other
   parameter names in simulation functions, so that we can choose them
   right from the start?**). Note that the longer the time frame, the
   longer is the code to run: it can take one full day for 30 years.
-  Run simulation from ``equilibrium\run_simulations.py``: note that
   this simply consists in iterating the ``compute_equilibrium``
   function over several years according to some predefined scenarios
   (**shouldn’t we allow for scenarios to be set as parameters rather
   than imports?**).
-  It should be noted that the code raises some warnings: this should be
   checked in the future. As a matter of fact, the process becomes
   increasingly slow with the number of iterations…
-  Save simulation results: shouldn’t we make the whole process
   implicit, as for validation results?

Plot output scenarios
---------------------

-  Again, shouldn’t we write this as part of a separate module and just
   call on the functions?
-  Even better, should we make the functions adapt to the selected time
   frame?
-  The code also raises some warnings that should be addressed

--------------

Add API reference + index (+ glossary for disambiguation)
=========================================================

Libraries
---------

-  Numpy
-  Pandas
-  Scipy
-  Seaborn
-  Time

User packages
-------------

1. ``inputs``
2. ``calibration``
3. ``equilibrium``
4. ``output``

NB: We need to enter system path C:/ as a parameter for all the pieces
of code + need to correct errors in SP_to_grid

NB: take care to circular references (no import statements within
functions), hidden coupling (not too many assumptions about other
files), centralize global variables/items (and reduce functions’
implicit context and side effects), no spaghetti or ravioli code, use
submodules if needed, replace ``import *`` statements (?), do not assign
a same variable name several times, think of tuples as immutable
equivalent of lists, use simple returns in functions (and raise
exceptions if needed), write docstrings for functions (and use
``doctest``), take care to variable names referencing the same object,
use ``enumerate()`` instead of counters, ``with open`` to read from
files (to ensure it closes)…

NB: should I add (empty) ``__init__.py`` files? use classes and methods
(only if object persistency)? use decorators (for memorization and
caching)? context managers? use function annotations and ``isinstance``?
Generators instead of iterables (with ``itertools``)?

Use pycodestyle (also autopep8, yapf, black)? Maybe not a good idea for
backward compatibility… Use ``.rst`` (with Sphinx, Read the docs)
instead of ``.md``?
