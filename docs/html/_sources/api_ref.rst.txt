API reference
=============

Inputs
------

Set parameters and options
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: inputs.parameters_and_options
    :members:

Import data
^^^^^^^^^^^
.. automodule:: inputs.data
    :members:


Equilibrium
-----------

Compute equilibrium
^^^^^^^^^^^^^^^^^^^
.. automodule:: equilibrium.compute_equilibrium
    :members:

Run simulations
^^^^^^^^^^^^^^^
.. automodule:: equilibrium.run_simulations
    :members:

Dynamic functions
^^^^^^^^^^^^^^^^^
.. automodule:: equilibrium.functions_dynamic
    :members:

Compute intermediate outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: equilibrium.sub.compute_outputs
    :members:

Define optimality conditions for solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: equilibrium.sub.functions_solver
    :members:


Calibration
-----------

Main functions for calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: calibration.calib_main_func
    :members:

Calibrate income net of commuting costs and gravity parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: calibration.sub.compute_income
    :members:

Estimate utility function parameters by optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: calibration.sub.estimate_parameters_by_optimization
    :members:

Estimate utility function parameters by scanning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: calibration.sub.estimate_parameters_by_scanning
    :members:

Import exogenous amenities (for amenity index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: calibration.sub.import_amenities
    :members:

Import employment data (for income and gravity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: calibration.sub.import_employment_data
    :members:

Log-likelihood (for utility function parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: calibration.sub.loglikelihood
    :members:


Outputs
-------

Process and display general values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: outputs.export_outputs
    :members:

Process values related to floods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: outputs.flood_outputs
    :members:

Display values related to floods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: outputs.export_outputs_floods
    :members: