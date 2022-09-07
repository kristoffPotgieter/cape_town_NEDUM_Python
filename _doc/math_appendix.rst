=============
Math appendix
=============

-------------------------------------------
Commuting choice and net income calculation
-------------------------------------------

The terms highlighted in orange relate to functional assumptions needed to solve the model. For each mode :math:`m`, residential location :math:`x`, job center :math:`c`, income group :math:`i`, and worker :math:`j`, **expected commuting cost** is:

.. math::

    t_{mj}(x,c,w_{ic}) = \chi_i(\tau_m(x,c) + \delta_m(x,c)w_{ic}) + \textcolor{orange}{\epsilon_{mxcij}}

* :math:`w_{ic}` is the (calibrated) wage earned.
* :math:`\chi_i` is the employment rate of the household, composed of two agents (``household_size`` parameter).
* :math:`\tau_m(x,c)` is the monetary transport cost.
* :math:`\delta_m(x,c)` is a time opportunity cost parameter (the fraction of working time spent commuting).
* :math:`\textcolor{orange}{\epsilon_{mxcij}}` follows a Gumbel minimum distribution of mean :math:`0` and (estimated) parameter :math:`\frac{1}{\lambda}`.

Then, commuters choose the mode that **minimizes their transport cost** (according to the properties of the Gumbel distribution):

.. math::

    min_m t_{mj}(x,c,w_{ic}) = -\frac{1}{\lambda}log(\sum_{m=1}^{M}exp[-\lambda\chi_i(\tau_m(x,c) + \delta_m(x,c)w_{ic})]) + \textcolor{orange}{\eta_{xcij}}

* :math:`\textcolor{orange}{\eta_{xcij}}` also follows a Gumbel minimum distribution of mean :math:`0` and scale parameter :math:`\frac{1}{\lambda}`.

In ``data.py``, this is imported with:

.. literalinclude:: ../inputs/data.py
   :language: python
   :lines: 1475-1478
   :lineno-start: 1475

Given their residential location :math:`x`, workers choose the workplace location :math:`c` that **maximizes their income net of commuting costs**:

.. math::

	max_c [y_{ic} - min_m t_{mj}(x,c,w_{ic})]

* :math:`y_{ic} = \chi_i w_{ic}` (``income_centers_init`` parameter)

This yields the **probability to choose to work** in location :math:`c` given residential location :math:`x`and income group :math:`i` (logit discrete choice model):

.. math::

    \pi_{c|ix} = \frac{exp[\lambda y_{ic} + log(\sum_{m=1}^{M}exp[-\lambda\chi_i(\tau_m(x,c) + \delta_m(x,c)w_{ic})])]}{\sum_{k=1}^{C} exp[\lambda y_{ik} + log(\sum_{m=1}^{M}exp[-\lambda\chi_i(\tau_m(x,k) + \delta_m(x,k)w_{ik})])]}

In ``data.py``, this corresponds to:

.. literalinclude:: ../inputs/data.py
   :language: python
   :lines: 1494-1500
   :lineno-start: 1494

From there, we calculate the **expected income net of commuting costs** for residents of group :math:`i` living in location :math:`x`:

.. math::

    \tilde{y}_i(x) = E[y_{ic} - min_m(t_m(x,c,w_{ic}))|x]

In ``data.py``, this corresponds to:

.. literalinclude:: ../inputs/data.py
   :language: python
   :lines: 1505-1510
   :lineno-start: 1505