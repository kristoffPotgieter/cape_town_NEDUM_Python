---
date: '2022-09-23T09:22:05.479Z'
docname: math_appendix
images: {}
path: /math-appendix
title: Math appendix
---

# Math appendix

## Commuting choice and net income calculation

The terms highlighted in orange relate to functional assumptions needed to solve the model. For each mode $m$, residential location $x$, job center $c$, income group $i$, and worker $j$, **expected commuting cost** is:

t_{mj}(x,c,w_{ic}) = \\chi_i(\\tau_m(x,c) + \\delta_m(x,c)w_{ic}) + \\textcolor{orange}{\\epsilon_{mxcij}}
* $w_{ic}$ is the (calibrated) wage earned.


* $\chi_i$ is the employment rate of the household, composed of two agents (`household_size` parameter).


* $\tau_m(x,c)$ is the monetary transport cost.


* $\delta_m(x,c)$ is a time opportunity cost parameter (the fraction of working time spent commuting).


* $\textcolor{orange}{\epsilon_{mxcij}}$ follows a Gumbel minimum distribution of mean $0$ and (estimated) parameter $\frac{1}{\lambda}$.

Then, commuters choose the mode that **minimizes their transport cost** (according to the properties of the Gumbel distribution):

min_m t_{mj}(x,c,w_{ic}) = -\\frac{1}{\\lambda}log(\\sum_{m=1}^{M}exp[-\\lambda\\chi_i(\\tau_m(x,c) + \\delta_m(x,c)w_{ic})]) + \\textcolor{orange}{\\eta_{xcij}}
* $\textcolor{orange}{\eta_{xcij}}$ also follows a Gumbel minimum distribution of mean $0$ and scale parameter $\frac{1}{\lambda}$.

In `data.py`, this is imported with:

```python
    # group (from calibration)
    if options["load_precal_param"] == 1:
        income_centers_init = scipy.io.loadmat(
            path_precalc_inp + 'incomeCentersKeep.mat')['incomeCentersKeep']
```

Given their residential location $x$, workers choose the workplace location $c$ that **maximizes their income net of commuting costs**:

max_c [y_{ic} - min_m t_{mj}(x,c,w_{ic})]
* $y_{ic} = \chi_i w_{ic}$ (`income_centers_init` parameter)

This yields the **probability to choose to work** in location $c$ given residential location $x$ (logit discrete choice model):

\\pi_{c|ix} = \\frac{exp[\\lambda y_{ic} + log(\\sum_{m=1}^{M}exp[-\\lambda\\chi_i(\\tau_m(x,c) + \\delta_m(x,c)w_{ic})])]}{\\sum_{k=1}^{C} exp[\\lambda y_{ik} + log(\\sum_{m=1}^{M}exp[-\\lambda\\chi_i(\\tau_m(x,k) + \\delta_m(x,k)w_{ik})])]}In `data.py`, this corresponds to:

```python
        householdSize = param["household_size"][j]
        # Here, -100000 corresponds to an arbitrary value given to incomes in
        # centers with too few jobs to have convergence in calibration (could
        # have been nan): we exclude those centers from the analysis
        whichCenters = incomeCenters[:, j] > -100000
        incomeCentersGroup = incomeCenters[whichCenters, j]

```

From there, we calculate the **expected income net of commuting costs** for residents of group $i$ living in location $x$:

\\tilde{y}_i(x) = E[y_{ic} - min_m(t_m(x,c,w_{ic}))|x]In `data.py`, this corresponds to:

```python
         ) = calcmp.compute_ODflows(
            householdSize, monetaryCost, costTime, incomeCentersGroup,
            whichCenters, param_lambda)

        # NB: we compute OD flows again later to get the full matrix

```
