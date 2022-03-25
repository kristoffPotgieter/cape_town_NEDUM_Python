# RUN CODE

- Main script is ```main.py``` : it computes equilibrium allocation and runs dynamic simulations over several years.
- ```plots.py``` plots various outputs from the model. Both scripts are cleaned and run correctly but the latter is not commented yet.
- Other files will be updated with the documentation.
- To run properly, the code should be put in a folder that also contains the appropriate data, according to the following tree structure (to be updated).

```shell
.
├── 2. Data
│   ├── 0. Precalculated inputs
│   ├── Aux data
│   ├── Coastal
│   ├── data_Cape_Town
│   ├── FATHOM
│   ├── Flood plains - from Claus
│   ├── Land occupation
│   ├── precalculated_transport
│   ├── CT Dwelling type data validation workbook 20201204 v2.xlsx
│   └── housing_types_grid_sal.xlsx
├── 4. Sorties
└── code capetown python
    ├── _docs
    ├── _old
    ├── _tests
    ├── calibration
    ├── equilibrium
    ├── inputs
    ├── outputs
    ├── LICENSE
    ├── main.py
    ├── manage.py
    ├── plots.py
    ├── README.md
    ├── requirements.txt
    └── setup.py
  
    
```

- In order to integrate user-written packages seamlessly,  folder should be set as a project in Spyder (or any other Python IDE).
- For reference, ```inputs``` defines functions for treating and loading data, ```equilibrium``` does so for computing static equilibrium and dynamic simulations, ```outputs``` for plotting and exporting results, and ```calibration``` for calibrating parameters. Note that, in its current version, the code runs on pre-calibrated data for the sake of speed. A side script will be written to run the calibration again if needed.
- A more extensive user guide is in the process of writing, with more detailed interpretation of the code and use cases for basic handling, such as changing options and parameters, updating data and scenarios, etc. For now, the code is being reorganized and commented. It will later be optimized and equipped with a proper API reference.


# VERSIONING

- The ```main``` branch contains the original code (with some extra features) from the WB working paper (adapted from Matlab to Python): it should run properly
- The ```TLM-edits``` branch contains some code and folder reorganization without rewriting anything: it should run properly
- The ```TLM-write``` branch is the main working branch. It contains some rewriting and commenting: code may not run fully due to ongoing work.
- All other branches are for internal use only