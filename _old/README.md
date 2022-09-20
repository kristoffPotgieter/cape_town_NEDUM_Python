# RUN CODE

- Main script is ```main.py``` : it computes equilibrium allocation and runs dynamic simulations over several years.
- ```plots.py``` plots various outputs from the model. Both scripts are cleaned and run correctly but the latter is not commented yet.
- Main script runs on pre-calibrated data. We need to run ```precalibration_main.py``` and ```precalibration_disamenity.py``` to recalibrate the parameters. Those scripts are not fully functional yet.
- ```validation.py``` is a notepad containing some validation exercises (ongoing).
- Other files will be updated with the documentation.
- To run properly, the code should be put in a folder that also contains the appropriate data, according to the following tree structure (to be updated).

```shell
.
├── Data
│   ├── Precalculated inputs
│   ├── Aux data
│   ├── data_Cape_Town
│   ├── Flood plains - from Claus
│   ├── Land occupation
│   ├── precalculated_transport
│   ├── CT Dwelling type data validation workbook 20201204 v2.xlsx
│   └── housing_types_grid_sal.xlsx
├── Output
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
    ├── precalibration_disamenity.py
    ├── precalibration_main.py
    ├── README.md
    ├── requirements.txt
    ├── setup.py   
    └── validation.py
  
    
```

- In order to integrate user-written packages seamlessly, folder should be set as a project in Spyder (or any other Python IDE).
- For reference, ```inputs``` defines functions for treating and loading data, ```equilibrium``` does so for computing static equilibrium and dynamic simulations, ```outputs``` for plotting and exporting results, and ```calibration``` for calibrating parameters.
- A more extensive user guide is in the process of writing, with more detailed interpretation of the code and use cases for basic handling, such as changing options and parameters, updating data and scenarios, etc. For now, the code is being reorganized and commented. It will later be optimized and equipped with a proper API reference.


# VERSIONING

- The ```main``` branch contains the original code (with some extra features) from the WB working paper (adapted from Matlab to Python)
- The ```TLM-edits``` branch contains some code and folder reorganization without rewriting anything
- The ```TLM-write``` branch is the main working branch. It contains some rewriting and commenting: code may not run fully due to ongoing work.
- All other branches are for internal use only