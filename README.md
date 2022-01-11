# lancement du code
- Le script principal est ```main.py```
- Bien mettre le code dans un dossier qui contient aussi les donnees. L'arborescence doit ressembler à ça:

```shell
.
├── 2. Data
│   ├── 0. Precalculated inputs
│   ├── Flood plains - from Claus
│   ├── Land occupation
│   ├── data_Cape_Town
│   ├── housing_types_grid_sal.xlsx
│   └── precalculated_transport
├── 4. Sorties
└── code capetown python
    ├── LICENSE
    ├── README.md
    ├── calibration
    ├── equilibrium
    ├── inputs
    ├── main.py
    ├── outputs
    ├── requirements.txt
    └── untitled.py
  
    
```

```param["pockets"]``` and ```param["backyard_pockets"]``` are amenities which depend on each settlement (e.g. because probability of eviction is different)

# Data
SP are census zones. When variables have a name with end with "SP" there are defined by such zones. E.g. ```housing_types_sp, data_sp```.

