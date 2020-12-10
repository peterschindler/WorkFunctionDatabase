# Work Function Database and ML Model

This repository contains the work function database (calculated by high-throughput density functional theory) and machine learning model to predict the work function of a surface, accompanying the preprint: 

[*Discovery of materials with extreme work functions by high-throughput density functional theory and machine learning*,
Peter Schindler, Evan R. Antoniuk, Gowoon Cheon, Yanbing Zhu, and Evan J. Reed, **arXiv:2011.10905**](https://arxiv.org/abs/2011.10905)

## Database

The database csv files can easily be loaded using `pandas.read_csv`. The columns contain the following information:
- `mpid`: Materials Project ID
- `miller`: Miller index of surface hkl
- `term`: Termination number indexing (unique terminations are numbered starting at 1)
- `surface_elements`: List of atomic numbers present in the topmost atomic layer
- `surface_elements_string`: Same information as `surface_elements` but as a concatenated string of elements
- `WF`: The DFT-calculated work function in eV
- `slab`: The surface slab as a Pymatgen dictionary 
- `energy`: Total energy of slab in eV 
- `Fermi`: Fermi level in eV
- `convergence`: DFT convergence parameters, E_cutoff - kx - ky - kz
- `nsites`: Number of atomic sites in the slab unit cell
- `slab_thickness`: Slab thickness in Angstroms
- `nterm`: Number of unique terminations for given orientation/material
- `mirror`: Whether or not there exists a mirror plane parallel to the surface

## ML Model

The folder `ML_model` contains code to generate surfaces from bulk input structures, to featurize them, and to make work function predictions.

### Requirements

The code has been tested with Python 3.6.6 and the following packages/versions: `ase==3.20.1`,`pymatgen==2020.6.8`,`scikit-learn==0.23.1`,`pandas==1.0.1`,`joblib==0.16.0` but may run with newer versions as well.

### How to use

To generate surfaces from the bulk and predict their work functions follow these steps:

1. Create `input.csv` file for your bulk input structure with two columns: 
	1. `material_id`: The Materials Project ID or any other unique label for the bulk input structure
	2. `cifs.conventional_standard`: The conventional unit cell cif information (e.g. as pulled with the same column name tag using `MPRester`)
2. Run `slab_generator.py` which creates all unique surfaces/terminations up to a Miller index of 1 (then stored in file `slabs-input.csv`).
3. Run `featurization_and_WF_predict.py` to featurize the above created surfaces and predict the work function. The predicted work functions are stored in file `slabs-input_predicted_WFs_0p3.csv` under column `WF_predicted`.


