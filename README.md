# README

_Author: Michał Denkiewicz (michal.denkiewicz@pw.edu.pl)_

This repository is a supplement to the [PAPER INFO HERE] manuscript.

## Analysis code

The Following notebooks contain the code for data analysis of the raw cKNOTs results.
- `data_consolidation.ipynb` - Noteboook used to create intermediate files before the analysis.
- `analysis.ipynb` - Contains the statistical analyses used in the manuscript.
- `graph_visualization.ipynb` - A Jupyter notebook that alolows to visualize a single CCD as a 
graph and a 3D model.
- `batch_models_analysis.ipynb` - Analyses the linking in batches of 3D models.

_Usage_: Simply exectue all cells in the notebook.

## Utilities

- `datasources.py` - Utility to load data from multiple files with a particular naming pattern into a Pandas DataFrame.
- `knots_tools.py` - Code closely related to operating on cKONTs data.
- `modvis.py` - Package gathers visualization code, as well as provides easy way to launch 3D modeling a obtain results.
- `read_chiadrop.py` - Utilities for reading  ChIA-Drop data described in the manuscript.
- `batch_models.py` - Standalone python program that runs batches of 3D modeling of a single CCD.
- `show_pyvista_model.py` - Utility to show a 3D model in pyvista outside jupyter.

## Bash scripts

The bash scripts realte to running of Spring Model modeling software.
