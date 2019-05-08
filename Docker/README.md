# bioLEC - Biodiversity metric based on landscape elevational connectivity


[![PyPI](https://img.shields.io/pypi/v/bioLEC.svg)](https://pypi.org/project/bioLEC/)

This folder contains notebooks to compute **landscape elevational connectivity** described in Bertuzzo et al. (2016) using a parallel LECmetrics python code.

***

![LEC computation](https://github.com/Geodels/bioLEC/blob/master/Notebooks/images/fig1.jpg?raw=true)

***

Notebooks environment will not be the best option for large landscape models and we will recommend the use of the python script: `runLEC.py` in HPC environment. the code will need to be

```bash
mpirun -np 400 python runLEC.py
```

The tool can be used to compute the LEC for any landscape file (X,Y,Z) and IPython functions are provided to extract output data directly from pyBadlands model.

***

![LEC computation](https://github.com/Geodels/bioLEC/blob/master/Notebooks/images/fig3.jpg?raw=true)

***

E. Bertuzzo, F. Carrara, L. Mari, F. Altermatt, I. Rodriguez-Iturbe & A. Rinaldo - Geomorphic controls on species richness. **PNAS**, 113(7) 1737-1742, [DOI: 10.1073/pnas.1518922113](http://www.pnas.org/content/113/7/1737), 2016.
