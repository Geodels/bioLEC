# bioLEC - Biodiversity metric based on landscape elevational connectivity

This folder contains notebooks to compute **landscape elevational connectivity** described in Bertuzzo et al. (2016) using a parallel LECmetrics python code.

***

<div align="center">
    <img width=1000 src="https://github.com/Geodels/bioLEC/blob/master/Notebooks/images/fig1.png" alt="sketch LEC computation" title="LEC computation."</img>
</div>


***

Notebooks environment will not be the best option for large landscape models and we will recommand the use of the python script: `runLEC.py` in HPC environment. the code will need to be

```bash
mpirun -np 400 python runLEC.py
```

The tool can be used to compute the LEC for any landscape file (X,Y,Z) and IPython functions are provided to extract output data directly from pyBadlands model.

***

<div align="center">
    <img width=1000 src="https://github.com/Geodels/bioLEC/blob/master/Notebooks/images/fig3.png" alt="sketch Badlands" title="LEC computation."</img>
</div>


***

E. Bertuzzo, F. Carrara, L. Mari, F. Altermatt, I. Rodriguez-Iturbe & A. Rinaldo - Geomorphic controls on species richness. **PNAS**, 113(7) 1737-1742, [DOI: 10.1073/pnas.1518922113](http://www.pnas.org/content/113/7/1737), 2016.
