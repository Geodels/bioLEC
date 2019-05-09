# bioLEC - Biodiversity metric based on landscape elevational connectivity

[![Docker Cloud Automated build](https://img.shields.io/docker/cloud/automated/geodels/biolec.svg)](https://hub.docker.com/r/geodels/biolec)
[![PyPI](https://img.shields.io/pypi/v/bioLEC.svg)](https://pypi.org/project/bioLEC/)

This folder contains notebooks to compute **landscape elevational connectivity** described in Bertuzzo et al. (2016) using a parallel LECmetrics python code.

#### Binder

Launch the demonstration at [mybinder.org](https://mybinder.org/v2/gh/Geodels/bioLEC/binder?filepath=Notebooks)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Geodels/bioLEC/binder?filepath=Notebooks)


## Navigation / Notebooks

### Examples

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

## Installation

### Dependencies

You will need **Python 2.7 or 3.5+**.
Also, the following packages are required:

 - [`numpy`](http://numpy.org)
 - [`scipy`](https://scipy.org)
 - [`pandas`](https://pandas.pydata.org/)
 - [`mpi4py`](https://pypi.org/project/mpi4py/)
 - [`scikit-image`](https://scikit-image.org/)

### Installing using pip

You can install `bioLEC` using the
[`pip package manager`](https://pypi.org/project/pip/) with either version of Python:

```bash
python2 -m pip install bioLEC
python3 -m pip install bioLEC
```

### Installing using Docker

A more straightforward installation which does not depend on specific compilers relies on the [docker](http://www.docker.com) virtualisation system.

To install the docker image and test it is working:

```bash
   docker pull geodels/biolec:latest
   docker run --rm geodels/biolec:latest help
```

To build the dockerfile locally, we provide a script. First ensure you have checked out the source code from github and then run the script in the Docker directory. If you modify the dockerfile and want to push the image to make it publicly available, it will need to be retagged to upload somewhere other than the GEodels repository.

```bash
git checkout https://github.com/Geodels/bioLEC.git
cd bioLEC
source Docker/build-dockerfile.sh
```


## Usage

A series of tests are located in the *tests* subdirectory.


## References

  1. E. Bertuzzo, F. Carrara, L. Mari, F. Altermatt, I. Rodriguez-Iturbe & A. Rinaldo - Geomorphic controls on species richness. **PNAS**, 113(7) 1737-1742, [DOI: 10.1073/pnas.1518922113](http://www.pnas.org/content/113/7/1737), 2016.

  1. T.R. Etherington - Least-cost modelling and landscape ecology: concepts, applications, and opportunities. Current Landscape Ecology Reports 1:40-53, [DOI: 10.1007/s40823-016-0006-9](https://link.springer.com/article/10.1007%2Fs40823-016-0006-9), 2016.

  1. S. van der Walt , J.L. Schönberger, J. Nunez-Iglesias, F. Boulogne, J.D. Warner, N. Yager, E. Gouillart & T. Yu - Scikit Image Contributors - scikit-image: image processing in Python, [PeerJ 2:e453](https://peerj.com/articles/453/), 2014.

  1. T.R. Etherington - Least-cost modelling with Python using scikit-image, [Blog](http://tretherington.blogspot.com/2017/01/least-cost-modelling-with-python-using.html), 2017.
