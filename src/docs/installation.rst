Installation
============

**bioLEC** is a pure Python package and can be installed via `pip` as well as **Docker**.

Dependencies
------------

You will need **Python 2.7 or 3.5+** and the following packages are required:
`numpy <http://numpy.org>`_, `scipy <https://scipy.org>`_, `pandas <https://pandas.pydata.org/>`_, `mpi4py <https://pypi.org/project/mpi4py/>`_, `scikit-image <https://scikit-image.org/>`_, `pyevtk <https://pypi.org/project/pyevtk/>`_.

.. note::
  The `mpi4py` module has to be installed as a wrapper around an existing installation of **MPI**. The easiest way to install it is to use *anaconda*, which will install a compatible version of MPI and configure mpi4py to use it:
  :code:`conda install -c conda-forge mpi4py`

The complete list of Dependencies is available in the **src/requirements.txt** file and looks like:

.. code-block:: bash

  numpy>=1.15
  six>=1.11.0
  setuptools>=38.4.0
  scipy>=1.0.1
  mpi4py>=3.0.0
  pandas>=0.24
  scipy>=1.2
  scikit-image>=0.15
  pyevtk>=1.1.1
  matplotlib>=3.0


Optionally, `lavavu <https://github.com/OKaluza/LavaVu>`_ is needed to support *interactive visualisation in Jupyter environment* (as shown in the notebook examples available from `binder <https://mybinder.org/v2/gh/Geodels/bioLEC/binder?filepath=Notebooks>`_).

Installing using pip
--------------------

|PyPI version shields.io|

.. |PyPI version shields.io| image:: https://img.shields.io/pypi/v/bioLEC.svg
   :target: https://pypi.org/project/bioLEC/

You can install **bioLEC** package using the latest stable release available via `pip <https://pypi.org/project/bioLEC/>`_!

:code:`pip install bioLEC`

If you need to install it for different python versions:

.. code-block:: bash

  $ python2 -m pip install bioLEC
  $ python3 -m pip install bioLEC

To install the *development version*: **Clone the repository using**

:code:`git clone https://github.com/Geodels/bioLEC.git`

Navigate into the bioLEC directory and run

.. code-block:: bash

  $ cd src/
  $ pip install -r requirements.txt
  $ pip install bioLEC
  $ pip install .


Installing using Docker
-----------------------

Another straightforward installation which does not depend on specific compilers relies on `docker <http://www.docker.com>`_ virtualisation system.

To install the docker image and test it is working:

.. code-block:: bash

  $ docker pull geodels/biolec:latest
  $ docker run --rm geodels/biolec:latest help

On Linux, to build the dockerfile locally, we provide a script. First ensure you have checked out the source code from github and then run the script in the Docker directory. If you modify the dockerfile and want to push the image to make it publicly available, it will need to be retagged to upload somewhere other than the GEodels repository.

.. code-block:: bash

  $ git checkout https://github.com/Geodels/bioLEC.git
  $ cd bioLEC
  $ source Docker/build-dockerfile.sh

.. note::
  For non-Linux platforms, the use of `Docker Desktop for Mac`_ or `Docker Desktop for Windows`_ is recommended. The docker container to look for is named **geodels/biolec**!

.. _`Docker Desktop for Mac`: https://docs.docker.com/docker-for-mac/
.. _`Docker Desktop for Windows`: https://docs.docker.com/docker-for-windows/


Testing installation
--------------------

A test is provided to check the correct installation of the **bioLEC** package.If you've cloned the source into a directory :code:`bioLEC`, you may verify it as follows:

Run the tests from bioLEC...

.. code-block:: bash

  $ python2 src/tests/testInstall.py
  $ python3 src/tests/testInstall.py

You will need to have all dependencies installed.
