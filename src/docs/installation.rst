Installation
============

A more straightforward installation which does not depend on specific compilers relies on the docker virtualisation system.

To install the docker image and test it is working:

.. code-block:: bash

  $ docker pull geodels/biolec:latest
  $ docker run --rm geodels/biolec:latest help

To build the dockerfile locally, we provide a script. First ensure you have checked out the source code from github and then run the script in the Docker directory. If you modify the dockerfile and want to push the image to make it publicly available, it will need to be retagged to upload somewhere other than the GEodels repository.

.. code-block:: bash

  $ git checkout https://github.com/Geodels/bioLEC.git
  $ cd bioLEC
  $ source Docker/build-dockerfile.sh


Dependencies
------------


You will need **Python 2.7 or 3.5+**.

Also, the following packages are required:

+ [`numpy`](http://numpy.org)
+ [`scipy`](https://scipy.org)
+ [`pandas`](https://pandas.pydata.org/)
+ [`mpi4py`](https://pypi.org/project/mpi4py/)
+ [`scikit-image`](https://scikit-image.org/)


Installing using pip
--------------------

|PyPI version shields.io|

.. |PyPI version shields.io| image:: https://img.shields.io/pypi/v/bioLEC.svg
   :target: https://pypi.org/project/bioLEC/

You can install `bioLEC` using the
[`pip package manager`](https://pypi.org/project/pip/) with either version of Python:


.. code-block:: bash

  $ python2 -m pip install bioLEC
  $ python3 -m pip install bioLEC


Installing using Docker
-----------------------

A more straightforward installation which does not depend on specific compilers relies on the [docker](http://www.docker.com) virtualisation system.

To install the docker image and test it is working:

.. code-block:: bash

  $ docker pull geodels/biolec:latest
  $ docker run --rm geodels/biolec:latest help

To build the dockerfile locally, we provide a script. First ensure you have checked out the source code from github and then run the script in the Docker directory. If you modify the dockerfile and want to push the image to make it publicly available, it will need to be retagged to upload somewhere other than the GEodels repository.

.. code-block:: bash

  $ git checkout https://github.com/Geodels/bioLEC.git
  $ cd bioLEC
  $ source Docker/build-dockerfile.sh
