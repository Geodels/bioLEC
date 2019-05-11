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

dededede
