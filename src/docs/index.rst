.. bioLEC documentation master file, created by
   sphinx-quickstart on Fri May 10 16:22:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

bioLEC - *Landscape elevational connectivity*
=============================================

.. image:: https://readthedocs.org/projects/biolec/badge/?version=latest
  :target: https://biolec.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

**bioLEC** is a parallel python package built to calculate the *Landscape elevational connectivity* (**LEC**).

.. image:: ../bioLEC/Notebooks/images/fig1.png
   :scale: 30 %
   :alt: LEC computation
   :align: center

.. note::
  **LEC** quantifies the closeness of a site to all others with **similar elevation**. It measures how easily a **species living in a given patch can spread and colonise other patches**. It is assumed to be **elevation-dependent** and the metric depends on how often a species adapted to a given elevation *needs to travel outside its optimal elevation range* when moving from its patch to any other in the landscape.

Contents
--------

.. toctree::
   method
   installation
   usage
   social
   bioLEC
   :maxdepth: 3

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
