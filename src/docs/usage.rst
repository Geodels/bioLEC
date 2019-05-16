Quick start guide
=================

Input

A simulation requires sets of times, frequencies, source positions and brightnesses, antenna positions, and direction-dependent primary beam responses. pyuvsim specifies times, frequencies, and array configuration via a UVData object (from the pyuvdata package), source positions and brightnesses via Source objects, and primary beams either through UVBeam or AnalyticBeam objects.

All sources are treated as point sources, with flux specified in Stokes parameters and position in right ascension / declination in the International Celestial Reference Frame (equivalently, in J2000 epoch).
Primary beams are specified as full electric field components, and are interpolated in angle and frequency. This allows for an exact Jones matrix to be constructed for each desired source position.
Multiple beam models may be used throughout the array, allowing for more complex instrument responses to be modeled.
These input objects may be made from a data file or from a set of yaml configuration files. See Running a simulation.

Outputs

Data from a simulation run are written out to a file in any format accessible with pyuvdata. This includes:
When read into a UVData object, the history string will contain information on the pyuvsim and pyuvdata versions used for that run (including the latest git hash, if available), and details on the catalog used.

Binder
------

.. image:: https://mybinder.org/badge_logo.svg
  :target: https://mybinder.org/v2/gh/Geodels/bioLEC/binder?filepath=Notebooks


Notebooks
---------

An example is provided...

HPC
---

The tool can be used to compute the **LEC** for any landscape file as long as the data is available from a **CSV file containing 3D coordinates (X,Y,Z)**.

.. attention::
  Notebooks environment will not be the best option for **large landscape models** and we will recommend the use of the python script: ``runLEC.py`` in **HPC environment**.

In this case, the code will need to be


.. code-block:: bash

  $ mpirun -np XX python runLEC.py

with XX represents the number of processor to use.

The python script ``runLEC.py`` is defined by:

.. code-block:: python

  import argparse
  from bioLEC import LEC

  # Parsing command line arguments
  parser = argparse.ArgumentParser(description='This is a simple entry to run bioLEC package from python.',add_help=True)

  # Required
  parser.add_argument('-i','--input', help='Input file name (csv file)',required=True)
  parser.add_argument('-o','--output',help='Output file name without extension', required=True)

  # Optional
  parser.add_argument('-p','--periodic',help='True/false option for periodic boundary conditions', required=False, action="store_true", default=False)
  parser.add_argument('-s','--symmetric',help='True/false option for symmetric boundary conditions', required=False, action="store_true", default=False)
  parser.add_argument('-w','--width',help='Float option for species niche width percentage', required=False, action="store_true", default=0.1)
  parser.add_argument('-f','--fix',help='Float option for species niche width fix values', required=False, action="store_true", default=None)
  parser.add_argument('-c','--connected',help='True/false option for computing the path based on the diagonal moves as well as the axial ones', required=False, action="store_true", default=True)
  parser.add_argument('-t','--top',help='Header lines in the elevation grid', required=False, action="store_true", default=0)
  parser.add_argument('-n','--nout',help='Number for output frequency during run', required=False, action="store_true", default=500)
  parser.add_argument('-d','--delimiter',help='String for elevation grid csv delimiter', required=False,action="store_true",default=',')
  parser.add_argument('-v','--verbose',help='True/false option for verbose', required=False,action="store_true",default=False)


  args = parser.parse_args()
  if args.verbose:
    print("Required arguments: ")
    print("   + Input file: {}".format(args.input))
    print("   + Output file without extension: {}".format(args.output))
    print("\nOptional arguments: ")
    print("   + Periodic boundary conditions for the elevation grid: {}".format(args.periodic))
    print("   + Symmetric boundary conditions for the elevation grid: {}".format(args.symmetric))
    print("   + Species niche width percentage based on elevation extent: {}".format(args.width))
    print("   + Species niche width based on elevation extent: {}".format(args.fix))
    print("   + Computes the path based on the diagonal moves as well as the axial ones: {}".format(args.connected))
    print("   + Elevation grid csv delimiter: {}".format(args.delimiter))
    print("   + Number of header lines: {}".format(args.top))
    print("   + Number for output frequency: {}\n".format(args.nout))

  biodiv = LEC.landscapeConnectivity(filename=args.input,periodic=args.periodic,symmetric=args.symmetric,
                                      sigmap=args.width,sigmav=args.fix,connected=args.connected,
                                      delimiter=args.delimiter,header=args.top)

  biodiv.computeLEC(args.nout)

  biodiv.writeLEC(args.output)
  if biodiv.rank == 0:
      biodiv.viewResult(imName=args.output+'.png')
      biodiv.viewElevFrequency(input=args.output+'.csv',imName=args.output+'_zfreq.png')
      biodiv.viewLECZFrequency(input=args.output+'.csv',imName=args.output+'_leczfreq.png')
      biodiv.viewLECFrequency(input=args.output+'.csv',imName=args.output+'_lecfreq.png')
      biodiv.viewLECZbar(input=args.output+'.csv',imName=args.output+'_lecbar.png')
