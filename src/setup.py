from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension
from os import path
import io

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(name = 'bioLEC',
          author            = "Tristan Salles",
          author_email      = "tristan.salles@sydney.edu.au",
          url               = "https://github.com/Geodels/bioLEC",
          version           = "1.0.0",
          description       = "A Python interface to compute biodiversity metric based on landscape elevational connectivity.",
          long_description  = long_description,
          long_description_content_type='text/markdown',
          packages          = ['bioLEC'],
          install_requires  = [
                        'numpy',
                        'scipy>=0.15.0',
                        'pandas',
                        'mpi4py',
                        'rasterio>=1.0.23',
                        'scikit-image'],
          python_requires   = '>=2.7, >=3.5',
          package_data      = {'bioLEC': ['Notebooks/notebooks/*ipynb',
                                          'Notebooks/notebooks/*py',
                                          'Notebooks/dataset/*',
                                          'Notebooks/images/*'] },
          include_package_data = True,
          classifiers       = ['Programming Language :: Python :: 2',
                               'Programming Language :: Python :: 2.6',
                               'Programming Language :: Python :: 2.7',
                               'Programming Language :: Python :: 3',
                               'Programming Language :: Python :: 3.3',
                               'Programming Language :: Python :: 3.4',
                               'Programming Language :: Python :: 3.5',
                               'Programming Language :: Python :: 3.6']
          )
