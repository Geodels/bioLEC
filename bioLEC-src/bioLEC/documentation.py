"""
Copyright 2019 Tristan Salles

bioLEC is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

bioLEC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with bioLEC.  If not, see <http://www.gnu.org/licenses/>.
"""

import pkg_resources as _pkg_resources
from distutils import dir_util as _dir_util


def install_documentation(path="./bioLEC-Notebooks"):
    """Install the example notebooks for bioLEC in the given location
    WARNING: If the path exists, the Notebook files will be written into the path
    and will overwrite any existing files with which they collide. The default
    path ("./bioLEC-Notebooks") is chosen to make collision less likely / problematic
    The documentation for bioLEC is in the form of jupyter notebooks.
    Some dependencies exist for the notebooks to be useful:
       - matplotlib: for some of the diagrams
       - lavavu: interactive viewer for an alternative means of viewing cartesian_meshes
       - cartopy: for plotting map examples
    bioLEC dependencies are explicitly imported into the notebooks including:
       - gc
       - numpy
       - scipy (for k-d tree point location)
       - skimage
       - pandas
       - mpi4py
    """

    ## Question - overwrite or not ? shutils fails if directory exists.

    Notebooks_Path = _pkg_resources.resource_filename('bioLEC', 'Notebooks')

    ct = _dir_util.copy_tree(Notebooks_Path, path,preserve_mode=1, preserve_times=1, preserve_symlinks=1, update=0, verbose=1, dry_run=0)

    return
