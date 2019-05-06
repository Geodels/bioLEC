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

#!/usr/bin/python
# -*- coding: utf-8 -*-

import gc
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
from skimage import graph

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

class landscapeConnectivity(object):
    """
    Class for building landscape elevational connectivity.

    Local species richness is found to be related to the landscape elevational connectivity, as quantified
    by the landscape elevational connectivity metric (LEC) that applies tools of complex network theory to
    measure the closeness of a site to others with similar habitat.

    Algorithm
    ---------
    E. Bertuzzo et al., 2016: Geomorphic controls on elevational gradients of species richness - PNAS
    ww.pnas.org/cgi/doi/10.1073/pnas.1518922113
    Bertuzzo, E. and Carrara, F. and Mari, L. and Altermatt, F. and Rodriguez-Iturbe, I. and Rinaldo, A. (2016)
    Geomorphic controls on elevational gradients of species richness
    Proceedings of the National Academy of Sciences 113(7), 1737-1742
    doi:10.1073/pnas.1518922113


    Parameters
    ----------
    filename    : (string) csv file name containing regularly spaced elevation grid
    periodic    : (bool default: False) applied periodic boundary to the elevation grid
    symmetric   : (bool default: False) applied symmetric boundary to the elevation grid
    sigmap      : (float default: 0.1) species niche width percentage  based on elevation extent
    sigmav      : (float default: None) species niche fixed width values
    connected   : (bool default: True) computes the path based on the diagonal moves as well as the axial ones
    delimiter   : (string default: r'\s+') elevation grid csv delimiter
    header      : (int or list of ints) row number(s) to use as the column names, and the start of the data

    Notes
    -----
    Landscape elevational connectivity (LEC) quantifies the closeness of a site to all others with
    similar elevation. Such closeness is computed over a graph whose edges represent connections
    among sites and whose weights are proportional to the cost of spreading through patches at
    different elevation.

    Although LEC simply depends on the elevation field and on the niche width, LEC predicts well the
    alpha-diversity simulated by full metacommunity models. LEC is able to capture the variability
    of diversity hosted at the same elevation, as opposed to a simpler predictor like the elevation
    frequency.
    """

    def __init__(self, filename=None, periodic=False, symmetric=False, sigmap=0.1, sigmav=None,
                        connected=True, delimiter=r'\s+', header=None):

        # Read DEM file
        self.demfile = filename
        df = pd.read_csv(self.demfile, sep=delimiter, engine='c',
                               header=header, na_filter=False, dtype=np.float,
                               low_memory=False)
        X = df['X']
        Y = df['Y']
        Z = df['Z']
        dx = ( X[1] - X[0] )
        nx = int((X.max() - X.min())/dx+1)
        ny = int((Y.max() - Y.min())/dx+1)
        Z = np.reshape(Z,(ny,nx))
        del X
        gc.collect()
        del Y
        gc.collect()

        self.connected = connected
        if periodic:
            periodicZ = np.zeros((Z.shape[0]*3,Z.shape[1]*3))
            kr0 = 0
            for r in range(3):
                kr1 = kr0 + Z.shape[0]
                kc0 = 0
                for c in range(3):
                    kc1 = kc0 + Z.shape[1]
                    periodicZ[kr0:kr1,kc0:kc1] = Z
                    kc0 = kc1
                kr0 = kr1
            self.data =  periodicZ
            del periodicZ
            gc.collect()
            self.nr = self.data.shape[0]
            self.nc = self.data.shape[1]
            self.nr0 = int(self.nr/3.)
            self.nr1 = self.nr0 + int(self.nr/3.)
            self.nc0 = int(self.nc/3.)
            self.nc1 = self.nc0 + int(self.nc/3.)
        elif symmetric:
            nZ = []
            kr0 = 0
            symmetricZ = np.zeros((Z.shape[0]*3,Z.shape[1]*3))
            nZ.append(Z[::-1,::-1])
            nZ.append(np.flipud(Z))
            nZ.append(Z[::-1,::-1])
            nZ.append(np.fliplr(Z))
            nZ.append(Z)
            nZ.append(np.fliplr(Z))
            nZ.append(Z[::-1,::-1])
            nZ.append(np.flipud(Z))
            nZ.append(Z[::-1,::-1])
            k=0
            for r in range(3):
                kr1 = kr0 + Z.shape[0]
                kc0 = 0
                for c in range(3):
                    kc1 = kc0 + Z.shape[1]
                    symmetricZ[kr0:kr1,kc0:kc1] = nZ[k]
                    kc0 = kc1
                    k+=1
                kr0 = kr1
            self.data =  symmetricZ
            del symmetricZ
            gc.collect()
            self.nr = self.data.shape[0]
            self.nc = self.data.shape[1]
            self.nr0 = int(self.nr/3.)
            self.nr1 = self.nr0 + int(self.nr/3.)
            self.nc0 = int(self.nc/3.)
            self.nc1 = self.nc0 + int(self.nc/3.)
        else:
            self.data = Z
            self.nr = self.data.shape[0]
            self.nc = self.data.shape[1]
            self.nr0 = 0
            self.nr1 = self.nr
            self.nc0 = 0
            self.nc1 = self.nc
        del Z
        gc.collect()

        if sigmap is not None:
            sigma = (self.data.max() - self.data.min()) * sigmap
        elif sigmav is not None:
            sigma = sigmav
        else:
            print('Species niche width is not specified!')
            sigma = (self.data.max() - self.data.min()) * sigmap

        self.sigma2 = 2. * np.square(sigma)
        self.nNodes = self.nc * self.nr

        self.LEC = None

        # Set MPI communications
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        return

    def _computeMinPath(self, r, c):
        """
        This function computes the minimum path between of specific nodes and all other nodes
        in the array.

        Parameters
        ----------
        c : (int) row indices of a consider point
        r : (int) column indices of a consider point

        Returns
        -------
        The minimum-cost path to the specified ending indices from the specified starting indices.

        Notes
        -----
        This function relies on scikit-image (image processing in python) and finds
        distance-weighted minimum cost paths through an n-d costs array.

        The calculation is based on MCP_Geometric and differs from MCP in that the cost
        of a path is not simply the sum of the costs along that path.

        This class instead assumes that the costs array contains at each position the “cost”
        of a unit distance of travel through that position. For example, a move (in 2-d) from (1, 1)
        to (1, 2) is assumed to originate in the center of the pixel (1, 1) and terminate in the
        center of (1, 2). The entire move is of distance 1, half through (1, 1) and half
        through (1, 2); thus the cost of that move is (1/2)*costs[1,1] + (1/2)*costs[1,2].

        On the other hand, a move from (1, 1) to (2, 2) is along the diagonal and is sqrt(2)
        in length. Half of this move is within the pixel (1, 1) and the other half in (2, 2),
        so the cost of this move is calculated as (sqrt(2)/2)*costs[1,1] + (sqrt(2)/2)*costs[2,2].

        """
        # Create the cost surface based on the square of the difference in elevation between the considered
        # node and all the others vertices
        weight = np.square( self.data - self.data[r, c] )

        # From the weight-surface we create a 'landscape graph' object which can then be
        # analysed using distance-weighted minimum cost path
        cost = graph.MCP_Geometric( weight, fully_connected=self.connected )

        # Calculate the least-cost distance from the start cell to all other cells
        return cost.find_costs(starts=[(r, c)])[0]

    def _splitRowWise(self):
        """
        From the number of processors available split the array in equal number row wise.
        This simple partitioning is used to perform LEC computation in parallel.

        Returns
        -------
        disps   : 1D array of integers, shape (np, ) where np is the number of processors
            number of nodes to consider on each partition
        startID : 1D array of integers, shape (np, ) where np is the number of processors
            starting index for each partition
        endID   : 1D array of integers, shape (np, ) where np is the number of processors
            ending index for each partition
        """

        tmpA = np.zeros((self.nr, self.nc))
        split = np.array_split(tmpA, self.size, axis = 0)
        split_sizes = []
        for i in range(0,len(split),1):
            split_sizes = np.append(split_sizes, len(split[i]))
        split_sizes_input = split_sizes*self.nr
        endID = np.cumsum(split_sizes)
        startID = np.zeros(self.size,int)
        startID[1:] = np.asarray(endID[0:-1])
        disps = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]
        del split_sizes_input
        gc.collect()
        del split_sizes
        gc.collect()
        del split
        gc.collect()
        del tmpA
        gc.collect()

        return disps, startID, endID.astype(int)

    def computeLEC(self, fout=500):
        """
        This function computes the minimum path arrays for all nodes in the given dataset and
        the measure of the closeness of site j to i in terms of elevational connectivity. Then it
        builds the landscape elevational connectivity array from previously computed
        measure of closeness calculation.

        Parameters
        ----------
        fout : (integer default: 500) output frequency.

        """

        startID = None
        endID = None
        disps = None

        if self.rank == 0:
            disps, startID, endID  = self._splitRowWise()

        rsID = self.comm.bcast(startID, root=0)
        reID = self.comm.bcast(endID, root=0)

        displacements = self.comm.bcast(disps, root = 0)
        del disps
        gc.collect()
        t1 = time.clock()
        if self.rank < self.size-1:
            steps =   int(displacements[self.rank+1]) - int(displacements[self.rank])
        else:
            steps = self.nNodes - int(displacements[self.rank])
        print('RUN - rank ',self.rank,' start ID ',rsID[self.rank],' end ID ',reID[self.rank])
        del displacements
        gc.collect()

        lc = reID[self.rank]-rsID[self.rank]
        localLEC = np.zeros((self.nr,self.nc))

        k = 0
        for r in range(rsID[self.rank],reID[self.rank]):
            for c in range(0,self.nc):
                if k%fout==0 and k>0:
                    print('Rank: ',self.rank,'  - Compute Cij ',time.clock()-t1,' step ',k,' out of ',steps)
                    t1 = time.clock()
                localLEC += np.exp(-self._computeMinPath(r, c)/self.sigma2)
                k += 1

        del self.data
        gc.collect()
        del startID
        gc.collect()
        del endID
        gc.collect()

        flatLEC = np.reshape(localLEC, (np.product(localLEC.shape),))
        del localLEC
        gc.collect()

        if self.rank==0:
            valLEC = np.zeros_like(flatLEC)
        else:
            valLEC = None

        self.comm.Reduce(flatLEC, valLEC, op=MPI.SUM, root=0)
        del flatLEC
        gc.collect()

        if self.rank == 0:
            self.LEC =  np.reshape(valLEC,(self.nr, self.nc))

        del valLEC
        gc.collect()

        return

    def writeLEC(self, filename='LECout.csv'):
        """
        This function writes the computed landscape elevational connectivity array to a file.

        Parameters
        ----------
        filename : (string) output file name.
        """

        if self.rank == 0:
            df = pd.DataFrame({'LEC':self.LEC[self.nr0:self.nr1,self.nc0:self.nc1].flatten()})
            df.to_csv(filename, columns=['LEC'], sep=',', index=False , header=1)

        del self.LEC
        gc.collect()

        self.comm.Barrier()

        return
