#!/usr/bin/python
# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Tristan Salles
# Licensed under the GNU LGPL Version 3

import gc
import sys
import time
import numpy as np
import pandas as pd

# For readthedoc...
try:
    from mpi4py import MPI
except ImportError:
    print('mpi4py is required and needs to be installed via pip')
    pass

# For readthedoc...
try:
    from skimage import graph
except ImportError:
    print('scikit-image is required and needs to be installed via pip')
    pass

from pyevtk.hl import gridToVTK

from pylab import rcParams
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.axes_grid1 import make_axes_locatable

class landscapeConnectivity(object):
    """
    *Landscape elevational connectivity* (**LEC**) quantifies the closeness of a site to all others with
    similar elevation. Such closeness is computed over a *graph* whose edges represent connections
    among sites and whose weights are proportional to the cost of spreading through patches at
    different elevation.

    Method:
        E. Bertuzzo et al., 2016: Geomorphic controls on elevational gradients of species richness - PNAS
        doi_

    .. _doi: http://www.pnas.org/cgi/doi/10.1073/pnas.1518922113

    Args:
        filename (str): CSV file name containing regularly spaced elevation grid [default: None]
        XYZ (3D Numpy Array): 3D coordinates array of shape (nn,3) where nn is the number of points [default: None]
        Z (2D Numpy Array): Elevation array of shape (nx,ny) where nx and ny are the number of points  along the X and Y axis [default: None]
        dx (float): grid spacing in metre when the Z argument defined above is used [default: None]
        periodic (bool):  applied periodic boundary to the elevation grid [default: False]
        symmetric (bool): applied symmetric boundary to the elevation grid [default: False]
        sigmap (float): species niche width percentage  based on elevation extent [default: 0.1]
        sigmav (float): species niche fixed width values [default: None]
        connected (bool): computes the path based on the diagonal moves as well as the axial ones [default: True]
        delimiter (str):  elevation grid csv delimiter [default: ' ']
        sl (float):  sea level position used to remove marine points from the LEC calculation [default: -1.e6]

    caution:
        There are 3 ways to import the elevation dataset in bioLEC:

            * either as a CSV file (argument: filename) containing 3 columns for X, Y and Z respectively with no header and ordered along the X axis first
            * or as a 3D numpy array (argument: XYZ) containing the X, Y and Z coordinates here again ordered along the X axis first
            * or as a 2D numpy array (argument: Z) containing the elevation matrix, in this case the dx argument is also required

    Note:
        Although LEC simply depends on the elevation field and on the niche width, *LEC predicts well the
        alpha-diversity simulated by full metacommunity models*.
    """

    def __init__(self, filename=None, XYZ=None, Z=None, dx=None, periodic=False, symmetric=False,
                 sigmap=0.1, sigmav=None, connected=True, delimiter=' ', sl=-1.e6, test=False):

        # Set MPI communications
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.test = test

        if self.rank == 0 and not self.test:
            print('bioLEC - LANDSCAPE ELEVATIONAL CONNECTIVITY')
            print('-------------------------------------------\n')

        # Read DEM file
        if filename is not None:
            self.demfile = filename
            df = pd.read_csv(self.demfile, sep=delimiter, engine='c',
                                   header=None, na_filter=False, dtype=np.float,
                                   low_memory=False)
            self.X = df.values[:,0]
            self.Y = df.values[:,1]
            self.Z = df.values[:,2]

        elif XYZ is not None:
            if XYZ.shape[1] != 3 :
                if self.rank == 0:
                    print('Problem XYZ variable has to be defined with the following shape (nn,3) with nn the number of points!')
                return
            self.X = XYZ[:,0]
            self.Y = XYZ[:,1]
            self.Z = XYZ[:,2]
            if self.X[1]==self.X[0]:
                if self.rank == 0:
                    print('Problem XYZ variable have to be ordered along the X axis first!')
                return

        elif Z is not None:
            if dx is None:
                if self.rank == 0:
                    print('When using the Z argument then grid spacing [dx] has to be defined!')
                return

            nx = Z.shape[0]
            ny = Z.shape[1]
            xgrid = np.arange(0,nx*dx,dx)
            ygrid = np.arange(0,ny*dx,dx)
            xi, yi = np.meshgrid(xgrid,ygrid)
            self.X = xi.flatten()
            self.Y = yi.flatten()
            self.Z = Z.flatten()
        else:
            if self.rank == 0:
                print('Problem either filename or XYZ or Z variables have to be declared for the function to run properly!')
            return

        dx = ( self.X[1] - self.X[0] )
        if dx == 0:
            if self.rank == 0:
                print('Problem dx is set to 0!')
            return

        self.nx = int((self.X.max() - self.X.min())/dx+1)
        self.ny = int((self.Y.max() - self.Y.min())/dx+1)
        Z = self.Z.reshape(self.ny,self.nx)

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

        self.sealevel = sl
        minz = max(self.data.min(),self.sealevel)
        self.nz = self.data.copy()
        self.nz[self.nz<self.sealevel] = -1.e6

        if sigmap is not None:
            sigma = (self.data.max() - minz) * sigmap
        elif sigmav is not None:
            sigma = sigmav
        else:
            sigma = (self.data.max() - minz) * sigmap
            if self.rank == 0:
                print('WARNING: Species niche width is not specified!')
                print('A default width is defined based on elevational range: {:.3f}'.format(sigma))

        self.sigma2 = 2. * np.square(sigma)
        self.nNodes = self.nc * self.nr

        self.LEC = None

        return

    def computeMinPath(self, r, c):
        """
        **Internal function** to compute the minimum path between a specific node and all other ones.

        Args:
            c (int): row indice of the consider point
            r (int): column indice of a consider point

        Returns:
            mincost: minimum-cost path for the specified node.

        Note:
            This function relies on scikit-image_ (image processing in python) and finds
            distance-weighted minimum cost paths through an n-d costs array.

        .. _scikit-image: https://scikit-image.org/docs/dev/api/skimage.graph.html
        """
        # Create the cost surface based on the square of the difference in elevation between the considered
        # node and all the others vertices
        weight = np.square( self.nz - self.nz[r, c] )

        # From the weight-surface we create a 'landscape graph' object which can then be
        # analysed using distance-weighted minimum cost path
        cost = graph.MCP_Geometric( weight, fully_connected=self.connected )

        # Calculate the least-cost distance from the start cell to all other cells
        return cost.find_costs(starts=[(r, c)])[0]

    def splitRowWise(self):
        """
        **Rowwise block partitioning** used to perform LEC computation in parallel.

        Returns:
            disps (1D array int): number of nodes to consider on each partition
            startID (1D array int): starting index for each partition
            endID (1D array int): ending index for each partition

        Note:
            From the number of processors (np) available split the array into equal number row wise.
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

    def _test_progress(self, job_title, progress):
        length = 20
        block = int(round(length*progress))
        msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
        if progress >= 1: msg += " DONE\r\n"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def computeLEC(self, fout=500):
        """
        This function computes the **minimum path for all nodes** in a given surface and
        **measure of the closeness** of each node to other at similar elevation range.

        It then provide the *landscape elevational connectivity* array from computed
        measure of closeness calculation.

        Args:
            fout (int): output frequency [default: 500]

        """

        time0 = time.clock()
        startID = None
        endID = None
        disps = None

        if self.rank == 0:
            disps, startID, endID  = self.splitRowWise()

        rsID = self.comm.bcast(startID, root=0)
        reID = self.comm.bcast(endID, root=0)

        if self.rank == 0 and self.size > 1:
            print("\n Define grid partition based on row number for parallel processing... " )
        self.comm.Barrier()

        displacements = self.comm.bcast(disps, root = 0)
        del disps
        gc.collect()
        t1 = time.clock()
        if self.rank < self.size-1:
            steps =   int(displacements[self.rank+1]) - int(displacements[self.rank])
        else:
            steps = self.nNodes - int(displacements[self.rank])

        if self.size > 1 and not self.test:
            print('  +  Domain for processor {:3d} starts at row {:4d} and end at row {:4d}'.format(self.rank,rsID[self.rank],reID[self.rank]))

        del displacements
        gc.collect()
        self.comm.Barrier()

        if self.rank == 0 and not self.test:
            print("\n Starting LEC computation... \n " )

        lc = reID[self.rank]-rsID[self.rank]
        localLEC = np.zeros((self.nr,self.nc))

        k = 0
        for r in range(rsID[self.rank],reID[self.rank]):
            for c in range(0,self.nc):
                if k%fout==0 and k>0:
                    if self.rank == 0 and not self.test:
                        print('  +  Compute closeness between sites in {:.2f} s - completion: {:.2f} %'.format(time.clock()-t1,k*100./steps))
                    if self.test:
                        self._test_progress("Test bioLEC installation:", k/steps)
                    t1 = time.clock()
                if self.nz[r,c]>=self.sealevel:
                    localLEC += np.exp(-self.computeMinPath(r, c)/self.sigma2)
                k += 1


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

        if self.test:
            self._test_progress("Test bioLEC installation:", k/steps)
            if abs(int(self.LEC.sum())-4283765.)<=1.:
                if abs(int(self.LEC.max())-760.)<=1.:
                    if abs(int(self.LEC.min())-26.)<=1.:
                        print('All tests were successful...')
                    else:
                        print('Error when testing installation...',self.LEC.min())
                else:
                    print('Error when testing installation...',self.LEC.max())
            else:
                print('Error when testing installation...',self.LEC.sum())

        del valLEC
        gc.collect()

        if self.rank == 0 and not self.test:
            print('\n Landscape Elevation Connectivity calculation took: {:.2f} s'.format(time.clock()-time0))

        return

    def viewResult(self, imName=None, size=(9,5), fsize=11, cmap1=plt.cm.summer_r, cmap2=plt.cm.coolwarm, dpi=200):
        """
        This function **plots** and **saves** in a figure the result of the *LEC computation*.

        Args:
            imName (str): (string) image name with extension
            size  : size of the image [default: (9,5)]
            fsize (int): title font size [default: 11]
            cmap1 : color map for elevation grid [default: plt.cm.summer_r]
            cmap2 : color map for LEC grid [default: cmap2=plt.cm.coolwarm]
            dpi  (int): resolution of the saved image [default: 200]

        """

        rcParams['figure.figsize'] = size
        fig, (ax1, ax2) = plt.subplots(1,2)

        fig.suptitle('Elevation grid and associated landscape elevational connectivity', fontsize=fsize)

        # SUBPLOT 1
        ax1.set_title('Elevation', fontsize=fsize-1)
        im1 = ax1.imshow(np.flipud(self.data), vmin=self.data.min(), vmax=self.data.max(),
                         cmap=cmap1, aspect='auto')

        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax1)
        cbar1.ax.tick_params(labelsize=fsize-3)

        # Remove ticks from ax1
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)

        # SUBPLOT 2
        ax2.set_title('LEC', fontsize=fsize-1)
        im2 = ax2.imshow(np.flipud(self.LEC), vmin=self.LEC.min(), vmax=self.LEC.max(),
                         cmap=cmap2, aspect='auto')

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.ax.tick_params(labelsize=fsize-3)

        # Remove xticks from ax2
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)

        # Make space for title
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        if self.size == 1:
            plt.show()

        if imName is not None:
            fig.savefig(imName, dpi=dpi, bbox_inches='tight')

        plt.close(fig)

        return

    def viewElevFrequency(self, input=None, imName=None, nbins=80, size=(6,6), fsize=11, dpi=200):
        """
        This function **plots** and **saves** in a figure the *distribution of elevation*.

        Args:
            input (str): CSV file name containing the dataset
            imName (str): (string) image name with extension
            nbins (int): number of bins for the histogram (default: 80)
            size  : size of the image [default: (6,6)]
            fsize (int): title font size [default: 11]
            dpi  (int): resolution of the saved image [default: 200]

        Warning:
            The filename needs to be provided without extension.
        """

        data = pd.read_csv(input+'.csv')
        minZ = max(self.sealevel,data.Z.min())
        ax = data.Z.plot(kind='hist', color='Blue', alpha=0.2, bins=nbins, xlim=(minZ,data.Z.max()))

        ax.set_title('Elevation frequency as function of site elevation', fontsize=fsize)

        plt.xlabel('Elevation', fontsize=fsize-1)
        fig = data.Z.plot(kind='density', figsize=size, ax=ax, xlim=(minZ,data.Z.max()),
                       linewidth=4, fontsize = fsize-3, secondary_y=True, y='Density').get_figure()

        ax.set_ylabel('Frequency count', fontsize=fsize-1)

        ax.tick_params(axis='x', labelsize=fsize-3)
        ax.tick_params(axis='y', labelsize=fsize-3)

        plt.ylabel('Density', fontsize=fsize-1)

        if self.size == 1:
            plt.show()

        if imName is not None:
            fig.savefig(imName, dpi=dpi, bbox_inches='tight')

        plt.close(fig)
        plt.close()

        del data
        gc.collect()

        return

    def viewLECFrequency(self, input=None, imName=None, size=(6,6), fsize=11, dpi=200):
        """
        This function **plots** and **saves** in a figure the *distribution of LEC with elevation*.

        Args:
            input (str): CSV file name containing the dataset
            imName (str): (string) image name with extension
            size  : size of the image [default: (6,6)]
            fsize (int): title font size [default: 11]
            dpi  (int): resolution of the saved image [default: 200]

        Warning:
            The filename needs to be provided without extension.
        """

        data = pd.read_csv(input+'.csv')
        minZ = max(self.sealevel,data['Z'].min())

        ff = data.plot(kind='scatter', x='Z', y='LEC', c='white', edgecolors='lightgray', figsize=size,
                  xlim=(minZ,data['Z'].max()), s=5)

        ff.set_title('Landscape elevational connectivity as function of site elevation', fontsize=fsize)

        plt.xlabel('Elevation', fontsize=fsize-1)
        ff.set_ylabel('LEC', fontsize=fsize-1)
        ff.tick_params(axis='x', labelsize=fsize-3)
        ff.tick_params(axis='y', labelsize=fsize-3)


        if self.size == 1:
            plt.show()

        if imName is not None:
            fig = ff.get_figure()
            fig.savefig(imName, dpi=dpi, bbox_inches='tight')

        plt.close()

        del data
        gc.collect()

        return

    def viewLECZbar(self, input=None, imName=None, nbins=40, size=(6,6), fsize=11, dpi=200):
        """
        This function **plots** and **saves** in a figure the *distribution of LEC and elevation with elevation*
        as well as the *standard deviation for LEC values* for each bins.

        Args:
            input (str): CSV file name containing the dataset
            imName (str): (string) image name with extension
            nbins (int): number of bins for the histogram (default: 40)
            size  : size of the image [default: (6,6)]
            fsize (int): title font size [default: 11]
            dpi  (int): resolution of the saved image [default: 200]

        Warning:
            The filename needs to be provided without extension.
        """

        data = pd.read_csv(input+'.csv')

        n, _ = np.histogram(data.Z, bins=nbins)
        sy, _ = np.histogram(data.Z, bins=nbins, weights=data.LEC)
        sy2, _ = np.histogram(data.Z, bins=nbins, weights=data.LEC*data.LEC)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)

        fig, ax = plt.subplots(figsize=size)

        ax.set_title('Landscape elevational connectivity as function of site elevation with error bars', fontsize=fsize)

        plt.plot((_[1:] + _[:-1])/2, mean,color='steelblue',zorder=2,linewidth=3)
        plt.scatter(data.Z, data.LEC, c='w',edgecolors='lightgray', zorder=0,alpha=1.,s=5)
        minZ = max(data.Z.min(),self.sealevel)
        ax.set_xlim(minZ,data.Z.max())
        ax.tick_params(axis='x', labelsize=fsize-3)
        ax.tick_params(axis='y', labelsize=fsize-3)

        plt.xlabel('Elevation', fontsize=fsize-1)
        plt.ylabel('LEC', fontsize=fsize-1)

        (_, caps, _) = plt.errorbar(
            (_[1:] + _[:-1])/2, mean, yerr=std, fmt='-o', c='steelblue',markersize=4, capsize=3,zorder=1,linewidth=1.25)

        for cap in caps:
            cap.set_markeredgewidth(1)

        ax2=ax.twinx()
        data.Z.plot(kind='density',secondary_y=True, ax=ax2, xlim=(minZ,data.Z.max()),
                       color='green', linewidth=3, zorder=0, fontsize = fsize-3)

        plt.ylabel('Density', color='green', fontsize=fsize-1)
        plt.tick_params(axis='x', labelsize=fsize-3)
        plt.tick_params(axis='y', labelsize=fsize-3)

        if self.size == 1:
            plt.show()

        if imName is not None:
            fig.savefig(imName, dpi=dpi, bbox_inches='tight')

        plt.close()

        del data, _, n, sy, sy2, mean, std, caps
        gc.collect()

        return

    def viewLECZFrequency(self, input=None, imName=None, size=(6,6), fsize=11, dpi=200):
        """
        This function **plots** and **saves** in a figure the *distribution of LEC and elevation with elevation*.

        Args:
            input (str): CSV file name containing the dataset
            imName (str): (string) image name with extension
            size  : size of the image [default: (6,6)]
            fsize (int): title font size [default: 11]
            dpi  (int): resolution of the saved image [default: 200]

        Warning:
            The filename needs to be provided without extension.
        """

        data = pd.read_csv(input+'.csv')
        minZ = max(self.sealevel,data['Z'].min())
        ax = data.plot(kind='scatter', x='Z', y='LEC', c='white', edgecolors='lightgray', figsize=size,
                  xlim=(minZ,data['Z'].max()), s=5)

        ax.set_title('Landscape elevational connectivity as function of site elevation', fontsize=fsize)

        plt.xlabel('Elevation', fontsize=fsize-1)
        ax2=ax.twinx()
        fig = data.Z.plot(kind='density',secondary_y=True, ax=ax2, xlim=(minZ,data['Z'].max()),
                       fontsize = fsize-3, linewidth=4).get_figure()

        ax.set_ylabel('LEC', fontsize=fsize-1)
        ax.tick_params(axis='x', labelsize=fsize-3)
        ax.tick_params(axis='y', labelsize=fsize-3)

        plt.ylabel('Density', fontsize=fsize-1)

        if self.size == 1:
            plt.show()

        if imName is not None:
            fig.savefig(imName, dpi=dpi, bbox_inches='tight')

        plt.close(fig)
        plt.close()

        del data
        gc.collect()

        return

    def writeLEC(self, filename='LECout'):
        """
        This function writes the computed landscape elevational connectivity array in a **CSV file**
        and create a **VTK visualisation file** (.vts).

        Args:
            filename (str): output file name without format extension.

        Warning:
            The filename needs to be provided without extension.
        """

        time0 = time.clock()
        if self.rank == 0:
            XX = self.X.flatten()
            YY = self.Y.flatten()
            ZZ = self.Z.flatten()
            LL = self.LEC[self.nr0:self.nr1,self.nc0:self.nc1].flatten()
            df = pd.DataFrame({'X':XX,'Y':YY,'Z':ZZ,'LEC':LL})
            df.to_csv(filename+'.csv', columns=['X', 'Y', 'Z', 'LEC'], sep=',', index=False , header=1)

            vtkfile = filename
            XX = XX.reshape(self.ny,self.nx)
            YY = YY.reshape(self.ny,self.nx)
            ZZ = ZZ.reshape(self.ny,self.nx)
            LL = LL.reshape(self.ny,self.nx)
            Xv = np.zeros((self.ny,self.nx,2))
            Yv = np.zeros((self.ny,self.nx,2))
            Zv = np.zeros((self.ny,self.nx,2))
            LECv = np.zeros((self.ny,self.nx,2))
            normLEC = np.zeros((self.ny,self.nx,2))

            Xv[:,:,0] = XX
            Yv[:,:,0] = YY
            Zv[:,:,0] = ZZ
            LECv[:,:,0] = LL
            normLEC[:,:,0] = LL/LL.max()

            Xv[:,:,1] = XX
            Yv[:,:,1] = YY
            Zv[:,:,1] = ZZ
            LECv[:,:,1] = LL
            normLEC[:,:,1] = LL/LL.max()

            gridToVTK(vtkfile, Xv, Yv, Zv, pointData = {"elevation" : Zv, "LEC" :LECv, "nLEC" :normLEC})
            del XX,YY, ZZ, LL, Xv, Yv, Zv, LECv, normLEC

        self.comm.Barrier()

        if self.rank == 0:
            print('\n Writing results on disk took: {:.2f} s'.format(time.clock()-time0))

        return
