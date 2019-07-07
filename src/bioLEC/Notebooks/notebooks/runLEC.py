import argparse
from mpi4py import MPI
import bioLEC as LEC

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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
parser.add_argument('-n','--nout',help='Number for output frequency during run', required=False, action="store_true", default=500)
parser.add_argument('-d','--delimiter',help='String for elevation grid csv delimiter', required=False,action="store_true",default=' ')
parser.add_argument('-l','--level',help='Float for sea level position', required=False,action="store_true",default=-1.e6)
parser.add_argument('-v','--verbose',help='True/false option for verbose', required=False,action="store_true",default=False)

args = parser.parse_args()
if args.verbose and rank == 0:
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
  print("   + Sea level position: {}".format(args.level))
  print("   + Number for output frequency: {}\n".format(args.nout))

biodiv = LEC.landscapeConnectivity(filename=args.input,periodic=args.periodic,symmetric=args.symmetric,
                                    sigmap=args.width,sigmav=args.fix,connected=args.connected,
                                    delimiter=args.delimiter,sl=args.level)

biodiv.computeLEC(args.nout)

biodiv.writeLEC(args.output)

if rank == 0:
    biodiv.viewResult(imName=args.output+'.png')
    biodiv.viewElevFrequency(input=args.output,imName=args.output+'_zfreq.png')
    biodiv.viewLECZFrequency(input=args.output,imName=args.output+'_leczfreq.png')
    biodiv.viewLECFrequency(input=args.output,imName=args.output+'_lecfreq.png')
    biodiv.viewLECZbar(input=args.output,imName=args.output+'_lecbar.png')
