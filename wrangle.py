#!/usr/bin/python3

"""
Provides analytical and empirical calculations of stochastic allele drift and summaries and plots of the quantities
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "1.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Development"

import os, sys, platform

import numpy
import pandas

from okazaki_drift.data_computations import split_by_generation, subset_alleles, add_selrep_from_line

# loads and prints arguments provided to the script in the right order:
if ( len(sys.argv) == 6 ):
    g0 = int(sys.argv[1])
    gN = int(sys.argv[2])
    variantsShiftData = sys.argv[3]
    allUnsplitData = sys.argv[4]
    project_dir = sys.argv[5]
    data_dir = sys.argv[5]
    results_dir = sys.argv[5]
elif ( len(sys.argv) == 8 ):
    g0 = int(sys.argv[1])
    gN = int(sys.argv[2])
    variantsShiftData = sys.argv[3]
    allUnsplitData = sys.argv[4]
    project_dir = sys.argv[5]
    data_dir = sys.argv[6]
    results_dir = sys.argv[7]
elif ( len(sys.argv) == 5 ):
    g0 = int(sys.argv[1])
    gN = int(sys.argv[2])
    variantsShiftData = sys.argv[3]
    allUnsplitData = sys.argv[4]
    project_dir = os.getcwd()
    data_dir = os.path.join( project_dir, "data")
    results_dir = os.path.join( project_dir, "results")
else:
    g0 = 12
    gN = 75
    variantsShiftData = "cmh_significant_sites.csv"
    allUnsplitData = "all_sites.csv"
    project_dir = os.getcwd()
    data_dir = os.path.join( project_dir, "data")
    results_dir = os.path.join( project_dir, "results")


print("file:", sys.argv[0])
print("Variants frequency change data:", variantsShiftData )
print("Subset frequency change data:", allUnsplitData)
print("Project dir:", project_dir)
print("Data dir:", data_dir)
print("Results dir:", results_dir)

# load data, if the 'sel' and 'rep' columns are not in the data the `add_selrep_from_line` can be used to create them
alldata = pandas.read_csv( os.path.join( data_dir, allUnsplitData), sep=',', header=0)
cmhdata = pandas.read_csv( os.path.join( data_dir, variantsShiftData), sep=',', header=0)

alljoin = split_by_generation(alldata, generati0N=(g0,gN), dontsplit=['Chrom', 'Pos', 'RefBase', 'VarBase', 'Line', 'sel', 'rep'], method='inner', savefile="", verbo=True)

maf0 = "MAF" + str(g0)
mafN = "MAF" + str(gN)
mafShift = str(mafN) + "-" + str(maf0)
subjoin = subset_alleles(cmhdata[ numpy.isfinite( cmhdata[ mafN+"-"+maf0] ) ], alljoin, allelabels=(maf0, mafN, mafShift), repetitions=1000, savefile=os.path.join( results_dir, "subset_" + allUnsplitData[:-4] + "_innersplit.csv"), verbo=True )
