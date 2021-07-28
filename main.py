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

import warnings
import timeit

# loads custom functions for drift and likelihood computations
from okazaki_drift.okazaki_drift import recursive_binomial_array
from okazaki_drift.data_computations import likelihood_of_data
from okazaki_drift.iox import read_json_file, write_json_file


# loads and prints arguments provided to the script in the right order:
if ( len(sys.argv) == 8 ):
    Ne = int(sys.argv[1])
    generations = int(sys.argv[2])
    maf0 = str(sys.argv[3])
    mafN = str(sys.argv[4])
    variantsShiftData = sys.argv[5]
    subsetsShiftData = sys.argv[6]
    project_dir = sys.argv[7]
    data_dir = sys.argv[7]
    results_dir = sys.argv[7]
elif ( len(sys.argv) == 10 ):
    Ne = int(sys.argv[1])
    generations = int(sys.argv[2])
    maf0 = str(sys.argv[3])
    mafN = str(sys.argv[4])
    variantsShiftData = sys.argv[5]
    subsetsShiftData = sys.argv[6]
    project_dir = sys.argv[7]
    data_dir = sys.argv[8]
    results_dir = sys.argv[9]
else:
    Ne = int(sys.argv[1])
    generations = int(sys.argv[2])
    maf0 = str(sys.argv[3])
    mafN = str(sys.argv[4])
    variantsShiftData = sys.argv[5]
    subsetsShiftData = sys.argv[6]
    project_dir = os.getcwd()
    data_dir = os.path.join( project_dir, "data")
    results_dir = os.path.join( project_dir, "results")

print("file:", sys.argv[0])
print("Variants frequency change data:", variantsShiftData )
print("Subset frequency change data:", subsetsShiftData)
print("Project dir:", project_dir)
print("Data dir:", data_dir)
print("Results dir:", results_dir)
print("Ne:", Ne)
print("# of generations:", generations)


# Defines frequencies from 0 to 1 in 0.01 increments
freqrange100 = numpy.arange(0, 1.01, 0.01)


# tries to load previously computed recursive binomial distributions from data folder, if they are not found, computes them from scratch using custom function "recursive_binomial_array"
try:
    dictG = read_json_file( os.path.join( results_dir, "binomial_array" + str(generations) + ".json"), printFlag=False )
    dictX = read_json_file( os.path.join( results_dir, "binomial_arrayX" + str(generations) + ".json"), printFlag=False)

    darray = numpy.array( dictG['recursive_binomial_array'] )
    xarray = numpy.array( dictX['recursive_binomial_array'] )
except:
    tic = timeit.default_timer()
    darray = recursive_binomial_array(p0list=freqrange100, Ne=Ne, generations=generations)
    tac = timeit.default_timer()

    print(">>> elapsed time:", tac-tic)

    dump = write_json_file( os.path.join( results_dir, "binomial_array" + str(generations) + ".json"), {"recursive_binomial_array": darray.tolist(), "2N": 2*Ne}, idnt=None)

    tic = timeit.default_timer()
    xarray = recursive_binomial_array(p0list=freqrange100, Ne=Ne, generations=generations, X=True)
    tac = timeit.default_timer()

    print(">>> elapsed time (X):", tac-tic)

    dump = write_json_file( os.path.join( results_dir, "binomial_arrayX" + str(generations) + ".json"), {"recursive_binomial_array": xarray.tolist(), "1.5N": 1.5*Ne}, idnt=None)


# from array with all distributions for every generation, a data frame with all distributions for the last generations only (starting with every frequency in the 0.01-increment list  above)
recursiveBinomialG = pandas.DataFrame( darray[:,-1,:], index=numpy.round(numpy.arange(0, 1.01, 0.01),2), columns=range(darray.shape[-1]) )

recursiveBinomialX = pandas.DataFrame( xarray[:,-1,:], index=numpy.round(numpy.arange(0, 1.01, 0.01),2), columns=range(xarray.shape[-1]) )


# loads variant data and subset data (the latter must be split and have the difference in frequency betewen initial and final generations)
variantdata = pandas.read_csv( os.path.join( data_dir, variantsShiftData ), sep=',', header=0)
subdata = pandas.read_csv( os.path.join( results_dir, subsetsShiftData), sep=',', header=0)


# computes the likelihood of observing a frequency given the initial frequency using custom function "likelihood_of_data"
variantdata['likelihood'] = likelihood_of_data( variantdata, recursiveBinomialG, recursiveBinomialX, (maf0, mafN) )
subdata['likelihood'] = likelihood_of_data( subdata, recursiveBinomialG, recursiveBinomialX, (maf0, mafN) )


# saves new file to <results_dir> folder
variantdata.to_csv( os.path.join( project_dir, "results", variantsShiftData[:-4] + "_likelihoods.csv" ), sep=',', header=True, index=0 )
subdata.to_csv( os.path.join( project_dir, "results", subsetsShiftData[:-4] + "_likelihoods.csv" ), sep=',', header=True, index=0 )
