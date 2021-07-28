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

from okazaki_drift.likelihood_plots import plot_logp_panels, show

# loads and prints arguments provided to the script in the right order:
if ( len(sys.argv) == 5 ):
    variantsShiftLikelihoods = sys.argv[1]
    subsetsShiftLikelihoods = sys.argv[2]
    mafShift = sys.argv[3]
    project_dir = sys.argv[4]
    data_dir = sys.argv[4]
    results_dir = sys.argv[4]
elif ( len(sys.argv) == 7 ):
    variantsShiftLikelihoods = sys.argv[1]
    subsetsShiftLikelihoods = sys.argv[2]
    mafShift = sys.argv[3]
    project_dir = sys.argv[4]
    data_dir = sys.argv[5]
    results_dir = sys.argv[6]
elif ( len(sys.argv) == 4 ):
    variantsShiftLikelihoods = sys.argv[1]
    subsetsShiftLikelihoods = sys.argv[2]
    mafShift = sys.argv[3]
    project_dir = os.getcwd()
    data_dir = os.path.join( project_dir, "data")
    results_dir = os.path.join( project_dir, "results")
else:
    # manual within-script setting of paths
    variantsShiftLikelihoods = "cmh_significant_sites_likelihoods.csv"
    subsetsShiftLikelihoods = "subset_all_sites_innersplit_likelihoods.csv"
    mafShift = "MAF75-MAF12"
    project_dir = os.getcwd()
    data_dir = os.path.join( project_dir, "data")
    results_dir = os.path.join( project_dir, "results")


print("file:", sys.argv[0])
print("Variants frequency change data:", variantsShiftLikelihoods )
print("Subset frequency change data:", subsetsShiftLikelihoods)
print("Project dir:", project_dir)
print("Data dir:", data_dir)


variantdata = pandas.read_csv( os.path.join( results_dir, variantsShiftLikelihoods), sep=',', header=0)
subdata = pandas.read_csv( os.path.join( results_dir, subsetsShiftLikelihoods), sep=',', header=0, index_col=0)

plot_logp_panels(variantdata, subdata, shiftlabel=mafShift, colors=["black", "CornFlowerBlue", "FireBrick"], edge=0, treatment="sel", replicates="rep", kde=True, figpath=os.path.join( project_dir, "figures", variantsShiftLikelihoods[:-4]+".png"), legend=["control", "long", "short"])

show()
