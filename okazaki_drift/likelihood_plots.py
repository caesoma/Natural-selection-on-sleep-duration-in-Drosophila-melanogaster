"""
Provides functions to plot line likelihoods and distribution of randomly chosen sites putatively under neutral evolution
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "1.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Development"


import os, sys
import numpy
import pandas
import warnings

import matplotlib.pyplot as pyplot
from matplotlib.pyplot import figure, plot, show
import seaborn


def plot_logp_panels(data, subdata, shiftlabel, colors, edge=0, treatment="sel", replicates="rep", kde=True, xaxis=[], yaxis=[], legend=[], fsize=0, figpath=""):

    if treatment not in data.columns:
        data[treatment] = "neutral"
        subdata[treatment] = "neutral"
    if replicates not in data.columns:
        data[replicates] = int(1)
        subdata[replicates] = int(1)

    R = len( sorted( set( data[replicates] ) ) )
    fig = figure(figsize=(12,3*R)) if (fsize==0) else figure(figsize=fsize)

    if ( len(colors) == R ):
        cmap = colors
    else:
        cmap = colors

    panel = lambda panx, pany: pyplot.subplot2grid( (R,2), (panx, pany) )

    for i,repp in enumerate( sorted( set( data[replicates] ) ) ):

        titleBool = True if (i==0) else False
        xlaBool = True if (i==(R-1)) else False


        panel(i,0)

        for j, treat in enumerate( sorted( set( data[treatment] ) ) ):
            legendBool = True if ( (legend==True) and (i==0) ) else False

            plot_logp_distribution_maf( data[(data[treatment]==treat) & (data[replicates]==repp) ], subdata[ (subdata[treatment]==treat) & (subdata[replicates]==repp) ], shiftlabel=shiftlabel, allele="major", tit=titleBool, xlab=xlaBool, kdensity=kde, kolor=cmap[j], kedge=edge, lab=str(treat) + " " + str(repp), legend=legendBool)

            if (xaxis != []):
                xlimits = [numpy.min(xaxis), numpy.max(xaxis)]
                pyplot.xticks([]) if (i==0) else None
                pyplot.xlim(xlimits)
            if (yaxis != []):
                ylimits = [numpy.min(yaxis), numpy.max(yaxis)]
                pyplot.ylim(ylimits)
                pyplot.yticks(yaxis)

        panel(i, 1)

        for j, treat in enumerate( sorted( set( data[treatment] ) ) ):
            plot_logp_distribution_maf( data[ (data[treatment]==treat) & (data[replicates]==repp) ], subdata[ (subdata[treatment]==treat) & (subdata[replicates]==repp) ], shiftlabel=shiftlabel, allele="minor", tit=titleBool, xlab=xlaBool, kdensity=kde, kolor=cmap[j], kedge=edge, lab=str(treat) + " " + str(repp), legend=False)

            if (xaxis != []):
                xlimits = [numpy.min(xaxis), numpy.max(xaxis)]
                pyplot.xticks([]) if (i==0) else None
                pyplot.xlim(xlimits)
            if (yaxis != []):
                ylimits = [numpy.min(yaxis), numpy.max(yaxis)]
                pyplot.ylim(ylimits)
                pyplot.yticks([])


    if (figpath == "show"):
        show()
    elif (figpath != ""):
        fig.savefig(figpath)

    return None



def plot_logp_distribution_maf(data, subdata, shiftlabel, allele="major", tit=False, xlab=False, kdensity=True, kolor="CornFlowerBlue", kedge=0, lab="line", legend=True):

    L = data.shape[0]

    logpSample = numpy.log( subdata.loc[ (subdata[shiftlabel]<0), "likelihood"].values ) if (allele=="major") else numpy.log( subdata.loc[ (subdata[shiftlabel]>0), "likelihood"].values )

    logpSampleArray = logpSample[:-(logpSample.shape[0] % L)].reshape([-1, L])

    if (kdensity==False):
        pyplot.hist( numpy.mean(logpSampleArray, axis=1 ), color=kolor, alpha=0.3, bins=20, density=False, label="(random set)")
    elif (kdensity==True):
        kdeargs = {'linewidth': kedge, 'alpha': 0.3, 'shade': True, "label": "(random set)" } if (legend==True) else {'linewidth': kedge, 'alpha': 0.3, 'shade': True }

        seaborn.distplot( numpy.mean(logpSampleArray, axis=1 ), hist=False, kde=True, rug=False, bins=20, color = kolor, hist_kws={'edgecolor':'none'}, kde_kws=kdeargs)

    logpVariants = numpy.mean( numpy.log( data.loc[ (data[shiftlabel]<0), "likelihood"].values) ) if (allele=="major") else numpy.mean( numpy.log( data.loc[ (data[shiftlabel]>0), "likelihood"].values ) )

    pyplot.axvline( logpVariants, color=kolor, label=lab )

    pyplot.title("towards " + str(allele) + " allele") if (tit==True) else pyplot.title("")
    pyplot.xlabel("average loglikelihood") if (xlab==True) else pyplot.xlabel("")
    pyplot.legend() if (legend==True) else None

    return None
