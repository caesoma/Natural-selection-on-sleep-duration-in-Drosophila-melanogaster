#!/usr/bin/python3

"""
Basic drift simulation functions
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "1.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Development"

import os
import numpy
import scipy.integrate
from numpy import sin, cos
from numpy.random import binomial as rbinomial, poisson as rpoisson, normal as rnormal

import pandas

import matplotlib
import matplotlib.pyplot as pyplot

from matplotlib.pyplot import figure, plot, show

# from IPython.display import Image

def drift_history(N, pt1, generation, history=None):
    if history==None:
        history = [pt1]

    if generation == 0:
        return history
    else:
        pt = list( rbinomial(N, pt1)/N ) if type(pt1) != float else rbinomial(N, pt1)/N
        # history.append( pt )
        return pt + drift(N, pt, generation - 1, history)


def drift(N, p, generation):
    if generation == 0:
        return []
    else:
        pt = list( rbinomial(N, p)/N ) if type(p) != float else rbinomial(N, p)/N
        # history.append( pt )
        return [p] + drift(N, pt, generation - 1)


def plot_drift(trajectories, Ne):

    _2N = 2*Ne

    runs = trajectories.shape[0]
    generations = trajectories.shape[1]


    subplotdim = (1,2)
    panel2 = lambda panx, pany: pyplot.subplot2grid( subplotdim, (panx, pany) )
    colormap = matplotlib.cm.viridis( [r/runs for r in range(runs)] )

    fig1 = figure( figsize=(15,5) )
    panel2(0,0)
    #plot(range(generations), numpy.transpose(p))
    [plot(range(generations), trajectories[r,:], color=colormap[r,:] ) for r in range(runs)]
    pyplot.yticks([i/10 for i in range(11)])
    pyplot.xlabel("generation")
    pyplot.ylabel("allele frequency")

    panel2(0,1)
    pyplot.hist(trajectories[:,-1], bins=[i/_2N for i in range(0, _2N+1, 4)], orientation="horizontal", density=True, color=colormap[0,:])
    pyplot.xticks([0])
    pyplot.yticks([])
    pyplot.xlabel("density")

    return fig1

def plot_multiple(p0, Ne, generations, shesalittlerunarray, save=False):

    for runs in shesalittlerunarray:
        trajectories = numpy.transpose( drift( 2*Ne, runs*[p0], generations ) )
        fign = plot_drift(trajectories, Ne)

        if save==True:
            fign.savefig("drift_trajectories"+str(generations)+"_"+str(runs)+".png", bbox_inches="tight")

    return None

os.chdir(os.path.expanduser("~/Desktop"))


p0 = 0.5
Ne = 42
generations = 30
runs = 100

p = numpy.transpose( drift( 2*Ne, runs*[p0], generations ) )

plot_drift(p, Ne)

"""
# plots
subplotdim = (1,2)
panel2 = lambda panx, pany: pyplot.subplot2grid( subplotdim, (panx, pany) )
colormap = matplotlib.cm.viridis( [r/runs for r in range(runs+1)] )

fig1 = figure( figsize=(15,5) )
panel2(0,0)
#plot(range(generations), numpy.transpose(p))
[plot(range(generations), p[r,:], color=colormap[r,:] ) for r in range(runs)]
pyplot.yticks([i/10 for i in range(11)])
pyplot.xlabel("generation")
pyplot.ylabel("allele frequency")

panel2(0,1)
pyplot.hist(p[:,-1], bins=[i/Ne for i in range(0, Ne+1, 4)], orientation="horizontal", density=True, color=colormap[0,:])
pyplot.xticks([0])
pyplot.yticks([])
pyplot.xlabel("density")

show()
"""
