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
from functools import reduce
import numba
import scipy.special, scipy.stats
import pandas, json

import warnings
import timeit

import matplotlib.pyplot as pyplot

from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import figure, plot, show

if ( platform.system() == 'Darwin' ):
    util_dir = os.path.join( os.path.expanduser("~"), "OneDrive - National Institutes of Health", "code", "python", "basicutils" )
elif ( platform.system() == 'Linux' ):
    util_dir = os.path.join( os.path.expanduser("~"), "ownCloud", "code", "python", "basicutils" )
elif ( platform.system() == 'Windows' ):
    util_dir = os.path.join( os.path.expanduser("~"), "OneDrive - National Institutes of Health", "code", "python", "basicutils" )

sys.path.append( util_dir )

from iox import read_json_file, write_json_file

def recursive_binomial_array(p0list, Ne, generations, X=False):

    _2N = int(1.5*Ne) if X else 2*Ne
    rangeneration = range(generations)

    allelerray = numpy.full([len(p0list), generations, _2N+1], numpy.NaN)

    for i,p0 in enumerate(p0list):
        print(">>> initial frequency: ", p0)
        allelerray[i,:,:] = recursive_binomial(p0, Ne, rangeneration, X=X);

    return allelerray



def binomial_allelenumber(k, _2N, allelist):
    # k: number of alleles in previous generations
    # _2N: effective population size
    p = k/_2N  # binomial probabilty for next generation
    distribution = scipy.stats.binom( _2N, p).pmf(allelist)  # binomial probability distribution for all possible number of alleles
    return distribution


def binomial_mix(pkg, _2N, allelist):  # binomial_mix :: (Int a, Float b) => a -> a -> [a] -> [b]
    mix = [ binomial_allelenumber(k, _2N, allelist) * pkgi for k,pkgi in enumerate(pkg) ]  # probability distributions conditioned on the previous generation distribution
    mixsum = numpy.sum( mix, axis=0 )  # sum of all probability distributions gives new probability distribution (binomial mixture)
    return list( mixsum )


def recursive_binomial(pkg, Ne, rangeneration, X=False):
    """ recursively compounded binomial distributions"""

    _2N = int(1.5*Ne) if X else 2*Ne  # effective population size
    krange = numpy.arange(0, _2N+1)  # range of possible number of alleles
    generation = list(rangeneration)[0]
    future = list(rangeneration)[1:]

    # print(">>> generation:", generation)

    if (generation==0):
        if ((type(pkg) == float) | (type(pkg) == numpy.float64)): # function takes a floating point single number as initial allele frequency
            pk0 = (_2N+1)*[0]
            idk = int( numpy.round( pkg*_2N, 0) ) # index of allele k should be the same as the number of alleles itself
            pk0[idk] = 1  # initial distribution in this case is that with 100% density at the number of alleles k
            pk1 = list( scipy.stats.binom(_2N,pkg).pmf(krange) )
        elif ( ( type(pkg) == numpy.ndarray ) and (len(pkg)==_2N) and (numpy.sum(pkg)==1) ):
            pk0 = list(pkg)
            pk1 = binomial_mix(pkg, _2N, krange)  # computes array with probability disributions given each probability distribution in list (mixture distribution)
        elif ((type(pkg) == list) and (len(pkg)==_2N) and (numpy.sum(pkg)==1) ):  # initial probabilities for each allele can be provided directly as list (if provided as array, as previous condition, it will be converted) describing distribution (must be a real statistical distribution, i.e. total density must be normalized to unity)
            pk0 = pkg
            pk = lambda kplus,kg: scipy.stats.binom(_2N,kg/_2N).pmf(kplus)
            pk1 = binomial_mix(pkg, _2N, krange)
        else:
            raise ValueError("unrecognized type -- p0 must be scalar proportion, numpy array of size 2Ne, or list of size 2Ne (with unity total density)")

        return [pk0] + [pk1] + recursive_binomial(pk1, Ne, future, X=X)

    elif (future==[]):
        #pk = lambda kplus,kg: scipy.stats.binom(_2N,kg/_2N).pmf(kplus)
        #pkglast = list( numpy.sum([[pk(kplus,k) * pkgi for kplus in krange ] for k,pkgi in enumerate(pkg)], axis=0) )

        return []  # [pkglast]

    else:
        pkgplus = binomial_mix(pkg, _2N, krange)

        return [pkgplus] + recursive_binomial(pkgplus, Ne, future, X=X)


@numba.jit
def mixture_binomial(pkg, Ne, generations, X=False):
    """ for-loop implementation of recursively compounded binomial distributions"""

    _2N = int(1.5*Ne) if X else 2*Ne  # effective population size
    krange = numpy.arange(0, _2N+1)  # range of possible number of alleles

    allelevolution = numpy.full( [ generations, _2N+1 ], numpy.NaN )  # array with probability distribution of allele number by generation (transposed)

    if ((type(pkg) == float) | (type(pkg) == numpy.float64)):  # function takes a float as initial allele frequency
        pk0 = (_2N+1)*[0]
        idk = int( numpy.round( pkg*_2N, 0) ) # index of allele k should be the same as the number of alleles itself
        pk0[idk] = 1  # initial distribution in this case is that with 100% density at the number of alleles k
    elif ((type(pkg) == numpy.ndarray) and (len(pkg)==_2N) and (numpy.sum(pkg)==1) ):
        pk0 = list(pkg)
    elif ((type(pkg) == list) and (len(pkg)==_2N) and (numpy.sum(pkg)==1) ): # initial probabilities for each allele can be provided directly as list (if provided as array, as previous condition, it will be converted) describing distribution (must be a real statistical distribution, i.e. total density must be normalized to unity)
        pk0 = pkg
    else:
        raise ValueError("unrecognized type -- p0 must be scalar proportion, numpy array of size 2Ne, or list of size 2Ne (with unity total density)")

    allelevolution[0,:] = pk0

    for g in range(1,generations):

        pkg = allelevolution[g-1,:]

        print(">>> generation:", g)

        pkgplus = numpy.full( [_2N+1,_2N+1], numpy.NaN)
        for k,pkgi in enumerate(pkg):
            pkgplus[k,:] = scipy.stats.binom(_2N,k/_2N).pmf(krange) * pkgi

        allelevolution[g,:] = pkgplus

    return allelevolution


def build_confidence_interval( p0, freqrange100, darray, Ne, zerodensity=[], leftdensity=[], rightdensity=[], X=False ):

    _2N = int(1.5*Ne) if X else 2*Ne
    freqrange2N = numpy.arange(0, 1+(1/(_2N)), 1/(_2N))
    p0index = list(freqrange100).index(p0)
    k0 = int(round(p0*_2N))

    probabilityDensity = darray[ p0index, -1, :]

    leftk = len(leftdensity)
    rightk = len(rightdensity)

    leftcdf = numpy.sum(leftdensity)
    rightcdf = numpy.sum(rightdensity)
    centercdf = numpy.sum(zerodensity)

    kdf = centercdf + leftcdf + rightcdf

    # print( (k0-leftk)*" " + leftk*"*" + "[" + str(k0) + "]" + rightk*"*" + (k0+rightk)*" " )

    if zerodensity==[]:
        density0 = [ probabilityDensity[k0] ]
        if (p0 <= 0.5):
            densileft = [ probabilityDensity[k0-1] ]
            return build_confidence_interval(p0, freqrange100, darray, Ne, zerodensity=density0, leftdensity=densileft, rightdensity=[], X=X)
        else:
            densiright = [ probabilityDensity[k0+1] ]
            return build_confidence_interval(p0, freqrange100, darray, Ne, zerodensity=density0, leftdensity=[], rightdensity=densiright, X=X)
    elif ( kdf < 0.95 ):

        if (leftcdf > rightcdf):
            #print(">>> adding density to the right")
            rindex = k0+(rightk+1)
            breitbart = [ probabilityDensity[ rindex ] ] if (rindex < len(probabilityDensity)) else []
            densiright = list(rightdensity) + breitbart
            return build_confidence_interval(p0, freqrange100, darray, Ne, zerodensity=zerodensity, leftdensity=leftdensity, rightdensity=densiright, X=X)

        elif (leftcdf < rightcdf):
            #print(">>> adding density to the left")
            lindex = k0-(leftk+1)
            jacobin = [ probabilityDensity[ lindex ] ] if (lindex >= 0) else []
            densileft = list(leftdensity) + jacobin
            return build_confidence_interval(p0, freqrange100, darray, Ne, zerodensity=zerodensity, leftdensity=densileft, rightdensity=rightdensity, X=X)

    else:
        #print(">>> left density:", leftdensity )
        #print(">>> center densityk:", zerodensity)
        #print(">>> right densityk:", rightdensity )

        print("cdf:", kdf)
        print(">>> left k:", k0-leftk)
        print(">>> center k:", k0)
        print(">>> right k:", (k0+rightk+1))

        return freqrange2N[k0-leftk], freqrange2N[k0+rightk]  # (k0-leftk), k0, (leftk+1+rightk)


def build_confidence_interval_map(freqrange100, darray, Ne, X=False):

    f = lambda p: build_confidence_interval( p, freqrange100, darray, Ne, zerodensity=[], leftdensity=[], rightdensity=[], X=X )

    barmap = list( map(f, freqrange100[1:-1]) )

    pandamap = pandas.DataFrame( freqrange100[1:-1], columns=['p0'] )

    # pandamap = pandas.DataFrame( barmap, columns=['lower', 'upper'] )

    pandamap['lower'] = numpy.array(barmap)[:,0]
    pandamap['upper'] = numpy.array(barmap)[:,1]

    return pandamap


def empirical_distributions_array(pseudodata, pseudometadata, Ne, generations, X=False):

    _2N = int(1.5*Ne) if X else 2*Ne
    rangeneration = range(generations)
    p0list = numpy.sort( list( set(pseudodata['freq_G 1']) ) )

    allelerray = numpy.full([len(p0list), generations, _2N+1], numpy.NaN)

    for i,p0 in enumerate(p0list):
        print(">>> initial frequency: ", p0)
        if (p0==0):
            allelerray[i,:,:] = generations*[ [1] + _2N*[0] ]
        elif (p0==1):
            allelerray[i,:,:] = generations*[ _2N*[0] + [1] ]
        else:
            allelerray[i,:,:] = empirical_distributions(pseudodata=pseudodata.loc[ ((pseudodata['freq_G 1']==numpy.round(p0, 2)) & (pseudometadata['line']=="L1")) ], Ne=Ne, bins=None, X=X)

    return allelerray


def empirical_distributions(pseudodata, Ne, bins=None, X=False):

    pseudodatatoo = pseudodata.iloc[:,1:] if reduce(lambda b,y: b or (y < 0) or (y > 1), list(set(pseudodata.iloc[:,0])), False) else pseudodata
    generations = pseudodatatoo.shape[1]

    g0 = pseudodatatoo.columns[0]

    _2N = int(1.5*Ne) if X else 2*Ne
    kint = _2N if (bins==None) else bins

    freqrange = numpy.arange(0, 1+(1/kint), 1/kint)
    freqrange2 = numpy.arange(0, 1+(2/kint), 1/kint)

    Y = numpy.full([generations, len(freqrange)], numpy.NaN)
    for j,l in enumerate( pseudodatatoo.columns ):

        Y[j,:] = numpy.histogram(pseudodata[l], bins=freqrange2, density=False)[0]

    return Y


def approximate_beta_array(p0list, Ne, generations, X=False):

    _2N = int(1.5*Ne) if X else 2*Ne
    rangeneration = range(generations)

    allelerray = numpy.full([len(p0list), generations, _2N+1], numpy.NaN)

    for i,p0 in enumerate(p0list):
        print(">>> initial frequency: ", p0)
        allelerray[i,:,:] = approximate_beta_distributed_frequencies(p0, Ne, Vq0=0, X=X)

    return allelerray


def approximate_beta_distributed_frequencies(p0, Ne, Vq0=0, trajectory=[], X=False):

    vartraj = variance_trajectory(p0, Ne, generations, Vq0, X=X) if (len(trajectory)==0) else trajectory
    generations = len(vartraj)

    _2N = int(1.5*Ne) if X else 2*Ne

    freqrange2 = numpy.arange(0, 1+(1/_2N), 1/_2N)

    B = numpy.full([generations, len(freqrange2)],  numpy.NaN)

    for j in range(0, generations):
        a,b = beta_parameters(p0, vartraj[j])
        B[j,:] = scipy.stats.beta(a,b).pdf(freqrange2)/_2N

    return B


def drift_variance(p0, Ne, T, Vq0, X=False):

    _2N = int(1.5*Ne) if X else 2*Ne

    Vq0Propagated = pow( 1 - ( 1/(_2N) ), T )*Vq0
    pq0Propagated = numpy.sum( [ pow( 1 - ( 1/(_2N) ) , t) for t in range(0,T) ] ) * p0*(1-p0)/(_2N)

    Vqt = pq0Propagated + Vq0Propagated

    return Vqt

def drift_variance2(p0, Ne, T, Vq0, X=False):

    _2N = int(1.5*Ne) if X else 2*Ne

    Vqt = p0*(1-p0) - (p0*(1-p0) - Vq0)* pow( 1 - ( 1/(_2N) ) , T)

    return Vqt

def variance_trajectory2(p0, Ne, T, Vq0, X=False):

    _2N = int(1.5*Ne) if X else 2*Ne

    Vqtrajectory = [drift_variance2(p0, _2N, t, Vq0) for t in range(T)]

    return numpy.array(Vqtrajectory)


def drift_variance_approx(p0, Ne, T, Vq0, X=False):

    _2N = int(1.5*Ne) if X else 2*Ne

    Vq0Propagated = numpy.exp( ( -T/(_2N) ) )*Vq0
    pq0Propagated = numpy.sum( [ numpy.exp(-( t/(_2N) ) ) for t in range(0,T) ] ) * p0*(1-p0)/(_2N)

    Vqt = pq0Propagated + Vq0Propagated

    return Vqt

def variance_trajectory(p0, Ne, T, Vq0, X=False):

    Vqtrajectory = [drift_variance(p0, Ne, t, Vq0, X=X) for t in range(T)]

    return numpy.array(Vqtrajectory)

def variance_trajectory_approx(p0, Ne, T, Vq0, X=False):

    Vqtrajectory = [drift_variance_approx(p0, Ne, t, Vq0, X=X) for t in range(T)]

    return numpy.array(Vqtrajectory)


def beta_parameters(mean, variance):

    a = (mean/variance) * (mean - (mean**2) - variance)
    b = (1-mean) * (mean - (mean**2) - variance) / variance

    return a,b


def variance_trajectory_plot(pseudodata, pseudometadata, trajectory=[], X=False):

    vartraj = variance_trajectory_approx(p0, Ne, generations, Vq0, X=X) if (len(trajectory)==0) else trajectory

    figure(figsize=(8,6));

    plot(range(1,pseudodata.shape[1]), numpy.var( pseudodata.loc[ pseudometadata['Simulation']<10 ], ddof=1, axis=0)[1:], label="10 simulations")

    plot(range(1,pseudodata.shape[1]), numpy.var( pseudodata.loc[ pseudometadata['Simulation']<100 ], ddof=1, axis=0)[1:], label="100 simulations")

    plot(range(1,pseudodata.shape[1]), numpy.var( pseudodata.loc[ pseudometadata['Simulation']<1000 ], ddof=1, axis=0)[1:], label="1000 simulations" )

    plot(range(1,pseudodata.shape[1]), numpy.var( pseudodata, ddof=1, axis=0)[1:] , label="10000 simulations");

    plot(range(1, pseudodata.shape[1]), vartraj, label="theoretical prediction")

    pyplot.xlabel("generation")
    pyplot.ylabel("Variance ($V_q$)")
    pyplot.yscale('linear')
    pyplot.title("$p_0=0.5$")
    pyplot.legend()

    return ">>> plots done, no output here"


def plot_allelefrequency_densities(freqrange, Y=numpy.arange(0), D=numpy.arange(0), B=numpy.arange(0), Ne=0, intervalist=1, subplot=True):

    _2N = len(freqrange)-1  # 2*Ne if Ne > 0 else

    generations = Y.shape[0] if (len(Y)>0) else D.shape[0]
    generatinterval = range(0, generations, intervalist) if (type(intervalist)==int) else intervalist
    ymax = numpy.max(D[1:,:]) if (len(D)>0) else numpy.max(Y[1:,:]/numpy.sum(Y[0,:]))

    panelcols = int( numpy.sqrt( len( generatinterval ) ) )
    panelrows = int( numpy.ceil( len( generatinterval ) / panelcols ) )
    if subplot: print(">>> subplot dimensions:", panelrows, panelcols)  # , 0+1)

    if (subplot==True):
        fig = figure(figsize=(12,8))
        pyplot.xlabel("allele frequency")
        for sub,g in enumerate(generatinterval):

            pyplot.subplot(panelrows, panelcols, sub+1)

            if len(Y)>0:
                pyplot.bar(freqrange, Y[g,:]/numpy.sum(Y[g,:]), width=1/_2N, color='CornFlowerBlue', label="simulation")
            if len(D)>0:
                plot(freqrange, D[g,:]/numpy.sum(D[g,:]), color='Crimson', label="binomial mixture")
            if len(B)>0:
                plot(freqrange, B[g,:], color='DarkOrange', label="beta distribution")

            if (g==generatinterval[0]):
                pyplot.legend()
            elif (g==generatinterval[-1]):
                pyplot.xlabel("allele frequency")
            pyplot.ylim([0,ymax/2])
            pyplot.yticks([])
            if g in generatinterval[-panelcols:]:
                pyplot.xticks(numpy.arange(0,1.2,0.2))
            else:
                pyplot.xticks([])
            pyplot.title("generation "+str(g))

    else:
        for sub,g in enumerate(generatinterval):
            figure(figsize=(6,5))
            pyplot.title("generation " + str(g))
            pyplot.xlabel("allele frequency")

            if len(Y)>0:
                pyplot.bar(freqrange, Y[g,:]/numpy.sum(Y[g,:]), width=1/_2N, color='CornFlowerBlue', label="simulation")
            if len(D)>0:
                plot(freqrange, D[g,:]/numpy.sum(D[g,:]), color='Crimson', label="binomial mixture")
            if len(B)>0:
                plot(freqrange, B[g,:], color='DarkOrange', label="beta distribution")

            pyplot.legend()

    return ">>> plots done, no output here"


def bar3d_empirical_distributions( freqrange, Y=numpy.arange(0), D=numpy.arange(0), B=numpy.arange(0) ):

    generations, _2N = Y.shape if len(Y)>0 else D.shape

    fig = figure()
    ax = fig.add_subplot(111, projection='3d')

    for g in range(1, generations):
        if len(Y)>0:
            ax.bar(freqrange, Y[g,:]/numpy.sum(Y[0,:]), width=1/_2N, zs=g, zdir='y', color="CornFlowerBlue", alpha=0.7)
        if len(D)>0:
            ax.plot(freqrange, numpy.repeat(g, len(freqrange)), D[g,:], color='Crimson', alpha=0.7)
        if len(B)>0:
            ax.plot(freqrange, numpy.repeat(g, len(freqrange)), B[g,:], color='DarkOrange', alpha=0.7)

    return ">>> plots done, no output here"


def bar3d_empirical_distributions2( freqrange, Y=numpy.arange(0), D=numpy.arange(0), B=numpy.arange(0) ):

    generations, _2N = Y.shape if len(Y)>0 else B.shape

    fig = figure()
    ax = fig.add_subplot(111, projection='3d')

    for g in range(1, generations):
        if len(Y)>0:
            ax.bar(freqrange, Y[g,:]/numpy.sum(Y[0,:]), width=1/_2N, zs=g, zdir='y', color="CornFlowerBlue", alpha=0.7)
        if len(D)>0:
            ax.plot(freqrange, numpy.repeat(g, len(freqrange)), D[g,:], color='Crimson', alpha=0.7)
        if len(B)>0:
            ax.plot(freqrange, numpy.repeat(g, len(freqrange)), B[g,:], color='DarkOrange', alpha=0.7)

    return ">>> plots done, no output here"


def weir3d_empirical_distributions(freqrange, Y=numpy.arange(0), D=numpy.arange(0), B=numpy.arange(0) ):

    generations, _2N = Y.shape if len(Y)>0 else D.shape

    F = numpy.transpose( len(freqrange)*[list( range(1,generations+1) ) ] )
    G = numpy.repeat(numpy.reshape(freqrange, [1,len(freqrange)]), generations, axis=0)

    fig = figure()
    ax = fig.add_subplot(111, projection='3d')

    if len(Y)>0:
        ax.plot_wireframe(F[1:], G[1:], Y[1:]/numpy.sum(Y[0,:]), color='CornFlowerBlue') # , rstride=10, cstride=10)
    if len(D)>0:
        ax.plot_wireframe(F[1:], G[1:], D[1:]/numpy.sum(D[0,:]), color='Crimson')
    if len(B)>0:
        ax.plot_wireframe(F[1:], G[1:], B[1:], color='DarkOrange')

    return ">>> plots done, no output here"
