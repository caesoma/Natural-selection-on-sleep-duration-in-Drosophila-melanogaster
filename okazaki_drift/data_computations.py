"""
Provides functions to compute wrangle allele frquency data and compute likelihood of observed frequency shifts under a drift model
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "1.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Development"

import numpy
import pandas
from numpy.random import choice


def drift_likelihood(p0, kN, recursiveBinomialN):

    """ Shortcut function to dataframe loc, for picking likelihood of observing a number of alleles k after N generations, given initial allele frequency p0 """

    logP = recursiveBinomialN.loc[p0,kN]

    return logP


def likelihood_of_data( data, recursiveBinomialN, recursiveBinomialX, allelabels, verbo=True ):

    """ Computes likelihood of each observed allele shift under a neutral evolution model """

    if (len(allelabels) == 2):
        p0id, pNid = allelabels
        pShiftId = pNid + "-" + p0id
    elif (len(allelabels) == 3):
        p0id, pNid, pShiftId = allelabels
    else:
        raise ValueError("wrong nubmer of data labels")

    likelihood = numpy.full(data.shape[0], numpy.NaN)


    print( ">>>    sample | initial | final | likelihood\n" ) if (verbo==True) else None

    for i, ind in enumerate(data.index):  # loop over all polymorphisms

        p0 = numpy.round( data.loc[ind, p0id], 2)
        pN = data.loc[ind, pNid]
        pshift = data.loc[ind, pShiftId]

        driftibution = recursiveBinomialX if (data.loc[ind].Chrom=="X") else recursiveBinomialN  # if polymorphism is on chromosome X use drift calculations with appropriate effective population size
        _2N = driftibution.shape[1] - 1

        freqrange2N = numpy.arange(0, 1+(1/(_2N)), 1/(_2N))  # frequency range considering the possible number of alleles, i.e. 1/2N increments (or 1/1.5N for X)

        kN = numpy.NaN if numpy.isnan(pN) else list( numpy.abs( freqrange2N - pN) ).index( numpy.min( numpy.abs( freqrange2N - pN ) ) )  # get number of alleles corresponding to frequency in generation 75, i.e. round to closest value considering 2Ne

        if numpy.isnan(pshift):
            likelii = numpy.NaN
        elif 0.01 <= p0 <= 0.99:
            likelii = drift_likelihood(p0, kN, driftibution)  # compute likelihood of observed frequency given initial frequency inder Wright-Fisher model
        elif ( (p0==0) and (pshift==0) ):
            likelii = 1  # if allele is lost (or as in next case, fixed) the likelihood of remaining so is 1
        elif ( (p0==1) and (pshift==0) ):
            likelii = 1
        elif (p0==0):
            likelii = drift_likelihood(0.01, kN, driftibution)  # if allele seems fixed/lost but frequency shifts, assume highest/lowest segregating frequency and compute likelihood under drift model
        elif (p0==1):
            likelii = drift_likelihood(0.99, kN, driftibution)
        else:
            raise ValueError("it seems like your frequency value is either lower than zero or greater than one")

        likelihood[i] = likelii

        if not (i % 1):  # print results every 1000th polymorphism
            print( ">>>    " + str(i) + "  |  " + str( round(p0, 2) )  + "  |  " + str( round(pN, 2) )  + "  |  " + str( likelihood[i] ) ) if (verbo==True) else None

    print( ">>>    sample | initial | final | likelihood\n" ) if (verbo==True) else None

    return likelihood


def line_likelihoods(metadata):

    """ Computes total likelihood per line """

    explines = list( set( metadata.Line ) )  # gets set of experimental lines from metadata
    explines.sort()

    logP = numpy.full( len( set( metadata.Line ) ), numpy.NaN)  # creates empyt array for likelihoods of each line

    for i,line in enumerate(explines):

        logP[i] = numpy.sum( numpy.log( metadata[ metadata.Line==line ].likelihood ) )  # take the log of the likelihoods and compute their sum for each line

        print(">>> line:", line, ", likelihood:", logP[i])

    return pandas.DataFrame(logP, index=explines, columns=['logP'])


def add_selrep_from_line(data, verbo=True):

    if (('sel' in variantdata.columns) and ('rep' in variantdata.columns)):
        print(">>>  'sel' and 'rep' columns already in data frame...") if (verbo==True) else None

    if 'sel' not in data.columns:
        print(">>> adding 'sel' columns to data frame...") if (verbo==True) else None

        data['sel'] = ""
        data.loc[ ((data.Line=="C1") | (data.Line=="C2")), "sel" ] = "control"
        data.loc[ ((data.Line=="S1") | (data.Line=="S2")), "sel" ] = "short"
        data.loc[ ((data.Line=="L1") | (data.Line=="L2")), "sel" ] = "long"

    if 'rep' not in data.columns:
        print(">>> adding 'rep' columns to data frame...") if (verbo==True) else None

        data['rep'] = 0
        data.loc[ ((data.Line=="C1") | (data.Line=="S1") | (data.Line=="L1")), "rep" ] = int(1)
        data.loc[ ((data.Line=="C2") | (data.Line=="S2") | (data.Line=="L2")), "rep" ] = int(2)

    return data


def split_by_generation(alldata, generati0N, dontsplit=['Chrom', 'Pos', 'RefBase', 'VarBase', 'Line', 'sel', 'rep'], method='inner', savefile="", verbo=True):

    """ splits long table with frequencies of each entry for either generation 12 or 75 into a wide table where each row has data on both generations 12 and 75. Join method "inner" eliminates entries where either frequency is missing """

    print(">>> splitting table to align initial and final generations") if (verbo==True) else None
    g0, gN = generati0N

    all0 = alldata.loc[ alldata.Generation==g0 ]
    allN = alldata.loc[ alldata.Generation==gN ]

    alljoin = pandas.merge(all0, allN, on=dontsplit, how=method)  # 'Minor_Allele', 'N Rows'

    newcolumns = []  # create new column names to replace automatically generated labels from split/join operations
    for j,label in enumerate(alljoin.columns):
        print(">>> label:", label) if (verbo==True) else None
        if label[-1] == 'x':
            newcolumns.append( alljoin.columns[j][:-2] + str(g0) )
        elif label[-1] == 'y':
            newcolumns.append( alljoin.columns[j][:-2] + str(gN) )
        else:
            newcolumns.append( alljoin.columns[j] )

    # rename columns with generation-labeled ones
    alljoin.columns = newcolumns  # replace auto-generated column names with new ones

    if (savefile != ""):
        alljoin.to_csv( savefile, sep=',', header=True, index=0 )

    return alljoin


def subset_alleles(cmhdata, alldata, allelabels, repetitions=1000, savefile="", verbo=True):
    """ Sample a number of random set of non-significant alleles the same size as the CMH-significant set """

    print(">>> picking random sets of alleles in matching chromosomes") if (verbo==True) else None
    if (len(allelabels) == 2):
        maf0, mafN = allelabels
        maf0N = pNid + "-" + p0id
    elif (len(allelabels) == 3):
        maf0, mafN, maf0N = allelabels
    else:
        raise ValueError("wrong nubmer of data labels")

    chromnumbers = pandas.DataFrame( index=set(alldata.Chrom), columns=["CMH", "all"] )  # create empty data frame to store number of alleles for each chromosome arm

    for arm in set(cmhdata.Chrom):
        chromnumbers.loc[arm, "CMH"] = cmhdata[ cmhdata.Chrom==arm ].shape[0]  # get number of significant alleles in each chromosome arm

        chromnumbers.loc[arm, "all"] = repetitions*chromnumbers.loc[arm, "CMH"] # number of alleles from chormosome arm in random set is number of random sets times number computed above

    if "Line" not in cmhdata.columns:
        cmhdata["Line"] = "neutral"
        alldata["Line"] = "neutral"

    subpositions = []
    nlines = len(set(cmhdata.Line))

    for arm in set(cmhdata.Chrom):
        print(">>> Chromosome:", arm) if (verbo==True) else None
        for linn in set(cmhdata.Line):
            print(">>> Line:", linn) if (verbo==True) else None

            nSNPs = int( chromnumbers.loc[arm, "all"]/nlines )

            lineChromIndices = alldata[ ( (alldata.Chrom==arm) & (alldata.Line==linn) & ( numpy.isfinite( alldata[maf0] ) ) & numpy.isfinite( alldata[mafN] ) ) ].index  # list all indices with the arm/line metadata

            subpositions.extend( list (choice( lineChromIndices, nSNPs, replace=False ) ) )  # choose random sample from indices with the metadata bove


    subjoin = alldata.loc[subpositions]

    subjoin['repetition'] = numpy.NaN
    for arm in set(alldata.Chrom):
        subjoin.loc[ subjoin.Chrom==arm, 'repetition' ] = [int(l) for l in range(1,repetitions+1) for _ in range(chromnumbers.loc[arm, "CMH"])]


    subjoin[maf0N] = subjoin[mafN] - subjoin[maf0]

    if (savefile != ""):
        subjoin.to_csv(savefile, sep=',', header=True, index=True)

    return subjoin
