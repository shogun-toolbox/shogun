"""
This module contains code for computing
Position Oligomer Importance Matrices
"""

#############################################################################################
#                                                                                           #
#    This program is free software; you can redistribute it and/or modify                   #
#    it under the terms of the GNU General Public License as published by                   #
#    the Free Software Foundation; either version 3 of the License, or                      #
#    (at your option) any later version.                                                    #
#                                                                                           #
#    This program is distributed in the hope that it will be useful,                        #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of                         #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                           #
#    GNU General Public License for more details.                                           #
#                                                                                           #
#    You should have received a copy of the GNU General Public License                      #
#    along with this program; if not, see http://www.gnu.org/licenses                       #
#    or write to the Free Software Foundation, Inc., 51 Franklin Street,                    #
#    Fifth Floor, Boston, MA 02110-1301  USA                                                #
#                                                                                           #
#############################################################################################

import numpy
from numpy import ones


def compute_poims(svm, kernel, poimdegree, max_len):
    """For a trained SVM, compute Position Oligomer Importance Matrices"""

    distr = ones((max_len,4))/4 ;
    kernel.prepare_POIM2(distr)

    kernel.compute_POIM2(poimdegree, svm) ;
    poim = kernel.get_POIM2()
    kernel.cleanup_POIM2()

    (poim, max_poim, diff_poim) = reshape_normalize_contribs(poim, poimdegree, max_len)
    (poim_weightmass, poim_totalmass) = compute_weight_mass(poim, poimdegree, max_len)

    poim_totalmass=poim_totalmass/numpy.sum(poim_totalmass)

    return (poim, max_poim, diff_poim, poim_totalmass)


def compute_weight_mass(C, maxOrder, seqLen):

    mass=numpy.zeros((maxOrder, seqLen), numpy.double);
    total=numpy.zeros((1, seqLen), numpy.double);
    for i in xrange(0,maxOrder):
        mass[i,:] = sum(numpy.abs(C[i]))
    total = sum(mass);

    return (mass,total)

def getstringprobsMC(maxOrder,distrib,length, abcSize):

    pmatrix = []
    for k in xrange(0,maxOrder):
	pmatrix.append(ones(4^k,len))

        for l in xrange(0,len):
            for sigma in xrange(0, abcSize):
                prob = distrib(sigma,l);
                for k in xrange(0, maxOrder):
                    for relpos in xrange(0, min(k,l)):
                        vi = genindexvector_spos(k,sigma-1,relpos,abcSize);
                        pmatrix[k][vi,l-relpos+1] = pmatrix[k][vi,l-relpos+1]*prob;

    return pmatrix

def getV2_poimMC(u, strprobs, abcSize):
  VV = [];
  for k in xrange(0, len(u)):
      m = abcSize^k;
      VV.append( numpy.ones(4**(k+1),1)*mean(u[k]*strprobs[k] ) )

  return VV

def reshape_normalize_contribs(C, maxOrder, seqLen, opts={}):

    alphabetSize = 4;
    Contribs = [] ;
    l=0;
    for i in xrange(0, maxOrder):
        L = l + (alphabetSize**(i+1)) * seqLen;
        vec=C[l:L].copy() ;
        Contribs.append(vec.reshape( seqLen, alphabetSize**(i+1) ).T) ;
        l = L;

    assert( l == len(C) );

    if opts.has_key("distribution"):
        strprobs = getstringprobsMC(length(Contribs), opts["distribution"], seqLen, 4);
        MyV2 = getV2_poimMC(Contribs, strprobs, seqLen, 4);

        for i in xrange(0, maxOrder ):
            Contribs[i] = Contribs[i] -MyV2[i];

    if opts.has_key("background"):
        for i in xrange(0, maxOrder ):
            Contribs[i] = Contribs[i]*(opts["background"][i]!=0);

    maxContribs = numpy.zeros( (maxOrder, seqLen), numpy.double );
    maxp_org = numpy.zeros( (maxOrder, seqLen), numpy.double );
    maxp_str= numpy.zeros( (maxOrder, seqLen), numpy.int );
    for i in xrange(0, maxOrder ):
        con=numpy.abs(Contribs[i]) ;
	maxContribs[i,:] = numpy.max(con, axis=0)
	maxp_str[i,:] = numpy.argmax(con, axis=0)

    diffmaxContribs = numpy.zeros( (maxOrder, seqLen), numpy.double );

    for k in xrange(1, maxOrder ):
	numsy=4**(k+1);
	for l in  xrange(0, seqLen-k):
            km=maxp_str[k,l] ;
            A=numpy.abs(Contribs[k-1][numpy.floor(km/4),l]);
            B=numpy.abs(Contribs[k-1][numpy.mod(km,numsy/4),l+1]);
            #zA=numpy.mod(km,4)+1;
            #zB=numpy.floor(km/(numsy/4))+1;
            #correction=sum([A/distribution(zA, l+k-1), B/distribution(zB, l)]);
            correction=numpy.max([A, B]);
            diffmaxContribs[k,l] = maxContribs[k,l] - correction;

    return (Contribs, maxContribs, diffmaxContribs)

