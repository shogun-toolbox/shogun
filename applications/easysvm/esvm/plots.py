"""
This module contains code for commonly used plots
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

import sys
import random
import numpy
import warnings
import shutil

from shogun.Features import Labels
from shogun.Evaluation import *

def plotroc(output, LTE, draw_random=False, figure_fname="", roc_label='ROC'):
    """Plot the receiver operating characteristic curve"""
    import pylab
    import matplotlib

    pylab.figure(1,dpi=150,figsize=(4,4))
    fontdict=dict(family="cursive",weight="bold",size=7,y=1.05) ;

    pm=PerformanceMeasures(Labels(numpy.array(LTE)), Labels(numpy.array(output)))

    points=pm.get_ROC()
    points=numpy.array(points).T # for pylab.plot
    pylab.plot(points[0], points[1], 'b-', label=roc_label)
    if draw_random:
        pylab.plot([0, 1], [0, 1], 'r-', label='random guessing')
    pylab.axis([0, 1, 0, 1])
    ticks=numpy.arange(0., 1., .1, dtype=numpy.float64)
    pylab.xticks(ticks,size=10)
    pylab.yticks(ticks,size=10)
    pylab.xlabel('1 - specificity (false positive rate)',size=10)
    pylab.ylabel('sensitivity (true positive rate)',size=10)
    pylab.legend(loc='lower right', prop = matplotlib.font_manager.FontProperties('tiny'))

    if figure_fname!=None:
        warnings.filterwarnings('ignore','Could not match*')
        tempfname = figure_fname + '.png'
	pylab.savefig(tempfname)
	shutil.move(tempfname,figure_fname)

    auROC=pm.get_auROC()
    return auROC ;

def plotprc(output, LTE, figure_fname="", prc_label='PRC'):
    """Plot the precision recall curve"""
    import pylab
    import matplotlib

    pylab.figure(2,dpi=150,figsize=(4,4))

    pm=PerformanceMeasures(Labels(numpy.array(LTE)), Labels(numpy.array(output)))

    points=pm.get_PRC()
    points=numpy.array(points).T # for pylab.plot
    pylab.plot(points[0], points[1], 'b-', label=prc_label)
    pylab.axis([0, 1, 0, 1])
    ticks=numpy.arange(0., 1., .1, dtype=numpy.float64)
    pylab.xticks(ticks,size=10)
    pylab.yticks(ticks,size=10)
    pylab.xlabel('sensitivity (true positive rate)',size=10)
    pylab.ylabel('precision (1 - false discovery rate)',size=10)
    pylab.legend(loc='lower right')

    if figure_fname!=None:
        warnings.filterwarnings('ignore','Could not match*')
        tempfname = figure_fname + '.png'
	pylab.savefig(tempfname)
	shutil.move(tempfname,figure_fname)

    auPRC=pm.get_auPRC()
    return auPRC ;

def plotcloud(cloud, figure_fname="", label='cloud'):
    """Plot the cloud of points (the first two dimensions only)"""
    import pylab
    import matplotlib

    pylab.figure(1,dpi=150,figsize=(4,4))

    pos = []
    neg = []
    for i in xrange(len(cloud)):
        if cloud[i][0]==1:
            pos.append(cloud[i][1:])
        elif cloud[i][0]==-1:
            neg.append(cloud[i][1:])

    fontdict=dict(family="cursive",weight="bold",size=10,y=1.05) ;
    pylab.title(label, fontdict)
    points=numpy.array(pos).T # for pylab.plot
    pylab.plot(points[0], points[1], 'b+', label='positive')
    points=numpy.array(neg).T # for pylab.plot
    pylab.plot(points[0], points[1], 'rx', label='negative')
    #pylab.axis([0, 1, 0, 1])
    #ticks=numpy.arange(0., 1., .1, dtype=numpy.float64)
    #pylab.xticks(ticks,size=10)
    #pylab.yticks(ticks,size=10)
    pylab.xlabel('dimension 1',size=10)
    pylab.ylabel('dimension 2',size=10)
    pylab.legend(loc='lower right')

    if figure_fname!=None:
        warnings.filterwarnings('ignore','Could not match*')
        tempfname = figure_fname + '.png'
	pylab.savefig(tempfname)
	shutil.move(tempfname,figure_fname)

def plot_poims(poimfilename, poim, max_poim, diff_poim, poim_totalmass, poimdegree, max_len):
    """Plot a summary of the information in poims"""
    import pylab
    import matplotlib

    pylab.figure(3, dpi=150, figsize=(4,5))

    # summary figures
    fontdict=dict(family="cursive",weight="bold",size=7,y=1.05) ;
    pylab.subplot(3,2,1)
    pylab.title('Total POIM Mass', fontdict)
    pylab.plot(poim_totalmass) ;
    pylab.ylabel('weight mass', size=5)

    pylab.subplot(3,2,3)
    pylab.title('POIMs', fontdict)
    pylab.pcolor(max_poim, shading='flat') ;

    pylab.subplot(3,2,5)
    pylab.title('Differential POIMs', fontdict)
    pylab.pcolor(diff_poim, shading='flat') ;

    for plot in [3, 5]:
        pylab.subplot(3,2,plot)
        ticks=numpy.arange(1., poimdegree+1, 1, dtype=numpy.float64)
        ticks_str = []
        for i in xrange(0, poimdegree):
            ticks_str.append("%i" % (i+1))
            ticks[i] = i + 0.5
        pylab.yticks(ticks, ticks_str)
        pylab.ylabel('degree', size=5)

    # per k-mer figures
    fontdict=dict(family="cursive",weight="bold",size=7,y=1.04) ;

    # 1-mers
    pylab.subplot(3,2,2)
    pylab.title('1-mer Positional Importance', fontdict)
    pylab.pcolor(poim[0], shading='flat') ;
    ticks_str = ['A', 'C', 'G', 'T']
    ticks = [0.5, 1.5, 2.5, 3.5]
    pylab.yticks(ticks, ticks_str, size=5)
    pylab.axis([0, max_len, 0, 4])

    # 2-mers
    pylab.subplot(3,2,4)
    pylab.title('2-mer Positional Importance', fontdict)
    pylab.pcolor(poim[1], shading='flat') ;
    i=0 ;
    ticks=[] ;
    ticks_str=[] ;
    for l1 in ['A', 'C', 'G', 'T']:
        for l2 in ['A', 'C', 'G', 'T']:
            ticks_str.append(l1+l2)
            ticks.append(0.5+i) ;
            i+=1 ;
    pylab.yticks(ticks, ticks_str, fontsize=5)
    pylab.axis([0, max_len, 0, 16])

    # 3-mers
    pylab.subplot(3,2,6)
    pylab.title('3-mer Positional Importance', fontdict)
    pylab.pcolor(poim[2], shading='flat') ;
    i=0 ;
    ticks=[] ;
    ticks_str=[] ;
    for l1 in ['A', 'C', 'G', 'T']:
        for l2 in ['A', 'C', 'G', 'T']:
            for l3 in ['A', 'C', 'G', 'T']:
                if numpy.mod(i,4)==0:
                    ticks_str.append(l1+l2+l3)
                    ticks.append(0.5+i) ;
                i+=1 ;
    pylab.yticks(ticks, ticks_str, fontsize=5)
    pylab.axis([0, max_len, 0, 64])

    # x-axis on last two figures
    for plot in [5, 6]:
        pylab.subplot(3,2,plot)
        pylab.xlabel('sequence position', size=5)


    # finishing up
    for plot in xrange(0,6):
        pylab.subplot(3,2,plot+1)
        pylab.xticks(fontsize=5)

    for plot in [1,3,5]:
        pylab.subplot(3,2,plot)
        pylab.yticks(fontsize=5)

    pylab.subplots_adjust(hspace=0.35) ;

    # write to file
    warnings.filterwarnings('ignore','Could not match*')
    pylab.savefig('/tmp/temppylabfig.png')
    shutil.move('/tmp/temppylabfig.png',poimfilename)

