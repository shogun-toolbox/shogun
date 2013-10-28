
#from matplotlib import rc
#rc('text', usetex=True)

fontsize = 16
contourFontsize = 12
showColorbar = False
xmin = -1
xmax = 1
ymin = -1.05
ymax = 1

import sys,os

import numpy
import shogun
from shogun.Kernel import GaussianKernel, LinearKernel, PolyKernel
from shogun.Features import RealFeatures, BinaryLabels
from shogun.Classifier import LibSVM

from numpy import arange
import matplotlib
from matplotlib import pylab
pylab.rcParams['contour.negative_linestyle'] = 'solid'

def features_from_file(fileName) :

    fileHandle = open(fileName)
    fileHandle.readline()
    features = []
    labels = []
    for line in fileHandle :
        tokens = line.split(',')
        labels.append(float(tokens[1]))
        features.append([float(token) for token in tokens[2:]])

    return RealFeatures(numpy.transpose(numpy.array(features))), features, BinaryLabels(numpy.array(labels,numpy.float))

def create_kernel(kname, features, kparam=None) :

    if kname == 'gauss' :
        kernel = GaussianKernel(features, features, kparam)
    elif kname == 'linear':
        kernel = LinearKernel(features, features)
    elif kname == 'poly' :
        kernel = PolyKernel(features, features, kparam, True, False)

    return kernel


def svm_train(kernel, labels, C1, C2=None):
    """Trains a SVM with the given kernel"""

    num_threads = 1

    kernel.io.disable_progress()
    svm = LibSVM(C1, kernel, labels)
    if C2:
        svm.set_C(C1, C2)
    svm.parallel.set_num_threads(num_threads)
    svm.io.disable_progress()
    svm.train()

    return svm

def svm_test(svm, kernel, features_train, features_test) :
    """predicts on the test examples"""

    kernel.init(features_train, features_test)
    output = svm.apply().get_labels()

    return output


def decision_boundary_plot(svm, features, vectors, labels, kernel, fileName = None, **args) :

    title = None
    if 'title' in args :
        title = args['title']
    xlabel = None
    if 'xlabel' in args :
        xlabel = args['xlabel']
    ylabel = None
    if 'ylabel' in args :
        ylabel = args['ylabel']
    fontsize = 'medium'
    if 'fontsize' in args :
        fontsize = args['fontsize']
    contourFontsize = 10
    if 'contourFontsize' in args :
        contourFontsize = args['contourFontsize']
    showColorbar = True
    if 'showColorbar' in args :
        showColorbar = args['showColorbar']
    show = True
    if fileName is not None :
        show = False
    if 'show' in args :
        show = args['show']


    # setting up the grid
    delta = 0.005
    x = arange(xmin, xmax, delta)
    y = arange(ymin, ymax, delta)

    Z = numpy.zeros((len(x), len(y)), numpy.float_)
    gridX = numpy.zeros((len(x) *len(y), 2), numpy.float_)
    n = 0
    for i in range(len(x)) :
        for j in range(len(y)) :
            gridX[n][0] = x[i]
            gridX[n][1] = y[j]
            n += 1

    if kernel.get_name() == 'Linear' and 'customwandb' in args:
        kernel.init_optimization_svm(svm)
        b=svm.get_bias()
        w=kernel.get_w()
        kernel.set_w(args['customwandb'][0])
        svm.set_bias(args['customwandb'][1])

    if kernel.get_name() == 'Linear' and 'drawarrow' in args:
        kernel.init_optimization_svm(svm)
        b=svm.get_bias()
        w=kernel.get_w()
        s=1.0/numpy.dot(w,w)/1.17
        pylab.arrow(0,-b/w[1], w[0]*s,s*w[1], width=0.01, fc='#dddddd', ec='k')
    grid_features = RealFeatures(numpy.transpose(gridX))
    results = svm_test(svm, kernel, features, grid_features)

    n = 0
    for i in range(len(x)) :
        for j in range(len(y)) :
            Z[i][j] = results[n]
            n += 1

    cdict = {'red'  :((0.0, 0.6, 0.6),(0.5, 0.8, 0.8),(1.0, 1.0, 1.0)),
             'green':((0.0, 0.6, 0.6),(0.5, 0.8, 0.8),(1.0, 1.0, 1.0)),
             'blue' :((0.0, 0.6, 0.6),(0.5, 0.8, 0.8),(1.0, 1.0, 1.0)),
             }
    my_cmap = matplotlib.colors.LinearSegmentedColormap('lightgray',cdict,256)
    im = pylab.imshow(numpy.transpose(Z),
                      interpolation='bilinear', origin='lower',
                      cmap=my_cmap, extent=(xmin,xmax,ymin,ymax) )

    if 'decisionboundaryonly' in args:
        C1 = pylab.contour(numpy.transpose(Z),
                [0],
                origin='lower',
                linewidths=(3),
                colors = ['k'],
                extent=(xmin,xmax,ymin,ymax))
    else:
        C1 = pylab.contour(numpy.transpose(Z),
                [-1,0,1],
                origin='lower',
                linewidths=(1,3,1),
                colors = ['k','k'],
                extent=(xmin,xmax,ymin,ymax))

        pylab.clabel(C1,
                inline=1,
                fmt='%1.1f',
                fontsize=contourFontsize)

    # plot the data
    lab=labels.get_labels()
    vec=numpy.array(vectors)
    idx=numpy.where(lab==-1)[0]
    pylab.scatter(vec[idx,0], vec[idx,1], s=300, c='#4444ff', marker='o', alpha=0.8, zorder=100)
    idx=numpy.where(lab==+1)[0]
    pylab.scatter(vec[idx,0], vec[idx,1], s=500, c='#ff4444', marker='s', alpha=0.8, zorder=100)

    # plot SVs
    if not 'decisionboundaryonly' in args:
        training_outputs = svm_test(svm, kernel, features, features)
        sv_idx=numpy.where(abs(training_outputs)<=1.01)[0]
        pylab.scatter(vec[sv_idx,0], vec[sv_idx,1], s=100, c='k', marker='o', alpha=0.8, zorder=100)

    if 'showmovedpoint' in args:
        x=-0.779838709677
        y=-0.1375
        pylab.scatter([x], [y], s=300, c='#4e4e61', marker='o', alpha=1, zorder=100, edgecolor='#454548')
        pylab.arrow(x,y-0.1, 0, -0.8/1.5, width=0.01, fc='#dddddd', ec='k')
        #pylab.show()


    if title is not None :
        pylab.title(title, fontsize=fontsize)
    if ylabel:
        pylab.ylabel(ylabel,fontsize=fontsize)
    if xlabel:
        pylab.xlabel(xlabel,fontsize=fontsize)
    if showColorbar :
        pylab.colorbar(im)

    # colormap:
    pylab.hot()
    if fileName is not None :
        pylab.savefig(fileName)
    if show :
        pylab.show()

def add_percent_ticks():
    ticks=pylab.getp(pylab.gca(),'xticks')
    ticklabels=len(ticks)*['']
    ticklabels[0]='0%'
    ticklabels[-1]='100%'
    pylab.setp(pylab.gca(), xticklabels=ticklabels)

    pylab.setp(pylab.gca(), yticklabels=['0%','100%'])
    ticks=pylab.getp(pylab.gca(),'yticks')
    ticklabels=len(ticks)*['']
    #ticklabels[0]='0%'
    ticklabels[-1]='100%'
    pylab.setp(pylab.gca(), yticklabels=ticklabels)

    xticklabels = pylab.getp(pylab.gca(), 'xticklabels')
    yticklabels = pylab.getp(pylab.gca(), 'yticklabels')
    pylab.setp(xticklabels, fontsize=fontsize)
    pylab.setp(yticklabels, fontsize=fontsize)



def create_figures(extension = 'pdf', directory = '../../tex/figures') :

    if extension[0] != '.' :
        extension = '.' + extension

    dpi=90

    # data and linear decision boundary
    features,vectors,labels = features_from_file('data/small_gc_toy.data')

    kernel = create_kernel('linear', features)
    svm = svm_train(kernel, labels, 0.7)

    pylab.figure(figsize=(8,6), dpi=dpi)
    decision_boundary_plot(svm, features, vectors, labels, kernel,
            fontsize=fontsize, contourFontsize=contourFontsize,
            title="Linear Separation", customwandb=(numpy.array([-0.05, -1.0]), -0.3),
            ylabel="GC Content Before 'AG'",xlabel="GC Content After 'AG'",
            show=False, showColorbar=showColorbar, decisionboundaryonly=True)
    add_percent_ticks()

    pylab.savefig(os.path.join(directory, 'data_and_linear_classifier' + extension))
    pylab.close()
#####################################################################################
    # data and svm decision boundary
    features,vectors,labels = features_from_file('data/small_gc_toy.data')

    kernel = create_kernel('linear', features)
    svm = svm_train(kernel, labels, 100)

    pylab.figure(figsize=(8,6), dpi=dpi)
    decision_boundary_plot(svm, features, vectors, labels, kernel,
            fontsize=fontsize, contourFontsize=contourFontsize,
            title="Maximum Margin Separation", drawarrow=True,
            ylabel="GC Content Before 'AG'",xlabel="GC Content After 'AG'",
            show=False, showColorbar=showColorbar)
    add_percent_ticks()

    pylab.savefig(os.path.join(directory, 'data_and_svm_classifier' + extension))
    pylab.close()
#####################################################################################


    # the effect of C on the decision surface:
    features,vectors,labels = features_from_file('data/small_gc_toy_outlier.data')

    pylab.figure(figsize=(16,6), dpi=dpi)
    pylab.subplot(121)
    kernel = create_kernel('linear', features)
    svm = svm_train(kernel, labels, 200)
    decision_boundary_plot(svm, features, vectors, labels, kernel,
            title = 'Soft-Margin with C=200', ylabel="GC Content Before 'AG'",
            xlabel="GC Content After 'AG'", fontsize=fontsize,
            contourFontsize=contourFontsize, show=False, showmovedpoint=True,
            showColorbar=showColorbar)
    add_percent_ticks()

    pylab.subplot(122)
    kernel = create_kernel('linear', features)
    svm = svm_train(kernel, labels, 2)
    decision_boundary_plot(svm, features, vectors, labels, kernel,
            title = 'Soft-Margin with C=2',
            ylabel="GC Content Before 'AG'",xlabel="GC Content After 'AG'",
            fontsize=fontsize, contourFontsize=contourFontsize, show=False, showColorbar=showColorbar)
    add_percent_ticks()
    #pylab.subplots_adjust(bottom=0.05, top=0.95)

    pylab.savefig(os.path.join(directory, 'effect_of_c' + extension))
    pylab.close()
####################################################################################

    # playing with nonlinear data:
    # the effect of kernel parameters

    features,vectors,labels = features_from_file('data/small_gc_toy_outlier.data')
    pylab.figure(figsize=(24,6), dpi=dpi)
    pylab.subplot(131)
    kernel = create_kernel('linear', features)
    svm = svm_train(kernel, labels, 100)
    decision_boundary_plot(svm, features, vectors, labels, kernel,
            title = 'Linear Kernel',
            ylabel="GC Content Before 'AG'",
            fontsize=fontsize, contourFontsize=contourFontsize, show=False,showColorbar=showColorbar)
    add_percent_ticks()

    pylab.subplot(132)
    kernel = create_kernel('poly', features, 2)
    svm = svm_train(kernel, labels, 100)
    decision_boundary_plot(svm, features, vectors, labels, kernel,
            title='Polynomial Kernel d=2',
            xlabel="GC Content After 'AG'",
            fontsize=fontsize, contourFontsize=contourFontsize, show=False,showColorbar=showColorbar)
    add_percent_ticks()

    pylab.subplot(133)
    kernel = create_kernel('poly', features, 5)
    svm = svm_train(kernel, labels, 10)
    decision_boundary_plot(svm, features, vectors, labels, kernel,
            title='Polynomial Kernel d=5',
            fontsize=fontsize, contourFontsize=contourFontsize, show=False,showColorbar=showColorbar)
    add_percent_ticks()
    #pylab.subplots_adjust(bottom=0.05, top=0.95)

    pylab.savefig(os.path.join(directory, 'params_polynomial' + extension))
    pylab.close()
####################################################################################

    #effects of sigma
    pylab.figure(figsize=(24,6), dpi=dpi)
    pylab.subplot(131)
    gamma = 0.1
    sigma = 20.0
    kernel = create_kernel('gauss', features, sigma)
    svm = svm_train(kernel, labels, 100)
    decision_boundary_plot(svm, features, vectors, labels, kernel,
            title='Gaussian Kernel Sigma=20',
            ylabel="GC Content Before 'AG'",
            fontsize=fontsize, contourFontsize=contourFontsize, show=False,showColorbar=showColorbar)
    add_percent_ticks()

    pylab.subplot(132)
    sigma = 1.0
    kernel = create_kernel('gauss', features, sigma)
    svm = svm_train(kernel, labels, 100)
    decision_boundary_plot(svm, features, vectors, labels, kernel,
            title='Gaussian Kernel Sigma=1',
            xlabel="GC Content After 'AG'",
            fontsize=fontsize, contourFontsize=contourFontsize, show=False,showColorbar=showColorbar)
    add_percent_ticks()

    pylab.subplot(133)
    sigma = 0.05
    kernel = create_kernel('gauss', features, sigma)
    svm = svm_train(kernel, labels, 100)
    decision_boundary_plot(svm, features, vectors, labels, kernel,
            title='Gaussian Kernel Sigma=0.05',
            fontsize=fontsize, contourFontsize=contourFontsize, show=False,showColorbar=showColorbar)
    add_percent_ticks()

    #pylab.subplots_adjust(bottom=0.05, top=0.95)

    pylab.savefig(os.path.join(directory, 'params_gaussian' + extension))
    pylab.close()
####################################################################################

if __name__ == '__main__' :

    extension = 'pdf'
    if len(sys.argv) > 1 :
        extension = sys.argv[1]
    pylab.ioff()
    create_figures(extension)
