#!/usr/bin/env python

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

import bz2
import time
import sys
from splicesites.utils import create_dataset, create_modsel
from esvm.utils import calcroc
from esvm.experiment import crossvalidation
from esvm.mldata import init_datasetfile
from numpy.linalg import norm
import numpy

def write_results(f, results):
    """
    Write out results
    """

    f.write('Kernel\tParameters\tC\tauROC\n');
    for i in xrange(len(results)):
        C=results[i][0]
        k_param=results[i][1]
        param_name=k_param[1]['name']
        kernel=k_param[0]
        if kernel.endswith('2'):
            kernel=kernel[:-1]
        kernel_parameters= param_name + '=' + `k_param[1][param_name]`
        perf = 100*results[i][2]

        f.write('%s\t' % kernel)
        f.write('%s\t' % kernel_parameters)
        f.write('C=%2.2f\t' % C)
        f.write('%2.1f%%\n' % perf)

def normalize(examples, subtract_mean=False, divide_std=False, rescale=False, norm_one=False):
    """
    Scale GC data to ... (be on a ball? just const? 0 mean, std 1?)
    """

    if subtract_mean:
        # mean = 0.0
        mean=numpy.mean(examples, axis=1)
        for i in xrange(examples.shape[1]):
            examples[:,i]-=mean

    if divide_std:
        # std = 1.0
        std=numpy.std(examples, axis=1)
        for i in xrange(examples.shape[1]):
            examples[:,i]/=(std+1e-10)

    if rescale:
        # scale to have on average 1 on linear kernel diagonal
        scale=numpy.sqrt(numpy.mean(numpy.diag(numpy.mat(examples).T*numpy.mat(examples))))
        examples/=scale

    if norm_one:
        # ball/circle
        for i in xrange(examples.shape[1]):
            examples[:,i]/=norm(examples[:,i])

    return examples ;

def run_single_experiment(results, num_fold_cv, kernelname, kparam, C, examples, labels):
    """
    Run a single experiment, i.e. for a fixed kernel and parameters
    do num_fold cross-validation
    """

    param_name=kparam['name']
    print 'Running C =', C, kernelname.title(), 'Kernel with', param_name, '=', kparam[param_name]
    (all_outputs, all_split) = crossvalidation(num_fold_cv, kernelname, kparam, C, examples, labels, 'dna', 'A')
    results.append( (C, (kernelname, kparam), calcroc(all_outputs,labels)) )

def splice_example(Cs, gcfilename,seqfilename,seq2filename, plot=False):
    """
    For the data files, apply the set of kernels
    """
    # hyperparameters
    num_fold_cv = 5

    # The area under the receiver operating characteristic
    results=[]

    # Read datasets

    # GC features
    fp = init_datasetfile(gcfilename,'vec')
    (gc_examples,gc_labels) = fp.readlines()
    gc_examples = normalize(gc_examples, subtract_mean=True)

    if plot:
        from pylab import scatter,show
        color=['b','r']
        scatter(gc_examples[0,], gc_examples[1,], s=400*(gc_labels+2), c=''.join([ color[(int(i)+1)/2] for i in gc_labels]), alpha=0.1)
        show()

    # 2 sequence features
    fp = init_datasetfile(seq2filename,'mseq')
    (dna2_examples,dna2_labels) = fp.readlines()

    # DNA sequences
    fp = init_datasetfile(seqfilename,'seq')
    (dna_examples,dna_labels) = fp.readlines()


    #Define experiments to carry out

    experiments=(
    # Linear kernel on GC content
    ('linear', {'scale':1.0, 'name':'scale'}, (gc_examples, gc_labels)),

    # Polynomial kernel on GC content
    ( 'poly', {'degree':3, 'name':'degree', 'inhomogene':True, 'normal':True}, (gc_examples, gc_labels)),
    ( 'poly', {'degree':5, 'name':'degree', 'inhomogene':True, 'normal':True}, (gc_examples, gc_labels)),

    # Gaussian kernel on GC content
    ('gauss', {'width':100.0, 'name':'width'}, (gc_examples, gc_labels)),
    ('gauss', {'width':1.0, 'name':'width'}, (gc_examples, gc_labels)),
    ('gauss', {'width':0.01, 'name':'width'}, (gc_examples, gc_labels)),

    # Spectrum kernel on 2 dna sequences
    ('spec2', {'degree':1, 'name':'degree'}, (dna2_examples, dna2_labels)),
    ('spec2', {'degree':3, 'name':'degree'}, (dna2_examples, dna2_labels)),
    ('spec2', {'degree':5, 'name':'degree'}, (dna2_examples, dna2_labels)),

    # Cumulative Spectrum kernel on 2 dna sequences
    ('cumspec2', {'degree':1, 'name':'degree'}, (dna2_examples, dna2_labels)),
    ('cumspec2', {'degree':3, 'name':'degree'}, (dna2_examples, dna2_labels)),
    ('cumspec2', {'degree':5, 'name':'degree'}, (dna2_examples, dna2_labels)),

    # Weighted degree kernel on dna sequences
    ('wd', {'degree':1,'shift':0, 'name':'degree'}, (dna_examples, dna_labels)),
    ('wd', {'degree':3,'shift':0, 'name':'degree'}, (dna_examples, dna_labels)),
    ('wd', {'degree':5,'shift':0, 'name':'degree'}, (dna_examples, dna_labels))
    )


    if Cs is None:
        for C in (0.01, 0.1, 1, 2, 5, 10):
            for e in experiments:
                run_single_experiment(results, num_fold_cv, e[0], e[1], C, e[2][0], e[2][1])
    else:
        for i in xrange(len(experiments)):
            e=experiments[i]
            run_single_experiment(results, num_fold_cv, e[0], e[1], Cs[i], e[2][0], e[2][1])

    return results

def get_best_results(results):
    methods=('linear', 'poly', 'gauss', 'spec2', 'cumspec2', 'wd')
    best_result=[]
    for m in methods:
        params=set()
        for r in results:
            if r[1][0]==m:
                params.add(tuple(zip(r[1][1].keys(),r[1][1].values())))

        for p in params:
            m_result=0.0
            m_best=None
            for r in results:
                if r[1][0]==m and r[1][1]==dict(p) and r[2]>m_result:
                    m_result=r[2]
                    m_best=r
            best_result.append(m_best)
    return best_result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create_data':
            create_dataset()
            sys.exit(0)
        elif sys.argv[1] == 'mselect':
            results = splice_example(None, 'C_elegans_acc_modsel_gc.csv','C_elegans_acc_modsel_seq.csv','C_elegans_acc_modsel_seq2.csv')
            #results = splice_example(None, 'C_elegans_acc_gc.csv','C_elegans_acc_seq.csv','C_elegans_acc_seq2.csv')
            import pickle
            pickle.dump(results, file('mselect_result.pickle','w'))
            sys.exit(0)
        elif sys.argv[1] == 'get_best':
            import pickle
            results=pickle.load(file('mselect_result.pickle'))
            best_result=get_best_results(results)
            write_results(sys.stdout, best_result)

            #print 'Cs=[',
            #for e in best_result:
            #    print e[0], ",",
            #print ']'
            sys.exit(0)
        else:
            print "unknown argument"
            sys.exit(1)

    # without any argument
    starttime = time.time()

    Cs = [ 5, 10, 10, \
           5, 0.01, 10, \
           10, 10, 0.01, \
           10, 10, 10, \
           1, 1, 2 ]

    # run the experiment
    results = splice_example(Cs, 'C_elegans_acc_gc.csv','C_elegans_acc_seq.csv','C_elegans_acc_seq2.csv', False)

    stoptime = time.time()
    elapsedtime = time.strftime('Elapsed time (HH.MM:SS): %H.%M:%S',time.gmtime(stoptime-starttime))
    print elapsedtime

    write_results(file('results.txt','w'), results)
    for curline in file('results.txt').readlines():
        print curline.strip()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create_data':
            create_dataset()
        elif sys.argv[1] == 'create_modsel':
            create_modsel()
            sys.exit()
    #main()
    print 'results in results.txt'
