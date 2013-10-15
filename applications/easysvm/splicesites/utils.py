import random
import bz2
import numpy
from numpy import array, where, concatenate
from numpy import kron, ones, sqrt, sum
from os.path import exists
from esvm.mldata import convert
try:
    import arff
    have_arff = True
except ImportError:
    have_arff = False



def create_dataset():
    """Read the file with first 100k sequences from C. elegans
    and generate some easier datasets.
    """

    if not have_arff:
        print 'import arff failed, currently cannot create data'
        return

    # convert data to arff format
    gen_arff('C_elegans_acc_100000.fasta.bz2','C_elegans_acc_gc.arff','C_elegans_acc_seq.arff',\
             'C_elegans_acc_seq2.arff','C_elegans_acc_freq.arff',\
             num_seqs=100000,subset=True,overwrite=True,normalise=False,\
             max_pos=200,max_neg=2000)

    print 'Convert from arff to csv and fasta'
    convert('C_elegans_acc_gc.arff','C_elegans_acc_gc.csv','vec')
    convert('C_elegans_acc_seq.arff','C_elegans_acc_seq.csv','seq')
    convert('C_elegans_acc_freq.arff','C_elegans_acc_freq.csv','vec')
    convert('C_elegans_acc_seq2.arff','C_elegans_acc_seq2.csv','mseq')
    convert('C_elegans_acc_seq.arff','C_elegans_acc_seq.fa','seq')


def create_modsel():
    """Read the file with last 100k sequences from C. elegans
    and generate some easier datasets.
    """

    if not have_arff:
        print 'import arff failed, currently cannot create data'
        return

    # convert data to arff format
    gen_arff('C_elegans_acc_modsel.fasta.bz2','C_elegans_acc_modsel_gc.arff','C_elegans_acc_modsel_seq.arff',\
             'C_elegans_acc_modsel_seq2.arff','C_elegans_acc_modsel_freq.arff',\
             num_seqs=100000,subset=True,overwrite=True,normalise=False,\
             max_pos=200,max_neg=2000)

    print 'Convert from arff to csv and fasta'
    convert('C_elegans_acc_modsel_gc.arff','C_elegans_acc_modsel_gc.csv','vec')
    convert('C_elegans_acc_modsel_seq.arff','C_elegans_acc_modsel_seq.csv','seq')
    convert('C_elegans_acc_modsel_freq.arff','C_elegans_acc_modsel_freq.csv','vec')
    convert('C_elegans_acc_modsel_seq2.arff','C_elegans_acc_modsel_seq2.csv','mseq')
    convert('C_elegans_acc_modsel_seq.arff','C_elegans_acc_modsel_seq.fa','seq')


def gen_arff(fastafilename,gcfilename,seqfilename,seq2filename,specfilename,\
             num_seqs=100000,subset=False,max_pos=200,max_neg=2000,\
             overwrite=False,normalise=True):
    """If data not yet created, generate 2 arff files
    - containing the two dimensional GC content before and after splice site
    - containing the sequence around the splice site.
    """
    if (exists(gcfilename) and exists(seqfilename)) and not overwrite:
        return

    print 'Creating %s and %s from %s' % (gcfilename,seqfilename,fastafilename)

    if fastafilename.find('acc')!= -1:
        # acceptor, AG at [40:42]
        window = (-40, 197, 42)
    elif fastafilename.find('don')!= -1:
        # donor, GT or GC at [40:42]
        window = (-40, 200, 42)
    else:
        print "Error: Cannot determine whether donor or acceptor"

    [strings, lab]=read_data(bz2.BZ2File(fastafilename), num_seqs, window)
    # Only a subset of the examples are used.
    if subset:
        [strings, lab] = take_subset(strings, lab, max_pos, max_neg)

    gcs=count_gs_and_cs(strings, (0, -window[0]), (-window[0]+2, -window[0]+2+window[2]))

    seq_upstream = []
    seq_downstream = []
    for curstr in strings:
        seq_upstream.append(curstr[0:-window[0]])
        seq_downstream.append(curstr[(-window[0]+2):(-window[0]+2+window[2])])
    seq_upstream = array(seq_upstream)
    seq_downstream = array(seq_downstream)

    spec_up = count_nt_freq(seq_upstream)
    spec_down = count_nt_freq(seq_downstream)

    if normalise:
        gcs = normalise_features(gcs)
        spec_up = normalise_features(spec_up)
        spec_down = normalise_features(spec_down)

    # sequence file
    alist = [('label',1,[]),('sequence',0,[])]
    f = open(seqfilename,'w')
    arff.arffwrite(f,alist,zip(lab,strings),name=fastafilename,comment='Converted from '+fastafilename)
    f.close()

    # 2 sequence file
    alist = [('label',1,[]),('upstream sequence',0,[]),('downstream sequence',0,[])]
    f = open(seq2filename,'w')
    arff.arffwrite(f,alist,zip(lab,seq_upstream,seq_downstream),\
                   name=fastafilename,comment='Converted from '+fastafilename)
    f.close()

    # gc contents
    alist = [('label',1,[]),('upstream',1,[]),('downstream',1,[])]
    data = []
    for ix,curlab in enumerate(lab):
        data.append((curlab,gcs[0,ix],gcs[1,ix]))
    f = open(gcfilename,'w')
    arff.arffwrite(f,alist,data,name=fastafilename,comment='Converted from '+fastafilename)
    f.close()

    # spectrum
    alist = [('label',1,[]),\
             ('upA',1,[]),('upC',1,[]),('upG',1,[]),('upT',1,[]),\
             ('downA',1,[]),('downC',1,[]),('downG',1,[]),('downT',1,[])]
    data = []
    for ix,curlab in enumerate(lab):
        data.append((curlab,spec_up[0,ix],spec_up[1,ix],spec_up[2,ix],spec_up[3,ix],\
                     spec_down[0,ix],spec_down[1,ix],spec_down[2,ix],spec_down[3,ix]))
    if len(specfilename)>0:
        f = open(specfilename,'w')
        arff.arffwrite(f,alist,data,name=fastafilename,comment='Converted from '+fastafilename)
        f.close()


def take_subset(strings, lab, max_pos=200, max_neg=2000):
    """Take a subset of the classes to the maximum numbers determined by
    max_pos and max_neg
    """
    random.seed(123456789)

    pos_idx = where(lab>0)[0]
    neg_idx = where(lab<0)[0]
    num_pos = len(pos_idx)
    num_neg = len(neg_idx)

    assert(num_pos < num_neg)
    assert(max_pos < max_neg)

    max_pos = min(max_pos,num_pos)
    max_neg = min(max_neg,num_neg)

    neg_sub_idx = array(random.sample(neg_idx,max_neg))
    assert(all(lab[neg_sub_idx]<0))
    pos_sub_idx = array(random.sample(pos_idx,max_pos))
    assert(all(lab[pos_sub_idx]>0))

    strings = concatenate((strings[pos_sub_idx],strings[neg_sub_idx]))
    lab = concatenate((lab[pos_sub_idx],lab[neg_sub_idx]))

    return (strings,lab)

def balance_classes(strings, lab, max_examples=1200,ratio=5.0):
    """Take a subset of negative examples such that
    the number of examples in the negative class are limited to ratio.

    Also limit the maximum number of examples.
    """
    random.seed(123456789)

    pos_idx = where(lab>0)[0]
    neg_idx = where(lab<0)[0]
    num_pos = len(pos_idx)
    num_neg = len(neg_idx)
    assert(num_pos < num_neg)

    max_pos = int(float(max_examples)/(ratio+1.0))

    if num_pos < max_pos:
        max_pos = num_pos

    pos_idx = pos_idx[:max_pos]
    num_pos = len(pos_idx)
    max_neg = int(num_pos*ratio)
    if num_neg < max_neg:
        max_neg = num_neg

    sub_idx = array(random.sample(neg_idx,max_neg))
    assert(all(lab[sub_idx]<0))

    strings = concatenate((strings[pos_idx],strings[sub_idx]))
    lab = concatenate((lab[pos_idx],lab[sub_idx]))

    return (strings,lab)

def normalise_features(feats):
    """Normalise each feature to zero mean and unit variance.
    Assume features are column wise matrix.

    """
    (numdim,numex) = feats.shape

    M = sum(feats,axis=1)/numex
    M = M.reshape(numdim,1)

    M2 = sum(feats**2,axis=1)/numex
    M2 = M2.reshape(numdim,1)
    SD = sqrt(M2-M**2)
    onevec = ones((1,numex))
    feats = (feats - kron(onevec,M))/(kron(onevec,SD))

    return feats

def read_data(f, num, window):
    """Read the fasta file containing splice sites."""
    labels=num*[0]
    strings=num*[0]

    l1 = f.readline()
    l2 = f.readline()
    line = 0
    num_alt_consensus = 0
    while l1 and l2 and line<num:
        consensus = l2[:-1][window[1]:window[1]+2]
        if (consensus == 'AG') or (consensus == 'GT'):
            if 'label=-1' in l1:
                labels[line]=-1
            elif 'label=1' in l1:
                labels[line]=+1
            else:
                print "error in line %d" % line
                return

            strings[line] = l2[:-1][window[1]+window[0] : window[1]+window[2]]
            line+=1
        else:
            num_alt_consensus+=1
            if consensus != 'GC':
                print line, consensus

        l1=f.readline()
        l2=f.readline()

    print "Number of GC consensus sites: %d" %num_alt_consensus
    if line+num_alt_consensus!=num:
        print "error reading file"
        return
    else:
        strings = strings[:line+1]
        labels = labels[:line+1]
        return (array(strings), array(labels, dtype=numpy.double))

def count_gs_and_cs(strings, range1, range2):
    """Count the number of G and C in the two ranges."""
    num=len(strings)
    gc_count=num*[(0,0)]

    for i in xrange(num):
        x=float(strings[i].count('G', range1[0], range1[1]) +
                        strings[i].count('C', range1[0], range1[1])) / abs(range1[1]-range1[0])
        y=float(strings[i].count('G', range2[0], range2[1]) +
                        strings[i].count('C', range2[0], range2[1])) / abs(range2[1]-range2[0])
        gc_count[i]=(x,y)

    return array(gc_count).T


def count_nt_freq(strings):
    """Count the nucleotide frequencies"""
    num = len(strings)
    strlen = len(strings[0])
    ntfreq = num*[(0,0,0,0)]

    for ix in xrange(num):
        a=float(strings[ix].count('A')) / strlen
        c=float(strings[ix].count('C')) / strlen
        g=float(strings[ix].count('G')) / strlen
        t=float(strings[ix].count('T')) / strlen
        ntfreq[ix]=(a,c,g,t)

    return array(ntfreq).T

