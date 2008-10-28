import genomic
import numpy
import sys


params={}

if sys.argv[1]=='train':
    if len(sys.argv)!=5 and len(sys.argv)!=6:
        sys.stderr.write( "Usage: %s train pos.fa neg.fa [kernel.def] my.svm\n" % sys.argv[0])
        sys.exit(-1)

    fil = open(sys.argv[2])
    (params["positives"],positive_headers)=read_fasta_list(fil, [], [])
    print "read %i positive sequences" % len(params["positives"])

    fil = open(sys.argv[3])
    (params["negatives"],negative_headers)=read_fasta_list(fil, [], [])
    print "read %i negative sequences" % len(params["negatives"])

    # create sequence and label dicts
    train_labels = [] 
    train_seqs = []
    min_len = 1e10
    max_len = 0
    for p in xrange(0,len(params["positives"])):
        seq = params["positives"][p]
        if len(seq)>max_len: max_len=len(seq)
        if len(seq)<min_len: min_len=len(seq)
        train_seqs.append(seq)
        train_labels.append(1.0)
    for p in xrange(0,len(params["negatives"])):
        seq=params["negatives"][p]
        if len(seq)>max_len: max_len=len(seq)
        if len(seq)<min_len: min_len=len(seq)
        train_seqs.append(seq)
        train_labels.append(-1.0)

    winlen=max_len

    if len(sys.argv)==5:
        svm_file = sys.argv[4]
        sensors=[ 
            sensor((-100, winlen/2, 100), ("WDS", 20, 20)), #centered wds
            sensor((-winlen/2+1, winlen/2, 0), ("SPEC", 7, None)), #left spectrum
            sensor((0, winlen/2, winlen/2), ("SPEC", 7, None)) #right spectrum
            ] 
    else:
        svm_file = sys.argv[5]
        fil=open(sys.argv[4], "r")
        sensors=[]
        line = fil.readline() 
        while len(line)>0:
            elems=line.split('=')

            if elems[0]=='num_sensors':
                num_sensors=int(elems[1]) 
                for i in xrange(0, num_sensors):
                    sensor_string = fil.readline() 
                    s = sensor((None,None,None), (None,None,None))
                    s.from_string(sensor_string)
                    sensors.append(s)
            line = fil.readline() 
    
    a=arts(sensors, train_seqs, train_labels)
    a.train_svm(C=1, num_threads=16)
    a.save_svm(svm_file, train_seqs, train_labels)

elif sys.argv[1]=='test':
    if len(sys.argv)!=5 and len(sys.argv)!=6:
        sys.stderr.write( "Usage: %s test [-reverse] my.svm test.fa pred.out\n" % sys.argv[0])
        sys.exit(-1)
    winlen=2000

    if sys.argv[2]=='-reverse':
        strand='-' 
        svm_fname = sys.argv[3]
        test_fname = sys.argv[4]
        pred_fname = sys.argv[5]
    else:
        strand='+' 
        svm_fname = sys.argv[2]
        test_fname = sys.argv[3]
        pred_fname = sys.argv[4]
    
    a=arts(None, None, None)
    a.load_svm(svm_fname)
    a.optimize_svm()

    fil = open(test_fname)
    (strs,headers)=read_fasta_list(fil, [], [], False)
    print "read %i test sequences" % len(strs)

    fil=open(pred_fname, "w+")
    for contig in xrange(0,len(strs)):
        print "predicting "+headers[contig]+strand
            
        str=strs[contig]
        if strand=='-':
            str=reverse_complement(str)
    
        test_positions=xrange(2000,len(str)-2000)
            #for p in xrange(2000,len(str)-2000):
            #    if numpy.mod(p,1000000)==0:
            ##        print p
            #    s=str[(p-winlen/2):(p+winlen/2)]
            #    if not('N' in s):
            #        test_positions.append(p)
        str=str.replace('N', 'A')

        res = a.slide_svm_over_string(str, test_positions)

        if strand=='+':
            for i in xrange(0,len(test_positions)):
                fil.write( '%i %c %i %f\n' % (contig, strand, test_positions[i], res[i]) )
        else:
            for i in xrange(0,len(test_positions)):
                fil.write( '%i %c %i %f\n' % (contig, strand, len(str)-test_positions[i], res[i]) ) 
    fil.close()
    
else:
    sys.stderr.write( "Usage: %s train pos.fa neg.fa [kernels.def] my.svm\n       %s test [-reverse] my.svm test.fa pred.out\n" % (sys.argv[0], sys.argv[0]))
    sys.exit(-1)
