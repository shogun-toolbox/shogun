import random

def permIter(seq):
    """Given some sequence 'seq', returns an iterator that gives
    all permutations of that sequence."""
    ## Base case
    if len(seq) == 1:
        yield(seq[0])
        raise StopIteration

    ## Inductive case
    for i in range(len(seq)):
        element_slice = seq[i:i+1]
        rest_iter = permIter(seq[:i] + seq[i+1:])
        for rest in rest_iter:
            yield(element_slice + rest)
    raise StopIteration

def xcombinations(items, n):
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for cc in xcombinations(items[:i]+items[i+1:],n-1):
                yield [items[i]]+cc

def xuniqueCombinations(items, n):
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for cc in xuniqueCombinations(items[i+1:],n-1):
                yield [items[i]]+cc
            
def xselections(items, n):
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for ss in xselections(items, n-1):
                yield [items[i]]+ss

def xpermutations(items):
    return xcombinations(items, len(items))

def comb(n, m):
	r=1;
	for i in xrange(n-m+1,n+1):
		r*=i
	for i in xrange(1,m+1):
		r/=i
	return r

def get_bit_string(v, width, (c0, c1)=('0','1')):
	s=''

	for i in xrange(width):
		if v & 1:
			s+=c1
		else:
			s+=c0
		v>>=1;

	return s[::-1]

def index_to_bits(c):
	v=0;
	for idx in c:
		v|= 1 << idx
	return v

def compute_coverage(mmasks, cmasks, v):
	for i in xrange(len(mmasks)):
		if ( mmasks[i] & (~v) == 0):
		#if (mmasks[i] & v) and ( mmasks[i] & (~v) == 0):
		#if (mmasks[i] & v):
			cmasks[i]+=1

	coverage=0
	for i in cmasks:
		if i:
			coverage+=1

	return coverage

def num_bits_on(v):
	n=0

	while v!=0:
		if (v & 1):
			n+=1
		v>>=1

	return n

def generate_mask(n, m, goodbits, mmask, cmasks, retries=30):
	cover_list=[]
	idx=range(len(cmasks))

# variant 1: consider only unused masks to or with
	for r in xrange(retries):
		l=n-goodbits
		mask=0
		for i in idx:
			if cmasks[i]==0:
				tmp= mask|mmasks[i]
				if num_bits_on(tmp)>l:
					break
				else:
					mask=tmp
		cvd=list(cmasks)
		cover_list.append((mask, compute_coverage(mmasks, cvd, mask)))
		random.shuffle(idx)

## variant 2: consider all masks to or with
#	for r in xrange(retries):
#		l=n-goodbits
#		mask=0
#		for i in idx:
#			tmp= mask|mmasks[i]
#			if num_bits_on(tmp)>l:
#				break
#			else:
#				mask=tmp
#		cvd=list(cmasks)
#		cover_list.append((mask, compute_coverage(mmasks, cvd, mask)))
#		random.shuffle(idx)
#
## variant 3: just choose n-goodbits bits
#
##	idx=range(n)
##	l=n-goodbits
##	for r in xrange(retries):
##		mask=index_to_bits(idx[:l])
##		cvd=list(cmasks)
##		cover_list.append((mask, compute_coverage(mmasks, cvd, mask)))
##		random.shuffle(idx)

	cover_list.sort(cmp=lambda x,y: x[1]-y[1], reverse=True)

	return cover_list[0][0]


if __name__ == '__main__':
	#import psyco
	#psyco.full()
	n=75;
	m=3;
	goodbits=20;
	retries=100;

	l=comb(n,m)
	mmasks=[ 0 for i in xrange(l) ]
	cmasks=[ 0 for i in xrange(l) ]
	masks=list()

	i=0
	for c in xuniqueCombinations(range(n),m):
		mmasks[i]=index_to_bits(c)
		i+=1

	coverage=0;
	while True:
		v = generate_mask(n, m, goodbits, mmasks, cmasks, retries)
		masks.append(v)
		old_coverage=coverage
		coverage=compute_coverage(mmasks, cmasks, v)
		print get_bit_string(v, n), coverage,  coverage-old_coverage, n-num_bits_on(v)

		if coverage == l:
			print "number of masks", len(masks)
			break

	masks.sort()
	print
	print
	for i in masks:
		print get_bit_string(i, n, ('0','1'))

	#for (i,m) in enumerate(mmasks):
	#	if cmasks[i]==0:
	#		print get_bit_string(m, n)
	#
	#for i in mmasks:
	#	print get_bit_string(i, n, ('0','1'))
