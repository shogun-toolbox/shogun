from sg import sg
from numpy import *
from numpy.random import randint, seed, rand, permutation
num_hmms=1;

def get_cubes (num=3):
	leng=50
	rep=5
	weight=1

	sequence=[]

	for i in xrange(num):
		# generate a sequence with characters 1-6 drawn from 3 loaded cubes
		loaded=[]
		for j in xrange(3):
			draw=[x*ones((1, ceil(leng*rand())), int)[0] \
				for x in xrange(1, 7)]
			loaded.append(permutation(concatenate(draw)))

		draws=[]
		for j in xrange(len(loaded)):
			data=ones((1, ceil(rep*rand())), int)
			draws=concatenate((j*data[0], draws))
		draws=permutation(draws)

		seq=[]
		for j in xrange(len(draws)):
			len_loaded=len(loaded[draws[j]])
			weighted=int(ceil(
				((1-weight)*rand()+weight)*len_loaded))
			perm=permutation(len_loaded)
			shuffled=[str(loaded[draws[j]][x]) for x in perm[:weighted]]
			seq=concatenate((seq, shuffled))

		sequence.append(''.join(seq))

	return sequence


sequence=get_cubes()

hmms=list()
liks=list()
for i in xrange(num_hmms):
	sg('send_command','new_hmm 3 6')
	sg('set_features','TRAIN',sequence,'CUBE')
	sg('send_command', 'convert TRAIN STRING CHAR STRING WORD 1')
	sg('send_command', 'bw')
	hmms.append(sg('get_hmm'))
	liks.append(sg('hmm_likelihood'))

sg('send_command','new_hmm 3 6')
sg('set_hmm',hmms[0][0],hmms[0][1],hmms[0][2],hmms[0][3])
sg('set_features','TRAIN',sequence,'CUBE')
l=sg('hmm_likelihood')
print l
