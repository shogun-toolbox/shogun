def hmm ():
	print 'HMM'

	N=3
	M=6
	order=1
	hmms=list()
	liks=list()

	from sg import sg
	sg('new_hmm',N, M)
	sg('set_features', 'TRAIN', fm_cube, 'CUBE')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order)
	sg('bw')
	hmm=sg('get_hmm')

	sg('new_hmm', N, M)
	sg('set_hmm', hmm[0], hmm[1], hmm[2], hmm[3])
	likelihood=sg('hmm_likelihood')

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train=lm.load_dna('../data/fm_train_dna.dat')
	fm_cube=lm.load_cubes('../data/fm_train_cube.dat')
	hmm()
