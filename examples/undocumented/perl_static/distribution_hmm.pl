from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()
traindna=lm.load_dna('../data/fm_train_dna.dat')
cubedna=lm.load_cubes('../data/fm_train_cube.dat')
parameter_list=[[traindna,cubedna,3,6,1,list(),list()],
		[traindna,cubedna,3,6,1,list(),list()]]

def distribution_hmm(fm_train=traindna,fm_cube=cubedna,N=3,M=6,
			   order=1,hmms=list(),links=list()):
	sg('new_hmm',N, M)
	sg('set_features', 'TRAIN', fm_cube, 'CUBE')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order)
	sg('bw')
	hmm=sg('get_hmm')

	sg('new_hmm', N, M)
	sg('set_hmm', hmm[0], hmm[1], hmm[2], hmm[3])
	likelihood=sg('hmm_likelihood')
	return likelihood

if __name__=='__main__':
	print('HMM')
	distribution_hmm(*parameter_list[0])

