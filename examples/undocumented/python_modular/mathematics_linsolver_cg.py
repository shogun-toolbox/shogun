#!/usr/bin/env python

from numpy import *
from scipy.io import mmread

# Loading an example sparse matrix of dimension 479x479, real, unsymmetric
mtx=mmread('../../../data/logdet/west0479.mtx.gz')

parameter_list=[[mtx,6000,10]]

def mathematics_linsolver_cg (matrix=mtx,max_iter=1000,seed=10):

	# Create a Hermitian sparse matrix
	from scipy.sparse import eye

	rows=matrix.shape[0]
	cols=matrix.shape[1]
	A=matrix.transpose()*matrix+eye(rows, cols)

	# Create a random vector (b) of the system Ax=b
	random.seed(seed)
	b=array(random.randn(rows))

	# create linear system with linear operator and vector
	from scipy.sparse import csc_matrix

	try:
		from shogun.Mathematics import RealSparseMatrixOperator
		from shogun.Mathematics import ConjugateGradientSolver

		op=RealSparseMatrixOperator(A.tocsc())
		solver=ConjugateGradientSolver()

		# set the iteration limit higher for poorly conditioned matrices
		solver.set_iteration_limit(max_iter)
		x=solver.solve(op, b)

		# verifying the solution via direct solving
		from scipy.sparse.linalg import spsolve, eigsh
		y=spsolve(A,b)
		print(sqrt(sum(map(lambda z: z*z,x-y))))

		return x

	except ImportError:
		print('Shogun not installed with Eigen3!')

if __name__=='__main__':
	print('CG')
	mathematics_linsolver_cg (*parameter_list[0])
