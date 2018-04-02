#!/usr/bin/env python

from numpy import *
from scipy.io import mmread

# Loading an example sparse matrix of dimension 479x479, real, unsymmetric
mtx=mmread('../../../data/logdet/west0479.mtx')

parameter_list=[[mtx,100,60,1]]

def mathematics_logdet (matrix=mtx,max_iter_eig=1000,max_iter_lin=1000,num_samples=1):

	from scipy.sparse import eye

	# Create a Hermitian sparse matrix
	rows=matrix.shape[0]
	cols=matrix.shape[1]
	A=matrix.transpose()*matrix+eye(rows, cols)

	from scipy.sparse import csc_matrix

	try:
		from shogun.Mathematics import RealSparseMatrixOperator
		from shogun.Mathematics import LanczosEigenSolver
		from shogun.Mathematics import CGMShiftedFamilySolver
		from shogun.Mathematics import LogRationalApproximationCGM
		from shogun.Mathematics import ProbingSampler
		from shogun.Mathematics import LogDetEstimator
		from shogun.Mathematics import Statistics

		# creating the linear operator, eigen-solver
		op=RealSparseMatrixOperator(A.tocsc())

		eig_solver=LanczosEigenSolver(op)

		# we can set the iteration limit high for poorly conditioned matrices
		eig_solver.set_max_iteration_limit(max_iter_eig)

		# alternatively, if the matrix is small, we can compute eigenvalues externally
		# and set min/max eigenvalues into the eigensolver
		# from scipy.sparse.linalg import eigsh

		# eigenvalues=eigsh(A, rows-1)
		# eig_solver.set_min_eigenvalue(eigenvalues[0][0])
		# eig_solver.set_max_eigenvalue(eigenvalues[0][-1])

		# create the shifted-family linear solver which solves for all the shifts
		# using as many matrix-vector products as one shift in CG iterations
		lin_solver=CGMShiftedFamilySolver()
		lin_solver.set_iteration_limit(max_iter_lin)


		# set the desired accuracy tighter to obtain better results
		# this determines the number of contour points in conformal mapping of
		# the rational approximation of the Cauchy's integral of f(A)*s, f=log
		desired_accuracy=1E-5

		# creating the log-linear-operator function
		op_func=LogRationalApproximationCGM(op,  eig_solver, lin_solver,\
			desired_accuracy)

		# set the trace sampler to be probing sampler, in which samples are obtained
		# by greedy graph coloring of the power of sparse matrix (default is power=1,
		# 2-distance coloring)
		trace_sampler=ProbingSampler(op)

		# estimating log-det
		log_det_estimator=LogDetEstimator(trace_sampler, op_func)

		# set the number of samples as required
		estimates=log_det_estimator.sample(num_samples)

		estimated_logdet=sum(estimates)/len(estimates)
		actual_logdet=Statistics.log_det(A)

		print(actual_logdet, estimated_logdet)

		return estimates

	except ImportError:
		print('One or many of the dependencies (Eigen3/LaPack/ColPack) not found!')

if __name__=='__main__':
	print('LogDetEstimator')
	mathematics_logdet (*parameter_list[0])
