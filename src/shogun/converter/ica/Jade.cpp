/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 */

#include <shogun/converter/ica/Jade.h>

#include <shogun/features/DenseFeatures.h>

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/ajd/JADiagOrth.h>

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;
typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;

using namespace shogun;

CJade::CJade() : CConverter()
{	
	init();
}

void CJade::init()
{
	m_mixing_matrix = SGMatrix<float64_t>();
	SG_ADD(&m_mixing_matrix, "mixing_matrix", "m_mixing_matrix", MS_NOT_AVAILABLE);
}

CJade::~CJade()
{
}

SGMatrix<float64_t> CJade::get_mixing_matrix() const
{
	return m_mixing_matrix;
}

CFeatures* CJade::apply(CFeatures* features)
{
	SG_REF(features);

	SGMatrix<float64_t> X = ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	int n = X.num_rows;
	int T = X.num_cols;
	int m = n;

	Eigen::Map<EMatrix> EX(X.matrix,n,T);

	// Mean center x
	EMatrix SPX = EX.rowwise() - (EX.colwise().sum() / T);

	EMatrix cov = (SPX * SPX.transpose()) / T;

	// Whitening & Projection onto signal subspace
	EigenSolver<EMatrix> eig;
	eig.compute(cov);

	// PCA
	EMatrix B = eig.pseudoEigenvectors().transpose();

	// Scaling
	EMatrix scales = eig.pseudoEigenvalueMatrix().cwiseSqrt();
	B = scales.inverse() * B;

	// Sphering
	SPX = B * SPX;

	// Estimation of the cumulant matrices
	int dimsymm = (m * ( m + 1)) / 2; // Dim. of the space of real symm matrices
	int nbcm = dimsymm; //  number of cumulant matrices	
	EMatrix CM = EMatrix::Zero(m,m*nbcm); // Storage for cumulant matrices
	EMatrix R(m,m); R.setIdentity();
	EMatrix Qij = EMatrix::Zero(m,m); // Temp for a cum. matrix
	EVector Xim = EVector::Zero(m); // Temp
	EVector Xjm = EVector::Zero(m); // Temp
	EVector Xijm = EVector::Zero(m); // Temp
	int Range = 0;

	for(int im = 0; im < m; im++)
	{
		Xim = SPX.row(im);		
		Xijm = Xim.cwiseProduct(Xim);
		Qij = SPX.cwiseProduct(Xijm.replicate(1,m).transpose()) * SPX.transpose() / (float)T - R - 2*R.col(im)*R.col(im).transpose();
		CM.block(0,Range,m,m) = Qij;
		Range = Range + m;	
		for(int jm = 0; jm < im; jm++)
		{
			Xjm = SPX.row(jm);
			Xijm = Xim.cwiseProduct(Xjm);
			Qij = sqrt(2) * SPX.cwiseProduct(Xijm.replicate(1,m).transpose()) * SPX.transpose() / (float)T - R.col(im)*R.col(jm).transpose() - R.col(jm)*R.col(im).transpose();
			CM.block(0,Range,m,m) = Qij;
			Range = Range + m;	
		}
	}

	index_t * M_dims = SG_MALLOC(index_t, 3);
	M_dims[0] = m;
	M_dims[1] = m;
	M_dims[2] = nbcm;
	SGNDArray< float64_t > M(M_dims, 3);
	
	for(int i = 0; i < nbcm; i++)
	{
		Eigen::Map<EMatrix> EM(M.get_matrix(i),m,m);
		EM = CM.block(0,i*m,m,m);
	}
	
	// Diagonalize
	SGMatrix<float64_t> Q = CJADiagOrth::diagonalize(M);
	Eigen::Map<EMatrix> EQ(Q.matrix,m,m);

	// Compute Mixing Matrix
	m_mixing_matrix = SGMatrix<float64_t>(m,m);
	Eigen::Map<EMatrix> C(m_mixing_matrix.matrix,n,n);
	C = EQ * B;

	// Unmix
	EX = C * EX;
	
	return features;
}

#endif // HAVE_EIGEN3
