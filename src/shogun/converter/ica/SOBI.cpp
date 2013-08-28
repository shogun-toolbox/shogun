/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 */

#include <shogun/converter/ica/SOBI.h>

#include <shogun/features/DenseFeatures.h>

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/ajd/JADiagOrth.h>

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;
typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;

using namespace shogun;

namespace { EMatrix cor(EMatrix x, int tau = 0, bool mean_flag = true); };

CSOBI::CSOBI() : CICAConverter()
{	
	init();
}

void CSOBI::init()
{
	m_tau = SGVector<float64_t>(4); 
	m_tau[0]=0; m_tau[1]=1; m_tau[2]=2; m_tau[3]=3;
	
	m_covs = SGNDArray<float64_t>();
	
	SG_ADD(&m_tau, "tau", "tau vector", MS_AVAILABLE);
}

CSOBI::~CSOBI()
{
}

void CSOBI::set_tau(SGVector<float64_t> tau)
{
	m_tau = tau;
}

SGVector<float64_t> CSOBI::get_tau() const
{
	return m_tau;
}

SGNDArray<float64_t> CSOBI::get_covs() const
{
	return m_covs;
}

CFeatures* CSOBI::apply(CFeatures* features)
{
	ASSERT(features);	
	SG_REF(features);

	SGMatrix<float64_t> X = ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	int n = X.num_rows;
	int m = X.num_cols;
	int N = m_tau.vlen;

	Eigen::Map<EMatrix> EX(X.matrix,n,m);	

	// Whitening or Sphering
	EMatrix M0 = cor(EX,int(m_tau[0]));	
	EigenSolver<EMatrix> eig;
	eig.compute(M0);
	EMatrix SPH = (eig.pseudoEigenvectors() * eig.pseudoEigenvalueMatrix().cwiseSqrt() * eig.pseudoEigenvectors ().transpose()).inverse();
	EMatrix spx = SPH*EX;

	// Compute Correlation Matrices
	index_t * M_dims = SG_MALLOC(index_t, 3);
	M_dims[0] = n;
	M_dims[1] = n;
	M_dims[2] = N;
	m_covs = SGNDArray< float64_t >(M_dims, 3);
	
	for(int t = 0; t < N; t++)
	{
		Eigen::Map<EMatrix> EM(m_covs.get_matrix(t),n,n);
		EM = cor(spx,m_tau[t]);
	}

	// Diagonalize
	SGMatrix<float64_t> Q = CJADiagOrth::diagonalize(m_covs);
	Eigen::Map<EMatrix> EQ(Q.matrix,n,n);

	// Compute Mixing Matrix
	m_mixing_matrix = SGMatrix<float64_t>(n,n);
	Eigen::Map<EMatrix> C(m_mixing_matrix.matrix,n,n);
	C = SPH.inverse() * EQ.transpose();

	// Normalize Estimated Mixing Matrix
	for(int t = 0; t < C.cols(); t++)
	{	
		C.col(t) /= C.col(t).maxCoeff();
	}

	// Unmix
	EX = C.inverse() * EX;
	
	return features;
}

// Computing time delayed correlation matrix
namespace 
{
	EMatrix cor(EMatrix x, int tau, bool mean_flag)
	{	
		int m = x.rows();
		int n = x.cols();

		// Center the data
		if ( mean_flag )
		{		
			EVector mean = x.rowwise().sum();
			mean /= n;
			x = x.colwise() - mean;
		}

		// Time-delayed Signal Matrix
		EMatrix L = x.leftCols(n-tau);
		EMatrix R = x.rightCols(n-tau);

		// Compute Correlations
		EMatrix K(m,m);
		K = (L * R.transpose()) / (n-tau);

		// Symmetrize
		K = (K + K.transpose()) / 2.0;

		return K;
	}
};
#endif // HAVE_EIGEN3
