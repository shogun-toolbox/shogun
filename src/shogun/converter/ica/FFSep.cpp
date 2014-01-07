/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 */

#include <converter/ica/FFSep.h>

#include <features/DenseFeatures.h>

#ifdef HAVE_EIGEN3

#include <mathematics/Math.h>
#include <mathematics/eigen3.h>
#include <mathematics/ajd/FFDiag.h>

using namespace shogun;
using namespace Eigen;

namespace { MatrixXd cor(MatrixXd x, int tau = 0, bool mean_flag = true); };

CFFSep::CFFSep() : CICAConverter()
{
	init();
}

void CFFSep::init()
{
	m_tau = SGVector<float64_t>(4);
	m_tau[0]=0; m_tau[1]=1; m_tau[2]=2; m_tau[3]=3;

	m_covs = SGNDArray<float64_t>();

	SG_ADD(&m_tau, "tau", "tau vector", MS_AVAILABLE);
}

CFFSep::~CFFSep()
{
}

void CFFSep::set_tau(SGVector<float64_t> tau)
{
	m_tau = tau;
}

SGVector<float64_t> CFFSep::get_tau() const
{
	return m_tau;
}

SGNDArray<float64_t> CFFSep::get_covs() const
{
	return m_covs;
}

CFeatures* CFFSep::apply(CFeatures* features)
{
	ASSERT(features);
	SG_REF(features);

	SGMatrix<float64_t> X = ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	int n = X.num_rows;
	int m = X.num_cols;
	int N = m_tau.vlen;

	Map<MatrixXd> EX(X.matrix,n,m);

	// Compute Correlation Matrices
	index_t * M_dims = SG_MALLOC(index_t, 3);
	M_dims[0] = n;
	M_dims[1] = n;
	M_dims[2] = N;
	m_covs = SGNDArray< float64_t >(M_dims, 3);

	for (int t = 0; t < N; t++)
	{
		Map<MatrixXd> EM(m_covs.get_matrix(t),n,n);
		EM = cor(EX,m_tau[t]);
	}

	// Diagonalize
	SGMatrix<float64_t> Q = CFFDiag::diagonalize(m_covs, m_mixing_matrix, tol, max_iter);
	Map<MatrixXd> EQ(Q.matrix,n,n);

	// Compute Mixing Matrix
	m_mixing_matrix = SGMatrix<float64_t>(n,n);
	Map<MatrixXd> C(m_mixing_matrix.matrix,n,n);
	C = EQ.inverse();

	// Normalize Estimated Mixing Matrix
	for (int t = 0; t < C.cols(); t++)
		C.col(t) /= C.col(t).maxCoeff();

	// Unmix
	EX = C.inverse() * EX;

	return features;
}

// Computing time delayed correlation matrix
namespace
{
	MatrixXd cor(MatrixXd x, int tau, bool mean_flag)
	{
		int m = x.rows();
		int n = x.cols();

		// Center the data
		if ( mean_flag )
		{
			VectorXd mean = x.rowwise().sum();
			mean /= n;
			x = x.colwise() - mean;
		}

		// Time-delayed Signal Matrix
		MatrixXd L = x.leftCols(n-tau);
		MatrixXd R = x.rightCols(n-tau);

		// Compute Correlations
		MatrixXd K(m,m);
		K = (L * R.transpose()) / (n-tau);

		// Symmetrize
		K = (K + K.transpose()) / 2.0;

		return K;
	}
};

#endif // HAVE_EIGEN3
