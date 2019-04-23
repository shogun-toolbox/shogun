/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/converter/ica/FFSep.h>

#include <shogun/features/DenseFeatures.h>


#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/ajd/FFDiag.h>

using namespace shogun;
using namespace Eigen;

namespace { MatrixXd cor(MatrixXd x, int tau = 0, bool mean_flag = true); };

FFSep::FFSep() : ICAConverter()
{
	init();
}

void FFSep::init()
{
	m_tau = SGVector<float64_t>(4);
	m_tau[0]=0; m_tau[1]=1; m_tau[2]=2; m_tau[3]=3;

	m_covs = SGNDArray<float64_t>();

	SG_ADD(&m_tau, "tau", "tau vector", ParameterProperties::HYPER);
}

FFSep::~FFSep()
{
}

void FFSep::set_tau(SGVector<float64_t> tau)
{
	m_tau = tau;
}

SGVector<float64_t> FFSep::get_tau() const
{
	return m_tau;
}

SGNDArray<float64_t> FFSep::get_covs() const
{
	return m_covs;
}

void FFSep::fit_dense(std::shared_ptr<DenseFeatures<float64_t>> features)
{
	ASSERT(features);

	auto X = features->get_feature_matrix();

	int n = X.num_rows;
	int m = X.num_cols;
	int N = m_tau.vlen;

	Eigen::Map<MatrixXd> EX(X.matrix,n,m);

	// Compute Correlation Matrices
	index_t * M_dims = SG_MALLOC(index_t, 3);
	M_dims[0] = n;
	M_dims[1] = n;
	M_dims[2] = N;
	m_covs = SGNDArray< float64_t >(M_dims, 3);

	for (int t = 0; t < N; t++)
	{
		Eigen::Map<MatrixXd> EM(m_covs.get_matrix(t),n,n);
		EM = cor(EX,m_tau[t]);
	}

	// Diagonalize
	SGMatrix<float64_t> Q = FFDiag::diagonalize(m_covs, m_mixing_matrix, tol, max_iter);
	Eigen::Map<MatrixXd> EQ(Q.matrix,n,n);

	// Compute Mixing Matrix
	m_mixing_matrix = SGMatrix<float64_t>(n,n);
	Eigen::Map<MatrixXd> C(m_mixing_matrix.matrix,n,n);
	C = EQ.inverse();

	// Normalize Estimated Mixing Matrix
	for (int t = 0; t < C.cols(); t++)
		C.col(t) /= C.col(t).maxCoeff();
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

