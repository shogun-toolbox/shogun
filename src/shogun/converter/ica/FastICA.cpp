/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Weijie Lin, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/converter/ica/FastICA.h>

#include <shogun/features/DenseFeatures.h>

#include <shogun/base/progress.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>

using namespace shogun;
using namespace Eigen;

namespace {

	MatrixXd sym_decorrelation(MatrixXd W)
	{
		MatrixXd K = W * W.transpose();

		SelfAdjointEigenSolver<MatrixXd> eig;
		eig.compute(K);

		return ((eig.eigenvectors() * eig.eigenvalues().cwiseSqrt().asDiagonal().inverse()) * eig.eigenvectors().transpose()) * W;
	}

	float64_t alpha = 1.0; // alpha must be in range [1.0 - 2.0]

	float64_t gx(float64_t x)
	{
		return std::tanh(x * alpha);
	}

	float64_t g_x(float64_t x)
	{
		return alpha * (1.0 - pow(gx(x),2));
	}

};

FastICA::FastICA() : RandomMixin<ICAConverter>()
{
	init();
}

void FastICA::init()
{
	whiten = true;
	SG_ADD(&whiten, "whiten", "flag indicating whether to whiten the data");
}

FastICA::~FastICA()
{
}

void FastICA::set_whiten(bool _whiten)
{
	whiten = _whiten;
}

bool FastICA::get_whiten() const
{
	return whiten;
}

void FastICA::fit_dense(std::shared_ptr<DenseFeatures<float64_t>> features)
{
	auto X = features->get_feature_matrix();
	require(X.data(), "Features have not been provided.");
	int n = X.num_rows;
	int p = X.num_cols;
	int m = n;

	Eigen::Map<MatrixXd> EX(X.matrix,n,p);

	// Whiten
	MatrixXd K;
	MatrixXd WX;
	if (whiten)
	{
		VectorXd mean = (EX.rowwise().sum() / (float64_t)p);
		MatrixXd SPX = EX.colwise() - mean;

		Eigen::JacobiSVD<MatrixXd> svd;
		svd.compute(SPX, Eigen::ComputeThinU);

		MatrixXd u = svd.matrixU();
		MatrixXd d = svd.singularValues();

		// for matching numpy/scikit-learn
		//u.rightCols(u.cols() - 1) *= -1;

		// see Hyvarinen (6.33) p.140
		K = u.transpose();
		for (int r = 0; r < K.rows(); r++)
			K.row(r) /= d(r);

		// see Hyvarinen (13.6) p.267 Here WX is white and data
		// in X has been projected onto a subspace by PCA
		WX = K * SPX;
		WX *= std::sqrt((float64_t)p);
	}
	else
	{
		WX = EX;
	}

	// Initial mixing matrix estimate
	if (m_mixing_matrix.num_rows != m || m_mixing_matrix.num_cols != m)
	{
		m_mixing_matrix = SGMatrix<float64_t>(m,m);
		random::fill_array(m_mixing_matrix, NormalDistribution<float64_t>(), m_prng);
	}

	Eigen::Map<MatrixXd> W(m_mixing_matrix.matrix, m, m);

	W = sym_decorrelation(W);

	float64_t lim = tol+1;
	for (auto i : SG_PROGRESS(range(0, max_iter), [&] { return lim > tol; }))
	{
		MatrixXd wtx = W * WX;

		MatrixXd gwtx = wtx.unaryExpr([](float64_t x) { return gx(x); });
		MatrixXd g_wtx = wtx.unaryExpr([](float64_t x) { return g_x(x); });

		MatrixXd W1 = (gwtx * WX.transpose()) / (float64_t)p - (g_wtx.rowwise().sum()/(float64_t)p).asDiagonal() * W;

		W1 = sym_decorrelation(W1);

		lim = ((W1 * W.transpose()).diagonal().cwiseAbs().array() - 1).abs().maxCoeff();

		W = W1;
	}

	// Unmix
	if (whiten)
		W = (W*K);

	// set m_mixing_matrix
	W = W.inverse();
}
