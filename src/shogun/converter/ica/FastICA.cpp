/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 * ported from scikit-learn
 */

#include <converter/ica/FastICA.h>

#include <features/DenseFeatures.h>

#ifdef HAVE_EIGEN3

#include <mathematics/Math.h>
#include <mathematics/eigen3.h>

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
		return CMath::tanh(x*alpha);
	}

	float64_t g_x(float64_t x)
	{
		return alpha * (1.0 - pow(gx(x),2));
	}

};

CFastICA::CFastICA() : CICAConverter()
{
	init();
}

void CFastICA::init()
{
	whiten = true;
	SG_ADD(&whiten, "whiten", "flag indicating whether to whiten the data", MS_NOT_AVAILABLE);
}

CFastICA::~CFastICA()
{
}

void CFastICA::set_whiten(bool _whiten)
{
	whiten = _whiten;
}

bool CFastICA::get_whiten() const
{
	return whiten;
}

CFeatures* CFastICA::apply(CFeatures* features)
{
	ASSERT(features);
	SG_REF(features);

	SGMatrix<float64_t> X = ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	int n = X.num_rows;
	int p = X.num_cols;
	int m = n;

	Map<MatrixXd> EX(X.matrix,n,p);

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
		WX *= CMath::sqrt((float64_t)p);
	}
	else
	{
		WX = EX;
	}

	// Initial mixing matrix estimate
	if (m_mixing_matrix.num_rows != m || m_mixing_matrix.num_cols != m)
	{
		m_mixing_matrix = SGMatrix<float64_t>(m,m);

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < m; j++)
				m_mixing_matrix(i,j) = CMath::randn_double();
		}
	}

	Map<MatrixXd> W(m_mixing_matrix.matrix, m, m);

	W = sym_decorrelation(W);

	int iter = 0;
	float64_t lim = tol+1;
	while (lim > tol && iter < max_iter)
	{
		MatrixXd wtx = W * WX;

		MatrixXd gwtx = wtx.unaryExpr(std::ptr_fun(&gx));
		MatrixXd g_wtx = wtx.unaryExpr(std::ptr_fun(&g_x));

		MatrixXd W1 = (gwtx * WX.transpose()) / (float64_t)p - (g_wtx.rowwise().sum()/(float64_t)p).asDiagonal() * W;

		W1 = sym_decorrelation(W1);

		lim = ((W1 * W.transpose()).diagonal().cwiseAbs().array() - 1).abs().maxCoeff();

		W = W1;

		iter++;
	}

	// Unmix
	if (whiten)
		W = (W*K);

	EX = W * EX;

	// set m_mixing_matrix
	W = W.inverse();

	return features;
}

#endif // HAVE_EIGEN3
