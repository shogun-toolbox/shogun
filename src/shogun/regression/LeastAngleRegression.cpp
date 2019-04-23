/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Giovanni De Toni,
 *          Saurabh Mahindre, Christopher Goldsworthy, Chiyuan Zhang,
 *          Viktor Gal, Abhinav Rai, Bjoern Esser, Weijie Lin, Pan Deng
 */

#include <shogun/lib/config.h>

#include <vector>
#include <limits>
#include <algorithm>

#include <shogun/base/progress.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/regression/LeastAngleRegression.h>

using namespace Eigen;
using namespace shogun;
using namespace std;

LeastAngleRegression::LeastAngleRegression()
    : DenseRealDispatch<LeastAngleRegression, LinearMachine>()
{
	init();
}

LeastAngleRegression::LeastAngleRegression(bool lasso)
    : DenseRealDispatch<LeastAngleRegression, LinearMachine>()
{
	init();

	m_lasso = lasso;
}

void LeastAngleRegression::init()
{
	m_lasso = true;
	m_max_nonz = 0;
	m_max_l1_norm = 0;
	m_epsilon = Math::MACHINE_EPSILON;
	SG_ADD(&m_epsilon, "epsilon", "Epsilon for early stopping", ParameterProperties::HYPER);
	SG_ADD(&m_max_nonz, "max_nonz", "Max number of non-zero variables", ParameterProperties::HYPER);
	SG_ADD(&m_max_l1_norm, "max_l1_norm", "Max l1-norm of estimator", ParameterProperties::HYPER);
	SG_ADD(&m_lasso, "lasso", "Max l1-norm of estimator", ParameterProperties::HYPER);
	watch_method("path_size", &LeastAngleRegression::get_path_size);
}

LeastAngleRegression::~LeastAngleRegression()
{

}

template <typename ST>
void LeastAngleRegression::find_max_abs(const std::vector<ST> &vec, const std::vector<bool> &ignore_mask,
	int32_t &imax, ST& vmax)
{
	imax = -1;
	vmax = -1;
	for (size_t i=0; i < vec.size(); ++i)
	{
		if (ignore_mask[i])
			continue;

		if (Math::abs(vec[i]) > vmax)
		{
			vmax = Math::abs(vec[i]);
			imax = i;
		}
	}
}

template <typename ST>
void LeastAngleRegression::plane_rot(ST x0, ST x1,
	ST &y0, ST &y1, SGMatrix<ST> &G)
{
	memset(G.matrix, 0, G.num_rows * G.num_cols * sizeof(ST));

	if (x1 == 0)
	{
		G(0, 0) = G(1, 1) = 1;
		y0 = x0;
		y1 = x1;
	}
	else
	{
		ST r = std::sqrt(x0 * x0 + x1 * x1);
		ST sx0 = x0 / r;
		ST sx1 = x1 / r;

		G(0,0) = sx0;
		G(1,0) = -sx1;
		G(0,1) = sx1;
		G(1,1) = sx0;

		y0 = r;
		y1 = 0;
	}
}

template <typename ST, typename U>
bool LeastAngleRegression::train_machine_templated(std::shared_ptr<DenseFeatures<ST>> data)
{
	std::vector<SGVector<ST>> m_beta_path_t;

	int32_t n_fea = data->get_num_features();
	int32_t n_vec = data->get_num_vectors();

	bool lasso_cond = false;
	bool stop_cond = false;

	// init facilities
	m_beta_idx.clear();
	m_beta_path_t.clear();
	m_num_active = 0;
	m_active_set.clear();
	m_is_active.resize(n_fea);
	fill(m_is_active.begin(), m_is_active.end(), false);

	SGVector<ST> y = regression_labels(m_labels)->template get_labels_t<ST>();
	typename SGVector<ST>::EigenVectorXtMap map_y(y.vector, y.size());

	// transpose(X) is more convenient to work with since we care
	// about features here. After transpose, each row will be a data
	// point while each column corresponds to a feature
	SGMatrix<ST> X (n_vec, n_fea);
	typename SGMatrix<ST>::EigenMatrixXtMap map_Xr = data->get_feature_matrix();
	typename SGMatrix<ST>::EigenMatrixXtMap map_X(X.matrix, n_vec, n_fea);
	map_X = map_Xr.transpose();

	SGMatrix<ST> X_active(n_vec, n_fea);

	// beta is the estimator
	SGVector<ST> beta(n_fea);
	beta.set_const(0);

	vector<ST> Xy(n_fea);
	typename SGVector<ST>::EigenVectorXtMap map_Xy(&Xy[0], n_fea);
	// Xy = X' * y
	map_Xy=map_Xr*map_y;

	// mu is the prediction
	vector<ST> mu(n_vec);

	// correlation
	vector<ST> corr(n_fea);
	// sign of correlation
	vector<ST> corr_sign(n_fea);

	// Cholesky factorization R'R = X'X, R is upper triangular
	SGMatrix<ST> R;

	ST max_corr = 1;
	int32_t i_max_corr = 1;

	// first entry: all coefficients are zero
	m_beta_path_t.push_back(beta.clone());
	m_beta_idx.push_back(0);

	//maximum allowed active variables at a time
	int32_t max_active_allowed = Math::min(n_vec-1, n_fea);

	//========================================
	// main loop
	//========================================
	int32_t nloop=0;
	auto pb = SG_PROGRESS(range(0, max_active_allowed));
	while (m_num_active < max_active_allowed && max_corr / n_vec > m_epsilon &&
	       !stop_cond)
	{
		COMPUTATION_CONTROLLERS

		// corr = X' * (y-mu) = - X'*mu + Xy
		typename SGVector<ST>::EigenVectorXtMap map_corr(&corr[0], n_fea);
		typename SGVector<ST>::EigenVectorXtMap map_mu(&mu[0], n_vec);

		map_corr = map_Xy - (map_Xr*map_mu);

		// corr_sign = sign(corr)
		for (size_t i=0; i < corr.size(); ++i)
			corr_sign[i] = Math::sign(corr[i]);

		// find max absolute correlation in inactive set
		find_max_abs(corr, m_is_active, i_max_corr, max_corr);

		if (!lasso_cond)
		{
			// update Cholesky factorization matrix
			if (m_num_active == 0)
			{
				// R isn't allocated yet
				R=SGMatrix<ST>(1,1);
				ST diag_k = map_X.col(i_max_corr).dot(map_X.col(i_max_corr));
				R(0, 0) = std::sqrt(diag_k);
			}
			else
				R=cholesky_insert(X, X_active, R, i_max_corr, m_num_active);
			activate_variable(i_max_corr);
		}

		// Active variables
		typename SGMatrix<ST>::EigenMatrixXtMap map_Xa(X_active.matrix, n_vec, m_num_active);
		if (!lasso_cond)
			map_Xa.col(m_num_active-1)=map_X.col(i_max_corr);

		SGVector<ST> corr_sign_a(m_num_active);
		for (index_t i=0; i < m_num_active; ++i)
			corr_sign_a[i] = corr_sign[m_active_set[i]];

		typename SGVector<ST>::EigenVectorXtMap map_corr_sign_a(corr_sign_a.vector, corr_sign_a.size());
		typename SGMatrix<ST>::EigenMatrixXtMap map_R(R.matrix, R.num_rows, R.num_cols);
		typename SGVector<ST>::EigenVectorXt solve = map_R.transpose().template triangularView<Lower>().template solve<OnTheLeft>(map_corr_sign_a);

		typename SGVector<ST>::EigenVectorXt GA1 = map_R.template triangularView<Upper>().template solve<OnTheLeft>(solve);

		// AA = 1/sqrt(GA1' * corr_sign_a)
		ST AA = GA1.dot(map_corr_sign_a);
		AA = 1 / std::sqrt(AA);

		typename SGVector<ST>::EigenVectorXt wA = AA*GA1;

		// equiangular direction (unit vector)
		vector<ST> u(n_vec);
		typename SGVector<ST>::EigenVectorXtMap map_u(&u[0], n_vec);

		map_u = map_Xa*wA;

		ST gamma = max_corr / AA;
		if (m_num_active < n_fea)
		{
			#pragma omp parallel for shared(gamma)
			for (index_t i=0; i < n_fea; ++i)
			{
				if (m_is_active[i])
					continue;

				// correlation between X[:,i] and u
				ST dir_corr = map_u.dot(map_X.col(i));

				ST tmp1 = (max_corr-corr[i])/(AA-dir_corr);
				ST tmp2 = (max_corr+corr[i])/(AA+dir_corr);
				#pragma omp critical
				{
				if (tmp1 > Math::MACHINE_EPSILON && tmp1 < gamma)
					gamma = tmp1;
				if (tmp2 > Math::MACHINE_EPSILON && tmp2 < gamma)
					gamma = tmp2;
				}
			}
		}

		int32_t i_kick=-1;
		int32_t i_change=i_max_corr;
		if (m_lasso)
		{
			// lasso modification to further refine gamma
			lasso_cond = false;
			ST lasso_bound = numeric_limits<ST>::max();

			for (index_t i=0; i < m_num_active; ++i)
			{
				ST tmp = -beta[m_active_set[i]] / wA(i);
				if (tmp > Math::MACHINE_EPSILON && tmp < lasso_bound)
				{
					lasso_bound = tmp;
					i_kick = i;
				}
			}

			if (lasso_bound < gamma)
			{
				gamma = lasso_bound;
				lasso_cond = true;
				i_change = m_active_set[i_kick];
			}
		}

		// update prediction: mu = mu + gamma * u
		map_mu += gamma*map_u;

		// update estimator
		for (index_t i=0; i < m_num_active; ++i)
			beta[m_active_set[i]] += gamma * wA(i);

		// early stopping on max l1-norm
		if (m_max_l1_norm > 0)
		{
			ST l1 = SGVector<ST>::onenorm(beta.vector, n_fea);
			if (l1 > m_max_l1_norm)
			{
				// stopping with interpolated beta
				stop_cond = true;
				lasso_cond = false;
				ST l1_prev = (ST)SGVector<ST>::onenorm(
				    m_beta_path_t[nloop].vector, n_fea);
				ST s = (m_max_l1_norm - l1_prev) / (l1 - l1_prev);

				typename SGVector<ST>::EigenVectorXtMap map_beta(
				    beta.vector, n_fea);
				typename SGVector<ST>::EigenVectorXtMap map_beta_prev(
				    m_beta_path_t[nloop].vector, n_fea);
				map_beta = (1-s)*map_beta_prev + s*map_beta;
			}
		}

		// if lasso cond, drop the variable from active set
		if (lasso_cond)
		{
			beta[i_change] = 0;
			R=cholesky_delete(R, i_kick);
			deactivate_variable(i_kick);

			// Remove column from active set
			int32_t numRows = map_Xa.rows();
			int32_t numCols = map_Xa.cols()-1;
			if( i_kick < numCols )
				map_Xa.block(0, i_kick, numRows, numCols-i_kick) =
					map_Xa.block(0, i_kick+1, numRows, numCols-i_kick).eval();
		}

		nloop++;
		m_beta_path_t.push_back(beta.clone());
		if (int32_t(m_num_active) >= get_path_size())
			m_beta_idx.push_back(nloop);
		else
			m_beta_idx[m_num_active] = nloop;

		// early stopping with max number of non-zero variables
		if (m_max_nonz > 0 && m_num_active >= m_max_nonz)
			stop_cond = true;
		SG_DEBUG("Added : {} , Dropped {}, Active set size {} max_corr {:.17f} ", i_max_corr, i_kick, m_num_active, max_corr);

		pb.print_progress();
	}
	pb.complete();

	//copy m_beta_path_t (of type ST) into m_beta_path
	// do also a cast to float64_t
	for (index_t i = 0; i < m_beta_path_t.size(); ++i)
	{
		SGVector<float64_t> va(m_beta_path_t[i].vlen);
		for (index_t p = 0; p < m_beta_path_t[i].vlen; ++p)
		{
			va.set_element(static_cast<float64_t>(m_beta_path_t[i][p]), p);
		}
		m_beta_path.push_back(va);
		observe(i, "beta_path", "Beta path", va.clone());
	}

	// assign default estimator
	set_w(SGVector<float64_t>(n_fea));
	switch_w(get_path_size()-1);

	if (max_corr / n_vec > m_epsilon)
	{
		io::warn(
		    "Convergence level ({}) not below tolerance ({}) after {} "
		    "iterations.",
		    max_corr / n_vec, m_epsilon, nloop);
	}

	return true;
}

template <typename ST>
SGMatrix<ST> LeastAngleRegression::cholesky_insert(const SGMatrix<ST>& X,
		const SGMatrix<ST>& X_active, SGMatrix<ST>& R, int32_t i_max_corr, int32_t num_active)
{
	typename SGMatrix<ST>::EigenMatrixXtMap map_X(X.matrix, X.num_rows, X.num_cols);
	typename SGMatrix<ST>::EigenMatrixXtMap map_X_active(X_active.matrix, X.num_rows, num_active);
	ST diag_k = map_X.col(i_max_corr).dot(map_X.col(i_max_corr));

	// col_k is the k-th column of (X'X)
	typename SGVector<ST>::EigenVectorXtMap map_i_max(X.get_column_vector(i_max_corr), X.num_rows);
	typename SGVector<ST>::EigenVectorXt R_k = map_X_active.transpose()*map_i_max;
	typename SGMatrix<ST>::EigenMatrixXtMap map_R(R.matrix, R.num_rows, R.num_cols);

	// R' * R_k = (X' * X)_k = col_k, solving to get R_k
	map_R.transpose().template triangularView<Lower>().template solveInPlace<OnTheLeft>(R_k);
	ST R_kk = std::sqrt(diag_k - R_k.dot(R_k));

	SGMatrix<ST> R_new(num_active+1, num_active+1);
	typename SGMatrix<ST>::EigenMatrixXtMap map_R_new(R_new.matrix, R_new.num_rows, R_new.num_cols);

	map_R_new.block(0, 0, num_active, num_active) = map_R;
	sg_memcpy(R_new.matrix+num_active*(num_active+1), R_k.data(), sizeof(ST)*(num_active));
	map_R_new.row(num_active).setZero();
	map_R_new(num_active, num_active) = R_kk;
	return R_new;
}

template <typename ST>
SGMatrix<ST> LeastAngleRegression::cholesky_delete(SGMatrix<ST>& R, int32_t i_kick)
{
	if (i_kick != m_num_active-1)
	{
		// remove i_kick-th column
		for (index_t j=i_kick; j < m_num_active-1; ++j)
			for (index_t i=0; i < m_num_active; ++i)
				R(i,j) = R(i,j+1);

		SGMatrix<ST> G(2,2);
		for (index_t i=i_kick; i < m_num_active-1; ++i)
		{
			plane_rot(R(i,i),R(i+1,i), R(i,i), R(i+1,i), G);
			if (i < m_num_active-2)
			{
				for (index_t k=i+1; k < m_num_active-1; ++k)
				{
					// R[i:i+1, k] = G*R[i:i+1, k]
					ST Rik = R(i,k), Ri1k = R(i+1,k);
					R(i,k) = G(0,0)*Rik + G(0,1)*Ri1k;
					R(i+1,k) = G(1,0)*Rik+G(1,1)*Ri1k;
				}
			}
		}
	}

	SGMatrix<ST> nR(m_num_active-1, m_num_active-1);
	for (index_t i=0; i < m_num_active-1; ++i)
		for (index_t j=0; j < m_num_active-1; ++j)
			nR(i,j) = R(i,j);

	return nR;
}

template bool LeastAngleRegression::train_machine_templated<floatmax_t>(std::shared_ptr<DenseFeatures<floatmax_t>> data);
template bool LeastAngleRegression::train_machine_templated<float64_t>(std::shared_ptr<DenseFeatures<float64_t>> data);
template bool LeastAngleRegression::train_machine_templated<float32_t>(std::shared_ptr<DenseFeatures<float32_t>> data);
template SGMatrix<float32_t> LeastAngleRegression::cholesky_insert(const SGMatrix<float32_t>& X, const SGMatrix<float32_t>& X_active, SGMatrix<float32_t>& R, int32_t i_max_corr, int32_t num_active);
template SGMatrix<float64_t> LeastAngleRegression::cholesky_insert(const SGMatrix<float64_t>& X, const SGMatrix<float64_t>& X_active, SGMatrix<float64_t>& R, int32_t i_max_corr, int32_t num_active);
template SGMatrix<floatmax_t> LeastAngleRegression::cholesky_insert(const SGMatrix<floatmax_t>& X, const SGMatrix<floatmax_t>& X_active, SGMatrix<floatmax_t>& R, int32_t i_max_corr, int32_t num_active);
