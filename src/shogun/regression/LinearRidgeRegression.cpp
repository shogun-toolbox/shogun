/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Chiyuan Zhang, Viktor Gal,
 *          Abhinav Rai, Youssef Emad El-Din, Heiko Strathmann
 */
#include <shogun/lib/config.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/regression/LinearRidgeRegression.h>

using namespace shogun;

LinearRidgeRegression::LinearRidgeRegression()
    : DenseRealDispatch<LinearRidgeRegression, LinearMachine>()
{
	init();
}

LinearRidgeRegression::LinearRidgeRegression(
    float64_t tau, std::shared_ptr<DenseFeatures<float64_t>> data, std::shared_ptr<Labels> lab)
    : DenseRealDispatch<LinearRidgeRegression, LinearMachine>()
{
	init();

	set_tau(tau);
	set_labels(lab);
	set_features(data);
}

void LinearRidgeRegression::init()
{
	set_tau(1e-6);
	m_use_bias = true;

	SG_ADD(&m_tau, "tau", "Regularization parameter", ParameterProperties::HYPER);
	SG_ADD(
	    &m_use_bias, "use_bias", "Whether or not to fit an offset term");
}

template <typename T>
bool LinearRidgeRegression::train_machine_templated(
    std::shared_ptr<const DenseFeatures<T>> feats)
{
	auto N = feats->get_num_vectors();
	auto D = feats->get_num_features();

	auto y = regression_labels(m_labels)->get_labels().as<T>();
	T tau = m_tau;

	SGVector<T> x_mean;
	T y_mean;
	if (m_use_bias)
	{
		x_mean = feats->mean();
		y_mean = linalg::mean(y);
	}

	SGVector<T> w;
	if (N >= D)
	{
		SGMatrix<T> cov = feats->cov();
		linalg::add_ridge(cov, tau);
		if (m_use_bias)
			linalg::rank_update(cov, x_mean, (T)-N);

		auto L = linalg::cholesky_factor(cov);
		auto Xy = feats->dot(y);
		if (m_use_bias)
			linalg::add(Xy, x_mean, Xy, (T)1, -N * y_mean);

		w = linalg::cholesky_solver(L, Xy);
	}
	else
	{
		if (m_use_bias)
			SG_NOTIMPLEMENTED

		SGMatrix<T> gram = feats->gram();
		linalg::add_ridge(gram, tau);
		auto L = linalg::cholesky_factor(gram);
		auto b = linalg::cholesky_solver(L, y);
		w = feats->dot(b);
	}
	set_w(w.template as<float64_t>());

	if (m_use_bias)
	{
		float64_t intercept = y_mean - linalg::dot(w, x_mean);
		set_bias(intercept);
	}

	return true;
}

bool LinearRidgeRegression::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool LinearRidgeRegression::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}
