/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/converter/ica/ICAConverter.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;
using namespace Eigen;

ICAConverter::ICAConverter() : Converter()
{
	init();
}

void ICAConverter::init()
{
	m_mixing_matrix = SGMatrix<float64_t>();
	max_iter = 200;
	tol = 1e-6;

	SG_ADD(&m_mixing_matrix, "mixing_matrix", "the mixing matrix");
	SG_ADD(&max_iter, "max_iter", "maximum number of iterations");
	SG_ADD(&tol, "tol", "the convergence tolerance");
}

ICAConverter::~ICAConverter()
{
}

void ICAConverter::set_mixing_matrix(SGMatrix<float64_t> mixing_matrix)
{
	m_mixing_matrix = mixing_matrix;
}

SGMatrix<float64_t> ICAConverter::get_mixing_matrix() const
{
	return m_mixing_matrix;
}

void ICAConverter::set_max_iter(int iter)
{
	max_iter = iter;
}

int ICAConverter::get_max_iter() const
{
	return max_iter;
}

void ICAConverter::set_tol(float64_t _tol)
{
	tol = _tol;
}

float64_t ICAConverter::get_tol() const
{
	return tol;
}

void ICAConverter::fit(std::shared_ptr<Features> features)
{
	REQUIRE(features, "Features are not provided\n");
	REQUIRE(
	    features->get_feature_class() == C_DENSE,
	    "ICA converters only work with dense features\n");
	REQUIRE(
	    features->get_feature_type() == F_DREAL,
	    "ICA converters only work with real features\n");

	fit_dense(std::dynamic_pointer_cast<DenseFeatures<float64_t>>(features));
}

std::shared_ptr<Features> ICAConverter::transform(std::shared_ptr<Features> features, bool inplace)
{
	REQUIRE(m_mixing_matrix.matrix, "ICAConverter has not been fitted.\n");

	auto X = std::dynamic_pointer_cast<DenseFeatures<float64_t>>(features)->get_feature_matrix();
	if (!inplace)
		X = X.clone();

	Eigen::Map<MatrixXd> EX(X.matrix, X.num_rows, X.num_cols);
	Eigen::Map<MatrixXd> C(
	    m_mixing_matrix.matrix, m_mixing_matrix.num_rows,
	    m_mixing_matrix.num_cols);

	// Unmix
	EX = C.inverse() * EX;

	return std::make_shared<DenseFeatures<float64_t>>(X);
}

std::shared_ptr<Features> ICAConverter::inverse_transform(std::shared_ptr<Features> features, bool inplace)
{
	REQUIRE(m_mixing_matrix.matrix, "ICAConverter has not been fitted.\n");

	auto X = std::dynamic_pointer_cast<DenseFeatures<float64_t>>(features)->get_feature_matrix();
	if (!inplace)
		X = X.clone();

	linalg::matrix_prod(m_mixing_matrix, X, X);

	return std::make_shared<DenseFeatures<float64_t>>(X);
}
