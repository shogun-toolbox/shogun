/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/converter/ica/ICAConverter.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CICAConverter::CICAConverter() : CConverter()
{
	init();
}

void CICAConverter::init()
{
	m_mixing_matrix = SGMatrix<float64_t>();
	max_iter = 200;
	tol = 1e-6;

	SG_ADD(&m_mixing_matrix, "mixing_matrix", "the mixing matrix", MS_NOT_AVAILABLE);
	SG_ADD(&max_iter, "max_iter", "maximum number of iterations", MS_NOT_AVAILABLE);
	SG_ADD(&tol, "tol", "the convergence tolerance", MS_NOT_AVAILABLE);
}

CICAConverter::~CICAConverter()
{
}

void CICAConverter::set_mixing_matrix(SGMatrix<float64_t> mixing_matrix)
{
	m_mixing_matrix = mixing_matrix;
}

SGMatrix<float64_t> CICAConverter::get_mixing_matrix() const
{
	return m_mixing_matrix;
}

void CICAConverter::set_max_iter(int iter)
{
	max_iter = iter;
}

int CICAConverter::get_max_iter() const
{
	return max_iter;
}

void CICAConverter::set_tol(float64_t _tol)
{
	tol = _tol;
}

float64_t CICAConverter::get_tol() const
{
	return tol;
}

void CICAConverter::fit(CFeatures* features)
{
	REQUIRE(features, "Features are not provided\n");
	REQUIRE(
	    features->get_feature_class() == C_DENSE,
	    "ICA converters only work with dense features\n");
	REQUIRE(
	    features->get_feature_type() == F_DREAL,
	    "ICA converters only work with real features\n");

	SG_REF(features);

	fit_dense(static_cast<CDenseFeatures<float64_t>*>(features));

	SG_UNREF(features);
}

CFeatures* CICAConverter::apply(CFeatures* features, bool inplace)
{
	REQUIRE(m_mixing_matrix.matrix, "ICAConverter has not been fitted.\n");

	SG_REF(features);

	auto X = features->as<CDenseFeatures<float64_t>>()->get_feature_matrix();
	if (!inplace)
		X = X.clone();

	Map<MatrixXd> EX(X.matrix, X.num_rows, X.num_cols);
	Map<MatrixXd> C(
	    m_mixing_matrix.matrix, m_mixing_matrix.num_rows,
	    m_mixing_matrix.num_cols);

	// Unmix
	EX = C.inverse() * EX;

	auto processed = new CDenseFeatures<float64_t>(X);
	SG_UNREF(features);

	return processed;
}
