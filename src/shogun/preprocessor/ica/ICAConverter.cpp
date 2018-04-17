/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/preprocessor/ica/ICAConverter.h>


using namespace shogun;

CICAConverter::CICAConverter() : CPreprocessor()
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

