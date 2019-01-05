/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/regression/LeastSquaresRegression.h>

using namespace shogun;

CLeastSquaresRegression::CLeastSquaresRegression()
: CLinearRidgeRegression()
{
	m_tau=0;
}
