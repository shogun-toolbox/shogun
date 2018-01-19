/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#include <shogun/regression/LeastSquaresRegression.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CLeastSquaresRegression::CLeastSquaresRegression()
: CLinearRidgeRegression()
{
	m_tau=0;
}

CLeastSquaresRegression::CLeastSquaresRegression(CDenseFeatures<float64_t>* data, CLabels* lab)
: CLinearRidgeRegression(0, data, lab)
{
}
#endif
