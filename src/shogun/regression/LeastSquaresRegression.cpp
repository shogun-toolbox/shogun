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

#include <utility>

using namespace shogun;

LeastSquaresRegression::LeastSquaresRegression()
: LinearRidgeRegression()
{
	m_tau=0;
}

LeastSquaresRegression::LeastSquaresRegression(std::shared_ptr<DenseFeatures<float64_t>> data, std::shared_ptr<Labels> lab)
: LinearRidgeRegression(0, std::move(data), std::move(lab))
{
}
#endif
