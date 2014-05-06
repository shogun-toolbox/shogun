 /*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 * Code adapted from 
 * and the reference paper is
 */

#include <shogun/machine/gp/VGInferenceMethod.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/machine/gp/MatrixOperations.h>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

CVGInferenceMethod::CVGInferenceMethod() : CKLInferenceMethod()
{
	init();
}

CVGInferenceMethod::CVGInferenceMethod(CKernel* kern,
		CFeatures* feat, CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
		: CKLInferenceMethod(kern, feat, m, lab, mod)
{
	init();
}

void CVGInferenceMethod::init()
{
}

CVGInferenceMethod::~CVGInferenceMethod()
{
}

float64_t CVGInferenceMethod::get_negative_log_marginal_likelihood()
{
	return 0;
}

void CVGInferenceMethod::update_alpha()
{

}

void CVGInferenceMethod::update_deriv()
{

}

SGVector<float64_t> CVGInferenceMethod::get_derivative_wrt_inference_method(
	const TParameter* param)
{
	return SGVector<float64_t>();
}

SGVector<float64_t> CVGInferenceMethod::get_derivative_wrt_likelihood_model(
	const TParameter* param)
{

	return SGVector<float64_t>();
}

SGVector<float64_t> CVGInferenceMethod::get_derivative_wrt_kernel(
	const TParameter* param)
{

	return SGVector<float64_t>();
}

SGVector<float64_t> CVGInferenceMethod::get_derivative_wrt_mean(
	const TParameter* param)
{
	return SGVector<float64_t>();
}

}

#endif /* HAVE_EIGEN3 */
