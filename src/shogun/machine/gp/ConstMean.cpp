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
 * Adapted from the GPML toolbox, specifically meanConst.m
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#include <shogun/machine/gp/ConstMean.h>

using namespace shogun;

CConstMean::CConstMean() : CMeanFunction()
{
	init();
}

CConstMean::~CConstMean()
{
}

CConstMean::CConstMean(float64_t mean)
 : CMeanFunction()
{
	init();
	m_mean=mean;
}

void CConstMean::init()
{
	m_mean=0.0;
	SG_ADD(&m_mean, "mean", "const value of mean function", MS_AVAILABLE, GRADIENT_AVAILABLE);
}

SGVector<float64_t> CConstMean::get_mean_vector(const CFeatures* features) const
{
	SGVector<float64_t> result(features->get_num_vectors());
	result.set_const(m_mean);
	return result;
}

SGVector<float64_t> CConstMean::get_parameter_derivative(const CFeatures* features,
	const TParameter* param, index_t index)
{
	REQUIRE(features,"The features should NOT be NULL\n");
	REQUIRE(param,"The param should NOT be NULL\n");

	if (!strcmp(param->m_name, "mean"))
	{
		SGVector<float64_t> derivative(features->get_num_vectors());
		derivative.set_const(1.0);
		return derivative;
	}
	else
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name);
		return SGVector<float64_t>();
	}
}
