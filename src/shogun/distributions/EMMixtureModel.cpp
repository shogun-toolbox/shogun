/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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
 */

#include <shogun/distributions/EMMixtureModel.h>
#include <shogun/distributions/Distribution.h> 
#include <shogun/mathematics/Math.h>

using namespace shogun;

CEMMixtureModel::CEMMixtureModel()
: CEMBase <MixModelData>()
{ }

CEMMixtureModel::~CEMMixtureModel()
{ }

float64_t CEMMixtureModel::expectation_step()
{
	float64_t log_likelihood=0;
	// for each data point
	for (int32_t i=0;i<data.alpha.num_rows;i++)
	{
		SGVector<float64_t> alpha_ij(data.alpha.num_cols);
		// for each component
		for (int32_t j=0;j<data.alpha.num_cols;j++)
		{
			CDistribution* jth_component=CDistribution::obtain_from_generic(data.components->get_element(j));
			alpha_ij[j]=CMath::log(data.weights[j])+jth_component->get_log_likelihood_example(i);
			SG_UNREF(jth_component);
		};

		float64_t normalize=CMath::log_sum_exp(alpha_ij);
		log_likelihood+=normalize;

		// fill row of alpha
		for (int32_t j=0;j<data.alpha.num_cols;j++)
			data.alpha(i,j)=CMath::exp(alpha_ij[j]-normalize);		
	}

	return log_likelihood;
}

void CEMMixtureModel::maximization_step()
{
	// for each component
	float64_t* alpha_j=NULL;
	float64_t sum_weights=0;
	for (int32_t j=0;j<data.alpha.num_cols;j++)
	{
		CDistribution* jth_component=CDistribution::obtain_from_generic(data.components->get_element(j));

		// update mean covariance of components
		alpha_j=data.alpha.matrix+j*data.alpha.num_rows;
		float64_t weight_j=jth_component->update_params_em(alpha_j,data.alpha.num_rows);

		// update weights
		sum_weights+=weight_j;
		data.weights[j]=weight_j;

		SG_UNREF(jth_component);
	}

	// update weights - normalization
	for (int32_t j=0;j<data.alpha.num_cols;j++)
		data.weights[j]/=sum_weights;
}
