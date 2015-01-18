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
#ifndef _DISCRETEDISTRIBUTION_H__
#define _DISCRETEDISTRIBUTION_H__

#include <shogun/lib/config.h>
#include <shogun/distributions/Distribution.h>

namespace shogun
{

/** @brief This is the base interface class for all discrete distributions. 
 */
class CDiscreteDistribution : public CDistribution
{
	public:
		/* constructor */
		CDiscreteDistribution() : CDistribution() {	};

		/* destructor */
		~CDiscreteDistribution() {	};

		/** learn distribution
		 *
		 * @param data training data
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL)=0;

		/** get number of parameters in model
		 *
		 * abstract base method
		 *
		 * @return number of parameters in model
		 */
		virtual int32_t get_num_model_parameters()=0;

		/** get model parameter (logarithmic)
		 *
		 * abstract base method
		 *
		 * @return model parameter (logarithmic)
		 */
		virtual float64_t get_log_model_parameter(int32_t num_param)=0;

		/** get partial derivative of likelihood function (logarithmic)
		 *
		 * abstract base method
		 *
		 * @param num_param derivative against which param
		 * @param num_example which example
		 * @return derivative of likelihood (logarithmic)
		 */
		virtual float64_t get_log_derivative(int32_t num_param, int32_t num_example)=0;

		/** compute log likelihood for example
		 *
		 * abstract base method
		 *
		 * @param num_example which example
		 * @return log likelihood for example
		 */
		virtual float64_t get_log_likelihood_example(int32_t num_example)=0;

		/** update parameters in the em maximization step for mixture model of which
		 * this distribution is a part
		 *
		 * abstract base method
		 *
		 * @param alpha_k "belongingness" values of various data points
		 */
		virtual void update_params_em(SGVector<float64_t> alpha_k)=0;
};
}
#endif /* _DISCRETEDISTRIBUTION_H__ */