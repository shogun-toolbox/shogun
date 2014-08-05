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
#ifndef _MIXTUREMODEL_H__
#define _MIXTUREMODEL_H__

#include <shogun/lib/config.h>
#include <shogun/distributions/Distribution.h>

namespace shogun
{

/** @brief This is the generic class for mixture models. The final distribution is a 
 * mixture of various simple distributions supplied by the user.
 *
 * \f$ f_X(x) = \Sigma_{m=1}^{K} w_m.g_m(x,\lambda_m)\f$
 */
class CMixtureModel : public CDistribution
{
	public:
		/* default constructor */
		CMixtureModel();

		/** constructor
		 * Note: Automatic initialization of parameters of components is not possible yet 
		 *
		 * @param components individual distributions forming the mixture (parameters must be initialized)
		 * @param weights initial \f$w_m\f$ ie. weights of individual distributions
		 */
		CMixtureModel(CDynamicObjectArray* components, SGVector<float64_t> weights);

		/* destructor */
		~CMixtureModel();

		/** @return object name */
		virtual const char* get_name() const { return "MixtureModel"; }

		/** learn mixture model
		 *
		 * @param data training data
		 * @return whether training was successful
		 */
		bool train(CFeatures* data=NULL);

		/** get number of parameters in model
		 *
		 * @return number of parameters in model
		 */
		int32_t get_num_model_parameters() { return 1; }

		/** get model parameter (logarithmic)
		 *
		 * @return model parameter (logarithmic)
		 */
		float64_t get_log_model_parameter(int32_t num_param=1);

		/** get partial derivative of likelihood function (logarithmic)
		 *
		 * @param num_param derivative against which param
		 * @param num_example which example
		 * @return derivative of likelihood (logarithmic)
		 */
		virtual float64_t get_log_derivative(int32_t num_param, int32_t num_example);

		/** compute log likelihood for example
		 *
		 * @param num_example which example
		 * @return log likelihood for example
		 */
		virtual float64_t get_log_likelihood_example(int32_t num_example);

		/** get weights
		 *
		 * @return weights
		 */
		SGVector<float64_t> get_weights() const;

		/** set weights
		 *
		 * @param weights mixing weights
		 */
		void set_weights(SGVector<float64_t> weights);

		/** get components
		 *
		 * @return components
		 */
		CDynamicObjectArray* get_components() const;

		/** set components
		 *
		 * @param components mixture components
		 */
		void set_components(CDynamicObjectArray* components);

		/** get number of components
		 *
		 * @return number of mixture components
		 */
		index_t get_num_components() const;

		/** Getter for mixture components
		 *
		 * @param index index of component
		 * @return component at index
		 */
		CDistribution* get_component(index_t index) const;

		/** set max iterations in EM
		 *
		 * @param max_iters maximum number of iterations allowed
		 */
		void set_max_iters(int32_t max_iters);

		/** get max iterations in EM
		 *
		 * @return max_iters maximum number of iterations allowed
		 */
		int32_t get_max_iters() const;

		/** set convergence tolerance criterion for EM
		 *
		 *	@param epsilon convergence tolerance
		 */
		void set_convergence_tolerance(float64_t epsilon);

		/** get convergence tolerance criterion for EM
		 *
		 *	@return epsilon convergence tolerance
		 */
		float64_t get_convergence_tolerance() const;

		/** sample from model
		 *
		 * @return sample
		 */
		SGVector<float64_t> sample();

		/** cluster point
		 *
		 * @return log likelihood of belonging to clusters and the log likelihood of being generated by this mixture model
		 * (The length of the returned vector is number of components + 1)
		 */
		SGVector<float64_t> cluster(SGVector<float64_t> point);		

	private:
		/** initialize and register members */
		void init();

	private:
		/** array of components */
		CDynamicObjectArray* m_components;

		/** weights */
		SGVector<float64_t> m_weights;

		/** max_iterations of EM */
		int32_t m_max_iters;

		/** convergence tolerance */
		float64_t m_conv_tol;
};
}
#endif /* _MIXTUREMODEL_H__ */