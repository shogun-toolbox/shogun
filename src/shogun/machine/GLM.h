/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 * 
 * Both the documentation and the code is heavily inspired by pyGLMnet.: https://github.com/glm-tools/pyglmnet/
 *
 * Authors: Tej Sukhatme
 */


#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/IterativeMachine.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/distributions/Distribution.h>
#include <shogun/optimization/GradientDescendUpdater.h>
#include <shogun/optimization/ConstLearningRate.h>
#include <shogun/optimization/ElasticNetPenalty.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/optimization/FirstOrderCostFunction.h>

namespace shogun
{
    enum GLM_DISTRIBUTION
	{
        //TODO GAUSSIAN,
        //TODO BINOMIAL,
        //TODO GAMMA,
        //TODO SOFTPLUS,
        //TODO PROBIT,
        POISSON
	};

class DotFeatures;
class Features;
class RegressionLabels;

/** @brief Class for estimating regularized generalized linear models (GLM).
 *   The regularized GLM minimizes the penalized negative log likelihood:
 *
 *  This used Elastic-net penalty which defaults to the ridge penalty when
 *  alpha = 0 and defaults to the lasso penalty when alpha = 1.
 *
 * */
class GLM : public RandomMixin<IterativeMachine<LinearMachine>>
{
 	public:
		friend class GLMCostFunction;
		
		/** default constructor */
		GLM();

 		GLM(GLM_DISTRIBUTION distribution, float64_t alpha, float64_t lambda, float64_t learning_rate, int32_t max_iterations, float64_t tolerance, float64_t eta);

 		/** destructor */
 		~GLM() override {}

		/** apply linear machine to data
		 * for regression problem
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL) override;

 		/** Returns the name of the SGSerializable instance.  It MUST BE
 		 *  the CLASS NAME without the prefixed `C'.
 		 *
 		 * @return name of the SGSerializable
 		 */
 		const char* get_name() const override { return "GLM"; }

		void set_tau(SGMatrix<float64_t> tau);

		SGMatrix<float64_t> get_tau();

	protected:

		void init_model(const std::shared_ptr<Features>& data) override;
		
		void iteration() override;

	private:

		SGVector<float64_t> non_linearity(const SGVector<float64_t> z);

	protected:

		/** Distribution type */
		GLM_DISTRIBUTION distribution = POISSON;

		/** a threshold parameter that linearizes the exp() function above eta. */
		float64_t m_eta = 2.0;

		/** regularization parameter :math:`\\lambda` of penalty term. */
		float64_t m_lambda = 0.1;

		/** the weighting between L1 penalty and L2 penalty term of the loss function. */
		float64_t m_alpha = 0.5;

		/** convergence threshold or stopping criteria. Optimization loop will stop when relative change in parameter norm is below the threshold. */
		float64_t m_tolerance = 1e-6;

		bool m_compute_bias = true;

		std::shared_ptr<GradientDescendUpdater> m_gradient_updater;

		std::shared_ptr<ConstLearningRate> m_learning_rate;

		std::shared_ptr<ElasticNetPenalty> m_penalty;


};

class GLMCostFunction: public FirstOrderCostFunction
{
public:

	friend class GLM;

	void set_target(const std::shared_ptr<GLM>&obj)
	{
		if(m_obj != obj)
			m_obj=obj;
	}

	void unset_target()
	{
		m_obj=NULL;
	}

	virtual float64_t get_cost()
	{
		//TODO
		return 0.0;
	}

	virtual SGVector<float64_t> obtain_variable_reference()
	{
		require(m_obj,"Object not set");
		return m_obj->m_w;
	}

	virtual SGVector<float64_t> get_gradient()
	{
		auto X = m_obj->LinearMachine::features->get_computed_dot_feature_matrix();
		auto y = m_obj->m_labels->get_values();
		SGVector<float64_t> w_old(m_obj->m_w);

		auto n_samples = y.vlen;
	
		auto z = compute_z(X, m_obj->m_w, m_obj->bias);
		auto mu = non_linearity(z);
		auto grad_mu = gradient_non_linearity(z);

		SGVector<float64_t> grad_w(m_obj->m_w.vlen);
		SGVector<float64_t> a;
		switch (m_obj->distribution)
		{
		case POISSON:
			a = linalg::element_prod(y, grad_mu);
			for(int i = 0; i<y.vlen; i++)
				a[i] /= mu[i];
			linalg::transpose_matrix(linalg::add(linalg::matrix_prod(SGMatrix(grad_mu), X, true, false), (linalg::matrix_prod(SGMatrix(a), X, true, false)), 1.0, -1.0));
			break;
	
		default:
			error("Distribution type {} not implemented.", m_obj->distribution);
			break;
		}

		grad_w = linalg::scale(grad_w, 1.0/n_samples);	
		return grad_w;
	}

	virtual float64_t get_gradient_bias()
	{
		auto X = m_obj->LinearMachine::features->get_computed_dot_feature_matrix();
		auto y = m_obj->m_labels->get_values();

		auto n_samples = y.vlen;
		auto z = compute_z(X, m_obj->m_w, m_obj->bias);
		auto mu = non_linearity(z);
		auto grad_mu = gradient_non_linearity(z);

		float64_t grad_bias = 0;
		switch (m_obj->distribution)
		{
		case POISSON:
			for (int i = 0; i < grad_mu.vlen; i++)
			{
				grad_bias += grad_mu[i];
				grad_bias -= y[i] * grad_mu[i] / mu[i];
			}
			break;
	
		default:
			error("Distribution type {} not implemented.", m_obj->distribution);
			break;
		}
		grad_bias /= n_samples;

		return grad_bias;
	}

	virtual const char* get_name() const { return "GLMCostFunction"; }

private:

	virtual const SGVector<float64_t> compute_z(const SGMatrix<float64_t> X, const SGVector<float64_t> w, const float64_t bias)
	{
		return linalg::matrix_prod(X, w, false);
	}

	virtual const SGVector<float64_t> non_linearity(const SGVector<float64_t> z)
	{
		SGVector<float64_t> result;
		float64_t l_bias = 0;
		switch (m_obj->distribution)
		{
		case POISSON:
			result = SGVector<float64_t>(z);

			if(m_obj->m_compute_bias)
				l_bias = (1 - m_obj->m_eta) * std::exp(m_obj->m_eta);

			for (int i = 0; i < z.vlen; i++)
			{
				if(z[i]>m_obj->m_eta)
					result[i] = z[i] * std::exp(m_obj->m_eta) + l_bias;
				else
					result[i] = std::exp(z[i]);
			}
			break;
	
		default:
			error("Distribution type {} not implemented.", m_obj->distribution);
			break;
		}
		return result;
	}

	virtual const SGVector<float64_t> gradient_non_linearity(const SGVector<float64_t> z)
	{
		SGVector<float64_t> result;
		switch (m_obj->distribution)
		{
		case POISSON:
			result = SGVector<float64_t>(z);
			for (int i = 0; i < z.vlen; i++)
			{
				if(z[i]>m_obj->m_eta)
					result[i] = std::exp(m_obj->m_eta);
				else
					result[i] = std::exp(z[i]);
			}
			break;

		default:
			error("Distribution type {} not implemented.", m_obj->distribution);
			break;
		}
		return result;
	}

	std::shared_ptr<GLM>m_obj;
};
}
// #endif
