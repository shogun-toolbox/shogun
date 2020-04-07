/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Alesis Novik, Sergey Lisitsyn, Heiko Strathmann,
 *          Evgeniy Andreev, Evan Shelhamer, Wuwei Lin, Yori Zwols
 */
#ifndef _GMM_H__
#define _GMM_H__

#include <shogun/lib/config.h>

#include <shogun/distributions/Distribution.h>
#include <shogun/distributions/Gaussian.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/RandomMixin.h>

#include <vector>

namespace shogun
{
/** @brief Gaussian Mixture Model interface.
 *
 * Takes input of number of Gaussians to fit and a covariance type to use.
 * Parameter estimation is done using either the Expectation-Maximization or
 * Split-Merge Expectation-Maximization algorithms. To estimate the GMM
 * parameters, the train(...) method has to be run to set the training data
 * and then either train_em(...) or train_smem(...) to do the actual
 * estimation.
 * The EM algorithm is described here:
 * http://en.wikipedia.org/wiki/Expectation-maximization_algorithm
 * The SMEM algorithm is described here:
 * http://mlg.eng.cam.ac.uk/zoubin/papers/uedanc.pdf
 */
class GMM : public RandomMixin<Distribution>
{
	public:
		/** default constructor */
		GMM();
		/** constructor
		 *
		 * @param n number of Gaussians
		 * @param cov_type covariance type
		 */
		GMM(int32_t n, ECovType cov_type=FULL);
		/** constructor
		 *
		 * @param components GMM components
		 * @param coefficients mixing coefficients
		 * @param copy true if should be copied
		 */
		GMM(std::vector<std::shared_ptr<Gaussian>> components, SGVector<float64_t> coefficients,
				bool copy=false);
		~GMM() override;

		/** cleanup */
		void cleanup();

		/** set training data for use with EM or SMEM
		 *
		 * @param data training data
		 *
		 * @return true
		 */
		bool train(std::shared_ptr<Features> data=NULL) override;

		/** learn model using EM
		 *
		 * @param min_cov minimum covariance
		 * @param max_iter maximum iterations
		 * @param min_change minimum change in log likelihood
		 *
		 * @return log likelihood of training data
		 */
		float64_t train_em(float64_t min_cov=1e-9, int32_t max_iter=1000,
				float64_t min_change=1e-9);

		/** learn model using SMEM
		 *
		 * @param max_iter maximum SMEM iterations
		 * @param max_cand maximum split-merge candidates
		 * @param min_cov minimum covariance
		 * @param max_em_iter maximum iterations for EM
		 * @param min_change minimum change in log likelihood
		 *
		 * @return log likelihood of training data
		 */
		float64_t train_smem(int32_t max_iter=100, int32_t max_cand=5,
				float64_t min_cov=1e-9, int32_t max_em_iter=1000,
				float64_t min_change=1e-9);

		/** maximum likelihood estimation
		 *
		 * @param alpha point assignment
		 * @param min_cov minimum covariance
		 */
		void max_likelihood(SGMatrix<float64_t> alpha, float64_t min_cov);

		/** get number of parameters in model
		 *
		 * @return number of parameters in model
		 */
		int32_t get_num_model_parameters() override;

		/** get model parameter (logarithmic)
		 *
		 * @return model parameter (logarithmic) if num_param < m_dim returns
		 * an element from the mean, else return an element from the covariance
		 */
		float64_t get_log_model_parameter(int32_t num_param) override;

		/** @return number of mixture components */
		index_t get_num_components() const;

		/** Getter for mixture components
		 * @param index index of component
		 * @return component at index
		 */
		std::shared_ptr<Distribution> get_component(index_t index) const;

		/** get partial derivative of likelihood function (logarithmic)
		 *
		 * @param num_param derivative against which param
		 * @param num_example which example
		 * @return derivative of likelihood (logarithmic)
		 */
		float64_t get_log_derivative(
			int32_t num_param, int32_t num_example) override;

		/** compute log likelihood for example
		 *
		 * abstract base method
		 *
		 * @param num_example which example
		 * @return log likelihood for example
		 */
		float64_t get_log_likelihood_example(int32_t num_example) override;

		/** compute likelihood for example
		 *
		 * abstract base method
		 *
		 * @param num_example which example
		 * @return likelihood for example
		 */
		float64_t get_likelihood_example(int32_t num_example) override;

		/** get nth mean
		 *
		 * @param num index of mean to retrieve
		 *
		 * @return mean
		 */
		virtual SGVector<float64_t> get_nth_mean(int32_t num);

		/** set nth mean
		 *
		 * @param mean new mean
		 * @param num index mean to set
		 */
		virtual void set_nth_mean(SGVector<float64_t> mean, int32_t num);

		/** get nth covariance
		 *
		 * @param num index of covariance to retrieve
		 *
		 * @return covariance
		 */
		virtual SGMatrix<float64_t> get_nth_cov(int32_t num);

		/** set nth covariance
		 *
		 * @param cov new covariance
		 * @param num index of covariance to set
		 */
		virtual void set_nth_cov(SGMatrix<float64_t> cov, int32_t num);

		/** get coefficients
		 *
		 * @return coeffiecients
		 */
		virtual SGVector<float64_t> get_coef();

		/** set coefficients
		 *
		 * @param coefficients mixing coefficients
		 */
		virtual void set_coef(const SGVector<float64_t> coefficients);

		/** get components
		 *
		 * @return components
		 */
		virtual std::vector<std::shared_ptr<Gaussian>> get_comp();

		/** set components
		 *
		 * @param components Gaussian components
		 */
		virtual void set_comp(std::vector<std::shared_ptr<Gaussian>> components);

		/** sample from model
		 *
		 * @return sample
		 */
		SGVector<float64_t> sample();

		/** cluster point
		 *
		 * @return log likelihood of belonging to clusters and the log likelihood of being generated by this GMM
		 * The length of the returned vector is number of components + 1
		 */
		SGVector<float64_t> cluster(SGVector<float64_t> point);

		/** @return object name */
		const char* get_name() const override { return "GMM"; }

	private:
		/** 1NN assignment initialization
		 *
		 * @param init_means initial means
		 *
		 * @return initial alphas
		 */
		SGMatrix<float64_t> alpha_init(SGMatrix<float64_t> init_means);

		/** Initialize parameters for serialization */
		void register_params();

		/** apply the partial EM algorithm on 3 components
		 *
		 * @param comp1 index of first component
		 * @param comp2 index of second component
		 * @param comp3 index of third component
		 * @param min_cov minimum covariance
		 * @param max_em_iter maximum iterations for EM
		 * @param min_change minimum change in log likelihood
		 */
		void partial_em(int32_t comp1, int32_t comp2, int32_t comp3,
				float64_t min_cov, int32_t max_em_iter, float64_t min_change);

	protected:
		/** Mixture components */
		std::vector<std::shared_ptr<Gaussian>> m_components;
		/** Mixture coefficients */
		SGVector<float64_t> m_coefficients;
};
}
#endif //_GMM_H__
