/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Michal Uricar, Sergey Lisitsyn, Evan Shelhamer, 
 *          Thoralf Klein, Fernando Iglesias, Fredrik Hallgren, Sanuj Sharma
 */

#ifndef _KERNELRIDGEREGRESSION_H__
#define _KERNELRIDGEREGRESSION_H__

#include <shogun/lib/config.h>
#include <shogun/regression/Regression.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/machine/KernelMachine.h>

namespace shogun
{

/** @brief Class KernelRidgeRegression implements Kernel Ridge Regression - a regularized least square
 * method for classification and regression.
 *
 * It is similar to support vector machines (cf. SVM). However in contrast to
 * SVMs a different objective is optimized that leads to a dense solution (thus
 * not only a few support vectors are active in the end but all training
 * examples). This makes it only applicable to rather few (a couple of
 * thousand) training examples. In case a linear kernel is used RR is closely
 * related to Fishers Linear Discriminant (cf. LDA).
 *
 * Internally (for linear kernels) it is solved via minimizing the following system
 *
 * \f[
 * \frac{1}{2}\left(\sum_{i=1}^N(y_i-{\bf w}\cdot {\bf x}_i)^2 + \tau||{\bf w}||^2\right)
 * \f]
 *
 * which boils down to solving a linear system
 *
 * \f[
 * {\bf w} = \left(\tau {\bf I}+ \sum_{i=1}^N{\bf x}_i{\bf x}_i^T\right)^{-1}\left(\sum_{i=1}^N y_i{\bf x}_i\right)
 * \f]
 *
 * and in the kernel case
 * \f[
 * {\bf \alpha}=\left({\bf K}+\tau{\bf I}\right)^{-1}{\bf y}
 * \f]
 * where K is the kernel matrix and y the vector of labels. The expressed
 * solution can again be written as a linear combination of kernels (cf.
 * KernelMachine) with bias \f$b=0\f$.
 */
class KernelRidgeRegression : public KernelMachine
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** default constructor */
		KernelRidgeRegression();

		/** constructor
		 *
		 * @param tau regularization constant tau
		 * @param k kernel
		 * @param lab labels
		 */
		KernelRidgeRegression(float64_t tau, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab);

		/** default destructor */
		~KernelRidgeRegression() override {}

		/** set regularization constant
		 *
		 * @param tau new tau
		 */
		inline virtual void set_tau(float64_t tau) { m_tau = tau; };

		/** set convergence precision for gauss seidel method
		 *
		 * @param epsilon new epsilon
		 */
		inline void set_epsilon(float64_t epsilon) { m_epsilon = epsilon; }

		/** load regression from file
		 *
		 * @param srcfile file to load from
		 * @return if loading was successful
		 */
		virtual bool load(FILE* srcfile);

		/** save regression to file
		 *
		 * @param dstfile file to save to
		 * @return if saving was successful
		 */
		virtual bool save(FILE* dstfile);

		/** get classifier type
		 *
		 * @return classifier type KernelRidgeRegression
		 */
		EMachineType get_classifier_type() override
		{
			return CT_KERNELRIDGEREGRESSION;
		}

		/** @return object name */
		const char* get_name() const override { return "KernelRidgeRegression"; }

	protected:
		/** Train regression
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based regressors are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		bool train_machine(std::shared_ptr<Features> data=NULL) override;

		/** Train regression using Cholesky decomposition.
		 * Assumes that m_alpha is already allocated.
		 *
		 *
		 * @return boolean to indicate success
		 */
		virtual bool solve_krr_system();

	private:
		void init();

	protected:
		/** regularization parameter tau */
		float64_t m_tau;

	private:
		/** epsilon constant */
		float64_t m_epsilon;

};
}

#endif // _KERNELRIDGEREGRESSION_H__
