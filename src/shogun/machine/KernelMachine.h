/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn, Soeren Sonnenburg, Tejas Jogi,
 *          Evgeniy Andreev, Evan Shelhamer, Yuyu Zhang, Chiyuan Zhang,
 *          Weijie Lin, Fernando Iglesias, Bjoern Esser, Thoralf Klein,
 *          Saurabh Goyal
 */

#ifndef _KERNEL_MACHINE_H__
#define _KERNEL_MACHINE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/SGVector.h>
#include <shogun/machine/NonParametricMachine.h>

namespace shogun
{
class Labels;
class BinaryLabels;
class RegressionLabels;
class Kernel;
class CustomKernel;
class Features;

/** @brief A generic KernelMachine interface.
 *
 * A kernel machine is defined as
 *  \f[
 *		f({\bf x})=\sum_{i=0}^{N-1} \alpha_i k({\bf x}, {\bf x_i})+b
 *	\f]
 *
 * where \f$N\f$ is the number of training examples
 * \f$\alpha_i\f$ are the weights assigned to each training example
 * \f$k(x,x')\f$ is the kernel
 * and \f$b\f$ the bias.
 *
 * Using an a-priori choosen kernel, the \f$\alpha_i\f$ and bias are determined
 * in a training procedure.
 */
class KernelMachine : public NonParametricMachine
{
	public:
		/** default constructor */
		KernelMachine();

		/** Convenience constructor to initialize a trained kernel
		 * machine
		 *
		 * @param k kernel
		 * @param alphas vector of alpha weights
		 * @param svs indices of examples, i.e. i's for x_i
		 * @param b bias term
		 */
		KernelMachine(const std::shared_ptr<Kernel>& k, const SGVector<float64_t> alphas, const SGVector<int32_t> svs, float64_t b);

		/** copy constructor
		 * @param machine machine having parameters to copy
		 */
		KernelMachine(const std::shared_ptr<KernelMachine>& machine);

		/** destructor */
		~KernelMachine() override;

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		const char* get_name() const override { return "KernelMachine"; }

		/** set kernel
		 *
		 * @param k kernel
		 */
		void set_kernel(std::shared_ptr<Kernel> k);

		/** get kernel
		 *
		 * @return kernel
		 */
		std::shared_ptr<Kernel> get_kernel();

		/** set batch computation enabled
		 *
		 * @param enable if batch computation shall be enabled
		 */
		void set_batch_computation_enabled(bool enable);

		/** check if batch computation is enabled
		 *
		 * @return if batch computation is enabled
		 */
		bool get_batch_computation_enabled();

		/** set linadd enabled
		 *
		 * @param enable if linadd shall be enabled
		 */
		void set_linadd_enabled(bool enable);

		/** check if linadd is enabled
		 *
		 * @return if linadd is enabled
		 */
		bool get_linadd_enabled();

		/** set state of bias
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		void set_bias_enabled(bool enable_bias);

		/** get state of bias
		 *
		 * @return state of bias
		 */
		bool get_bias_enabled();

		/** get bias
		 *
		 * @return bias
		 */
		float64_t get_bias();

		/** set bias to given value
		 *
		 * @param bias new bias
		 */
		void set_bias(float64_t bias);

		/** get support vector at given index
		 *
		 * @param idx index of support vector
		 * @return support vector
		 */
		int32_t get_support_vector(int32_t idx);

		/** get alpha at given index
		 *
		 * @param idx index of alpha
		 * @return alpha
		 */
		float64_t get_alpha(int32_t idx);

		/** set support vector at given index to given value
		 *
		 * @param idx index of support vector
		 * @param val new value of support vector
		 * @return if operation was successful
		 */
		bool set_support_vector(int32_t idx, int32_t val);

		/** set alpha at given index to given value
		 *
		 * @param idx index of alpha vector
		 * @param val new value of alpha vector
		 * @return if operation was successful
		 */
		bool set_alpha(int32_t idx, float64_t val);

		/** get number of support vectors
		 *
		 * @return number of support vectors
		 */
		int32_t get_num_support_vectors();

		/** set alphas to given values
		 *
		 * @param alphas float vector with all alphas to set
		 */
		void set_alphas(SGVector<float64_t> alphas);

		/** set support vectors to given values
		 *
		 * @param svs integer vector with all support vectors indexes to set
		 */
		void set_support_vectors(SGVector<int32_t> svs);

		/** @return all support vectors */
		SGVector<int32_t> get_support_vectors();

		/** @return vector of alphas */
		SGVector<float64_t> get_alphas();

		/** create new model
		 *
		 * @param num number of alphas and support vectors in new model
		 */
		bool create_new_model(int32_t num);

		/** initialise kernel optimisation
		 *
		 * @return if operation was successful
		 */
		bool init_kernel_optimization();

		/** apply kernel machine to data
		 * for regression task
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL) override;

		/** apply kernel machine to data
		 * for binary classification task
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		std::shared_ptr<BinaryLabels> apply_binary(std::shared_ptr<Features> data=NULL) override;

		/** apply kernel machine to one example
		 *
		 * @param num which example to apply to
		 * @return classified value
		 */
		float64_t apply_one(int32_t num) override;

		/** Stores feature data of the SV indices and sets it to the lhs of the
		 * underlying kernel. Then, all SV indices are set to identity.
		 *
		 * May be overwritten by subclasses in case the model should be stored
		 * differently.
		 */
		virtual void store_model_features();

	protected:

		/** apply get outputs
		 *
		 * @param data features to compute outputs
		 * @return outputs
		 */
		SGVector<float64_t> apply_get_outputs(const std::shared_ptr<Features>& data);


	private:
		/** register parameters and do misc init */
		void init();

	protected:
		/** kernel */
		std::shared_ptr<Kernel> kernel;

		/** if batch computation is enabled */
		bool use_batch_computation;

		/** if linadd is enabled */
		bool use_linadd;

		/** if bias shall be used */
		bool use_bias;

		/**  bias term b */
		float64_t m_bias;

		/** coefficients alpha */
		SGVector<float64_t> m_alpha;

		/** array of ``support vectors'' (indices of feature objects) */
		SGVector<int32_t> m_svs;
};
}
#endif /* _KERNEL_MACHINE_H__ */
