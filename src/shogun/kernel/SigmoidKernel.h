/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _SIGMOIDKERNEL_H___
#define _SIGMOIDKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{
namespace params {
	template <typename KernelType>
	class GammaFeatureNumberInit;
}
/** @brief The standard Sigmoid kernel computed on dense real valued features.
 *
 * Formally, it is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})=\mbox{tanh}(\gamma {\bf x}\cdot{\bf x'}+c)
 * \f]
 */
class SigmoidKernel: public DotKernel
{
	friend class params::GammaFeatureNumberInit<SigmoidKernel>;

	public:
		/** default constructor  */
		SigmoidKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param gamma gamma
		 * @param coef0 coefficient 0
		 */
		SigmoidKernel(int32_t size, float64_t gamma, float64_t coef0);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param size cache size
		 * @param gamma gamma
		 * @param coef0 coefficient 0
		 */
		SigmoidKernel(const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r, int32_t size,
			float64_t gamma, float64_t coef0);

		~SigmoidKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** clean up kernel */
		void cleanup() override;

		/** return what type of kernel we are
		 *
		 * @return kernel type SIGMOID
		 */
		EKernelType get_kernel_type() override { return K_SIGMOID; }

		/** return the kernel's name
		 *
		 * @return name Sigmoid
		 */
		const char* get_name() const override { return "SigmoidKernel"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b) override
		{
			return tanh(std::get<float64_t>(m_gamma)*DotKernel::compute(idx_a,idx_b)+coef0);
		}

	protected:
		/** gamma */
		AutoValue<float64_t> m_gamma = AutoValueEmpty{};
		/** coefficient 0 */
		float64_t coef0 = 0.0;
};
}
#endif /* _SIGMOIDKERNEL_H__ */
