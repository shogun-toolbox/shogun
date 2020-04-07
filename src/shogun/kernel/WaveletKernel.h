/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang
 */

#ifndef _WAVELETKERNEL_H___
#define _WAVELETKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{
/** @brief the class WaveletKernel
 *
 * It is defined as
 *
 * \f[
 * k({\bf x},({\bf x'})= \prod_{i=0}^{l}h(\frac{(x-c)}{a})\cdot h(\frac{(x'-c)}{a})
 * \f]
 *
 * Where \f$h(x)\f$ is the mother wavelet function
 *
 * \f[
 * h({\bf x}=cos(1.75*x)\cdot exp(\frac{(-x^2)}{2})
 * \f]
 *
 */
class WaveletKernel: public DotKernel
{
	public:
		/** default constructor  */
		WaveletKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param Wdilation is Dilation coefficient
		 * @param Wtranslation is Translation coefficient
		 */
		WaveletKernel(int32_t size, float64_t Wdilation, float64_t Wtranslation);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param size cache size
		 * @param Wdilation is Dilation coefficient
		 * @param Wtranslation is Translation coefficient
		 */
		WaveletKernel(const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r, int32_t size,float64_t Wdilation, float64_t Wtranslation);

		~WaveletKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		void cleanup() override;

		/** return what type of kernel we are
		 *
		 * @return kernel type wavelet
		 */
		EKernelType get_kernel_type() override { return K_WAVELET; }

		/** return the kernel's name
		 *
		 * @return name Wavelet
		 */
		const char* get_name() const override { return "WaveletKernel"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b) override;

	protected:
		/** h(x) is a mother wavelet function */
		inline float64_t MotherWavelet(float64_t h)
		{
			return cos(1.75*h)*exp(-h*h/2);
		}

	private:
		void init();

	protected:
		/** dilation coefficient */
		float64_t Wdilation;
		/** translation coefficient */
		float64_t Wtranslation;
};
}
#endif /* _WAVELETKERNEL_H__ */
