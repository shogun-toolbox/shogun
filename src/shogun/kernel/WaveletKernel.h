/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Siddharth Kherada
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WAVELETKERNEL_H___
#define _WAVELETKERNEL_H___

#include <lib/common.h>
#include <kernel/DotKernel.h>
#include <features/DotFeatures.h>

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
class CWaveletKernel: public CDotKernel
{
	public:
		/** default constructor  */
		CWaveletKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param Wdilation is Dilation coefficient
		 * @param Wtranslation is Translation coefficient
		 */
		CWaveletKernel(int32_t size, float64_t Wdilation, float64_t Wtranslation);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param size cache size
		 * @param Wdilation is Dilation coefficient
		 * @param Wtranslation is Translation coefficient
		 */
		CWaveletKernel(CDotFeatures* l, CDotFeatures* r, int32_t size,float64_t Wdilation, float64_t Wtranslation);

		virtual ~CWaveletKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type wavelet
		 */
		virtual EKernelType get_kernel_type() { return K_WAVELET; }

		/** return the kernel's name
		 *
		 * @return name Wavelet
		 */
		virtual const char* get_name() const { return "WaveletKernel"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

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
