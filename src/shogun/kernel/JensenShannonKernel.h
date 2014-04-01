/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef _JENSENSHANNONKERNEL_H___
#define _JENSENSHANNONKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief The Jensen-Shannon kernel operating on real-valued vectors computes
 * the Jensen-Shannon distance between the features. Often used in computer vision.
 *
 * It is defined as
 * \f[
 * k({\bf x},{\bf x'})= \sum_{i=0}^{l} \frac{x_i}{2} \log_2\frac{x_i+x'_i}{x_i} + \frac{x'_i}{2} \log_2\frac{x_i+x'_i}{x'_i}
 * \f]
 * */
class CJensenShannonKernel: public CDotKernel
{
	public:
		/** default constructor  */
		CJensenShannonKernel();

		/** constructor
		 *
		 * @param size cache size
		 */
		CJensenShannonKernel(int32_t size);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param size cache size
		 */
		CJensenShannonKernel(
			CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r,
			int32_t size=10);

		virtual ~CJensenShannonKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** return what type of kernel we are
		 *
		 * @return kernel type JENSENSHANNON
		 */
		virtual EKernelType get_kernel_type() { return K_JENSENSHANNON; }

		/** return the kernel's name
		 *
		 * @return name JensenShannonKernel
		 */
		virtual const char* get_name() const { return "JensenShannonKernel"; }

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

};
}
#endif /* _JENSENSHANNONKERNEL_H___ */
