/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _GAUSSIANKERNEL_H___
#define _GAUSSIANKERNEL_H___

#include "lib/common.h"
#include "kernel/DotKernel.h"
#include "features/DotFeatures.h"

namespace shogun
{
	class CDotFeatures;
/** @brief The well known Gaussian kernel (swiss army knife for SVMs)
 * computed on CDotFeatures.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau})
 * \f]
 *
 * where \f$\tau\f$ is the kernel width.
 */
class CGaussianKernel: public CDotKernel
{
	public:
		/** default constructor
		 *
		 */
		CGaussianKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 */
		CGaussianKernel(int32_t size, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param size cache size
		 */
		CGaussianKernel(CDotFeatures* l, CDotFeatures* r,
			float64_t width, int32_t size=10);

		virtual ~CGaussianKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type GAUSSIAN
		 */
		virtual EKernelType get_kernel_type() { return K_GAUSSIAN; }

		/** return the kernel's name
		 *
		 * @return name Gaussian
		 */
		inline virtual const char* get_name() const { return "GaussianKernel"; }

		/** return the kernel's width
		 *
		 * @return kernel width
		 */
		inline virtual float64_t get_width(void) const {
			return width;
		}

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

		/** Can (optionally) be overridden to post-initialize some
		 *  member variables which are not PARAMETER::ADD'ed.  Make
		 *  sure that at first the overridden method
		 *  BASE_CLASS::LOAD_SERIALIZABLE_POST is called.
		 *
		 *  @exception ShogunException Will be thrown if an error
		 *                             occurres.
		 */
		virtual void load_serializable_post(void) throw (ShogunException);

	private:
		/** helper function to compute quadratic terms in
		 * (a-b)^2 (== a^2+b^2-2ab)
		 *
		 * @param buf buffer to store squared terms (will be allocated)
		 * @param df dot feature object based on which k(i,i) is computed
		 * */
		void precompute_squared();

		/** helper function to compute quadratic terms in
		 * (a-b)^2 (== a^2+b^2-2ab)
		 *
		 * @param buf buffer to store squared terms (will be allocated)
		 * @param df dot feature object based on which k(i,i) is computed
		 * */
		void precompute_squared_helper(float64_t* &buf, CDotFeatures* df);

		void init();

	protected:
		/** width */
		float64_t width;
		/** squared left-hand side */
		float64_t* sq_lhs;
		/** squared right-hand side */
		float64_t* sq_rhs;
};
}
#endif /* _GAUSSIANKERNEL_H__ */
