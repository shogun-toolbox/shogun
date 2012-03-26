/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Soeren Sonnenburg
 * Copyright (C) 2011 Soeren Sonnenburg
 */

#ifndef _DIRECTORKERNEL_H___
#define _DIRECTORKERNEL_H___

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{
class CDirectorKernel: public CKernel
{
	public:
		/** default constructor
		 *
		 */
		CDirectorKernel() : CKernel()
		{
		}

		/** constructor
		 *
		 */
		CDirectorKernel(int32_t size) : CKernel(size)
		{
		}

		/** default constructor
		 *
		 */
		virtual ~CDirectorKernel()
		{
			cleanup();
		}

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
		}

		/** clean up kernel */
		virtual void cleanup()
		{
		}

		/** return what type of kernel we are
		 *
		 * @return kernel type DIRECTOR
		 */
		virtual EKernelType get_kernel_type() { return K_DIRECTOR; }

		 /** return what type of features kernel can deal with
		  *
		  * @return feature type ANY
		  */
		virtual EFeatureType get_feature_type() { return F_ANY; }

		 /** return what class of features kernel can deal with
		  *
		  * @return feature class ANY
		  */
		virtual EFeatureClass get_feature_class() { return C_ANY; }

		/** return the kernel's name
		 *
		 * @return name Director
		 */
		inline virtual const char* get_name() const { return "DirectorKernel"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b)
		{
			SG_ERROR("Compute method of Director Kernel needs to be overridden.\n");
			return 0;
		}
};
}
#endif /* _GAUSSIANKERNEL_H__ */
