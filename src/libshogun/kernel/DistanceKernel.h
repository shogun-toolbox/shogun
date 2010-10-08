/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Christian Gehl
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"


#ifndef _DISTANCEKERNEL_H___
#define _DISTANCEKERNEL_H___

#include "lib/common.h"
#include "kernel/Kernel.h"
#include "distance/Distance.h"

namespace shogun
{
	class CDistance;

/** @brief The Distance kernel takes a distance as input.
 *
 * It turns a distance into something kernel like by computing
 *
 * \f[
 *     k({\bf x}, {\bf x'}) = e^{-\frac{dist({\bf x}, {\bf x'})}{width}}
 * \f]
 */
class CDistanceKernel: public CKernel
{
	public:
		/** default constructor  */
		CDistanceKernel(void);

		/** constructor
		 *
		 * @param cache cache size
		 * @param width width
		 * @param dist distance
		 */
		CDistanceKernel(int32_t cache, float64_t width, CDistance* dist);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param dist distance
		 */
		CDistanceKernel(
			CFeatures *l, CFeatures *r, float64_t width, CDistance* dist);

		virtual ~CDistanceKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** return what type of kernel we are
		 *
		 * @return kernel type DISTANCE
		 */
		inline virtual EKernelType get_kernel_type() { return K_DISTANCE; }
		/** return feature type the kernel can deal with
		 *
		 * @return feature type of distance used
		 */
		inline virtual EFeatureType get_feature_type() { return distance->get_feature_type(); }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class of distance used
		 */
		inline virtual EFeatureClass get_feature_class() { return distance->get_feature_class(); }

		/** return the kernel's name
		 *
		 * @return name Distance
		 */
		inline virtual const char* get_name() const { return distance->get_name(); }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b);

	private:
		/** distance */
		CDistance* distance;
		/** width */
		float64_t width;
};
}
#endif /* _DISTANCEKERNEL_H__ */
