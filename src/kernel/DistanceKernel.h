/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Christian Gehl
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"


#ifndef _DISTANCEKERNEL_H___
#define _DISTANCEKERNEL_H___

#include "lib/common.h"
#include "kernel/Kernel.h"
#include "distance/Distance.h"

/** kernel Distance */
class CDistanceKernel: public CKernel
{
	public:
		/** constructor
		 *
		 * @param cache cache size
		 * @param width width
		 * @param dist distance
		 */
		CDistanceKernel(INT cache, DREAL width, CDistance* dist);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param dist distance
		 */
		CDistanceKernel(CFeatures *l, CFeatures *r, DREAL width, CDistance* dist);

		virtual ~CDistanceKernel();

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
		inline virtual const CHAR* get_name() { return distance->get_name(); }

		/** load kernel init_data
		 *
		 * @param src file to load from
		 * @return if loading was successful
		 */
		bool load_init(FILE* src);

		/** save kernel init_data
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		bool save_init(FILE* dest);

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		DREAL compute(INT idx_a, INT idx_b);

	private:
		/** distance */
		CDistance* distance;
		/** width */
		DREAL width;
};

#endif /* _DISTANCEKERNEL_H__ */

