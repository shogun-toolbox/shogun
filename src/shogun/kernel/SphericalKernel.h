/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Based on GaussianKernel, Written (W) 1999-2010 Soeren Sonnenburg
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#ifndef _SPHERICALKERNEL_H__
#define _SPHERICALKERNEL_H__

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CDistance;

/** @brief Spherical kernel
 *
 * Formally described as
 *
 * \f[
 *     k(x, y) = 1 - \frac{3}{2} \frac{|x-y|}{\sigma}
 *     + \frac{1}{2} \left( \frac{|x-y|}{\sigma} \right)^3
 *     \mbox{if}~ |x-y| \leq \sigma \mbox{, zero otherwise}
 * \f]
 *
 */

class CSphericalKernel: public CKernel
{
	public:
	/** default constructor */
	CSphericalKernel();

	/** constructor
	 *
	 * @param size cache size
	 * @param sigma kernel parameter sigma
	 * @param dist distance
	 */
	CSphericalKernel(int32_t size, float64_t sigma, CDistance* dist);

	/** constructor
	 *
	 * @param l features of left-side
	 * @param r features of right-side
	 * @param sigma kernel parameter sigma
	 * @param dist distance
	 */
	CSphericalKernel(CFeatures *l, CFeatures *r, float64_t sigma, CDistance* dist);

	/** initialize kernel with features
	 *
	 * @param l features of left-side
	 * @param r features of right-side
	 * @return true if successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);

	void init();

	/**
	 * @return kernel type
	 */
	virtual EKernelType get_kernel_type() { return K_SPHERICAL; }

	/**
	 * @return type of features
	 */
	virtual EFeatureType get_feature_type() { return distance->get_feature_type(); }

	/**
	 * @return class of features
	 */
	virtual EFeatureClass get_feature_class() { return distance->get_feature_class(); }

	/**
	 * @return name of kernel
	 */
	virtual const char* get_name() const { return "SphericalKernel"; }

	/** set the kernel's sigma
	 *
	 * @param s kernel sigma
	 */
	virtual void set_sigma(float64_t s)
	{
		sigma=s;
	}

	/** return the kernel's sigma
	 *
	 * @return kernel sigma
	 */
	virtual float64_t get_sigma() const
	{
		return sigma;
	}

	virtual ~CSphericalKernel();
protected:

	/** distance */
	CDistance* distance;

	/** width */
	float64_t sigma;

	/**
	 * compute kernel function for features a and b
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

#endif /* _SPHERICALKERNEL_H__ */
