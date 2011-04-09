/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef CAUCHYKERNEL_H_
#define CAUCHYKERNEL_H_

#include "lib/common.h"
#include "kernel/Kernel.h"
#include "distance/Distance.h"

namespace shogun
{

class CDistance;

/** @brief Wave kernel
 *
 * Formally described as
 *
 * \f[
 * 		K(x,x') = \frac{1}{1+\frac{\| x-x' \|^2}{\sigma}}
 * \f]
 *
 */

class CCauchyKernel: public CKernel
{
public:
	/** default constructor */
	CCauchyKernel();

	/** constructor
	 * @param cache size of cache
	 * @param sigma kernel parameter sigma
	 * @param dist distance to be used
	 */
	CCauchyKernel(int32_t cache, float64_t sigma, CDistance* dist);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param sigma kernel parameter sigma
	 * @param dist distance to be used
	 */
	CCauchyKernel(CFeatures *l, CFeatures *r, float64_t sigma, CDistance* dist);

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);

	/**
	 * @return kernel type
	 */
	inline virtual EKernelType get_kernel_type() { return K_CAUCHY; }

	/**
	 * @return type of features
	 */
	inline virtual EFeatureType get_feature_type() { return distance->get_feature_type(); }

	/**
	 * @return class of features
	 */
	inline virtual EFeatureClass get_feature_class() { return distance->get_feature_class(); }

	/**
	 * @return name of kernel
	 */
	inline virtual const char* get_name() const { return "Cauchy"; }

	virtual ~CCauchyKernel();
protected:

	/// distance to be used
	CDistance* distance;

	/// sigma parameter of kernel
	float64_t sigma;

	/**
	 * compute kernel for specific feature vectors
	 * corresponding to [idx_a] of left-side and [idx_b] of right-side
	 * @param idx_a left-side index
	 * @param idx_b right-side index
	 * @return kernel value
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);
};
}

#endif /* CAUCHYKERNEL_H_ */
