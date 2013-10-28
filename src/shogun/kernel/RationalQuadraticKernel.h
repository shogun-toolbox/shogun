/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Rafał Surowiecki
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#ifndef RATIONAL_QUADRATIC_KERNEL_H_
#define RATIONAL_QUADRATIC_KERNEL_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CDistance;

/** @brief Rational Quadratic kernel
 *
 * Formally described as
 *
 * \f[
 *		K(x,x') = 1 -\frac{\| x-x'\|^2}{\|x-x'\|^2 + coef}
 * \f]
 *
 */

class CRationalQuadraticKernel: public CKernel
{
public:
	/** default constructor */
	CRationalQuadraticKernel();

	/** constructor
	 * @param cache size of cache
	 * @param coef kernel parameter coefficient
	 * @param dist distance to be used
	 */
	CRationalQuadraticKernel(int32_t cache, float64_t coef, CDistance* dist);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param c kernel parameter coefficient
	 * @param dist distance to be used
	 */
	CRationalQuadraticKernel(CFeatures *l, CFeatures *r, float64_t c, CDistance* dist);

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);

	/**
	 * @return kernel type
	 */
	virtual EKernelType get_kernel_type() { return K_RATIONAL_QUADRATIC; }

	/**
	 * @return type of features
	 */
	virtual EFeatureType get_feature_type() { return m_distance->get_feature_type(); }

	/**
	 * @return class of features
	 */
	virtual EFeatureClass get_feature_class() { return m_distance->get_feature_class(); }

	/**
	 * @return name of kernel
	 */
	virtual const char* get_name() const { return "RationalQuadraticKernel"; }

	/**
	 * @return coef - coefficient parameter of kernel
	 */
	virtual const float64_t get_coef() const { return m_coef; }

	/** setter for degree parameter
	 *  @param coef coefficient parameter of kernel
	 */
	inline void set_coef(float64_t coef) { m_coef=coef; }

	virtual ~CRationalQuadraticKernel();
protected:
	/**
	 * compute kernel for specific feature vectors
	 * corresponding to [idx_a] of left-side and [idx_b] of right-side
	 * @param idx_a left-side index
	 * @param idx_b right-side index
	 * @return kernel value
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);

private:
	/**Initialize parameters for serialization*/
	void init();

protected:
	/// distance to be used
	CDistance* m_distance;

	/// coefficient parameter of kernel
	float64_t m_coef;
};
}

#endif /* RATIONAL_QUADRATIC_KERNEL_H_ */
