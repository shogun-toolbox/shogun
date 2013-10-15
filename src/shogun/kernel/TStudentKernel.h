/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Andrew Tereskin
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#ifndef TSTUDENTKERNEL_H_
#define TSTUDENTKERNEL_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CDistance;

/** @brief Generalized T-Student kernel
 *
 * Formally described as
 *
 * \f[
 *		K(x,x') = \frac{1}{1+\| x-x' \|^degree}
 * \f]
 * with degree=1 by default
 */

class CTStudentKernel: public CKernel
{
public:
	/** default constructor */
	CTStudentKernel();

	/** constructor
	 * @param cache size of cache
	 * @param d kernel parameter degree
	 * @param dist distance to be used
	 */
	CTStudentKernel(int32_t cache, float64_t d, CDistance* dist);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param d kernel parameter degree
	 * @param dist distance to be used
	 */
	CTStudentKernel(CFeatures *l, CFeatures *r, float64_t d, CDistance* dist);

	virtual ~CTStudentKernel();

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);

	/**
	 * @return kernel type
	 */
	virtual EKernelType get_kernel_type() { return K_TSTUDENT; }

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
	virtual const char* get_name() const { return "TStudentKernel"; }

	/** getter for degree parameter
	 *  @return kernel parameter degree
	 */
	inline float64_t get_degree() { return this->degree; }

	/** setter for degree parameter
	 *  @param value kernel parameter degree
	 */
	inline void set_degree(float64_t value) { this->degree = value; }

private:
	void init();

protected:

	/// distance to be used
	CDistance* distance;

	/// degree parameter of kernel
	float64_t degree;

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

#endif /* TSTUDENTKERNEL_H_ */
