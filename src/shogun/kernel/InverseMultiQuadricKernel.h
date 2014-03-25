/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Abhinav Maurya
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2011 Indian Institute of Technology Bombay
 */



#ifndef INVERSEMULTIQUADRIC_H_
#define INVERSEMULTIQUADRIC_H_

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CDistance;
/** @brief InverseMultiQuadricKernel
*
* \f[
*             K(x,x') = 1/(\sqrt{\| x - x' \|^2 +c^2})
* \f]
*/

class CInverseMultiQuadricKernel: public CKernel
{
public:
	/** default constructor */
	CInverseMultiQuadricKernel();

	/** constructor
	 * @param cache size of cache
	 * @param coef kernel parameter coef
	 * @param dist distance to be used
	 */
	CInverseMultiQuadricKernel(int32_t cache, float64_t coef, CDistance* dist);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param coef kernel parameter coef
	 * @param dist distance to be used
	 */
	CInverseMultiQuadricKernel(CFeatures *l, CFeatures *r, float64_t coef, CDistance* dist);

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);

	/**
	 * @return kernel type
	 */
	virtual EKernelType get_kernel_type() { return K_INVERSEMULTIQUADRIC; }

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
	virtual const char* get_name() const { return "InverseMultiQuadricKernel"; }

	/** getter for coef parameter
	 *  @return kernel parameter coefficient
	 */
	inline float64_t get_coef() { return this->coef; }

	/** setter for coef parameter
	 *  @param value kernel parameter coefficient
	 */
	inline void set_coef(float64_t value) { this->coef = value; }

	virtual ~CInverseMultiQuadricKernel();

protected:

	/** distance to be used
	 */
	CDistance* distance;

	/** theta parameter of kernel - coefficient
	 */
	float64_t coef;

	/**
	 * compute kernel for specific feature vectors
	 * corresponding to [idx_a] of left-side and [idx_b] of right-side
	 * @param idx_a left-side index
	 * @param idx_b right-side index
	 * @return kernel value
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
	virtual void load_serializable_post() throw (ShogunException);

private:

	void init();
};

}

#endif /* INVERSEMULTIQUADRIC_H_ */
