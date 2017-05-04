/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Written (W) 2011 Abhinav Maurya
 * Written (W) 2012 Heiko Strathmann
 * Written (W) 2016 Soumyajit De
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef GAUSSIANKERNEL_H
#define GAUSSIANKERNEL_H

#include <shogun/lib/config.h>
#include <shogun/kernel/ShiftInvariantKernel.h>

namespace shogun
{

class CFeatures;
class CDotFeatures;

/** @brief The well known Gaussian kernel (swiss army knife for SVMs) computed
 * on CDotFeatures.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau})
 * \f]
 *
 * where \f$\tau\f$ is the kernel width.
 *
 */
class CGaussianKernel: public CShiftInvariantKernel
{
public:
	/** default constructor */
	CGaussianKernel();

	/** constructor
	 *
	 * @param width width
	 */
	CGaussianKernel(float64_t width);

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
	CGaussianKernel(CDotFeatures* l, CDotFeatures* r, float64_t width, int32_t size=10);

	/** destructor */
	virtual ~CGaussianKernel();

	/** @param kernel is casted to CGaussianKernel, error if not possible
	 * is SG_REF'ed
	 * @return casted CGaussianKernel object
	 */
	static CGaussianKernel* obtain_from_generic(CKernel* kernel);

	/** Make a shallow copy of the kernel */
	virtual CSGObject* shallow_copy() const;

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
	virtual EKernelType get_kernel_type()
	{
		return K_GAUSSIAN;
	}

	/** @return feature type of distance used */
	virtual EFeatureType get_feature_type()
	{
		return F_ANY;
	}

	/** @return feature class of distance used */
	virtual EFeatureClass get_feature_class()
	{
		return C_ANY;
	}

	/** return the kernel's name
	 *
	 * @return name Gaussian
	 */
	virtual const char* get_name() const { return "GaussianKernel"; }

	/** set the kernel's width
	 *
	 * @param w kernel width
	 */
	virtual void set_width(float64_t w);

	/** return the kernel's width
	 *
	 * @return kernel width
	 */
	virtual float64_t get_width() const;

	/** return derivative with respect to specified parameter
	 *
	 * @param param the parameter
	 * @param index the index of the element if parameter is a vector
	 *
	 * @return gradient with respect to parameter
	 */
	virtual SGMatrix<float64_t> get_parameter_gradient(const TParameter* param, index_t index=-1);

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

	/** Can (optionally) be overridden to post-initialize some member
	 * variables which are not PARAMETER::ADD'ed. Make sure that at first
	 * the overridden method BASE_CLASS::LOAD_SERIALIZABLE_POST is called.
	 *
	 *  @exception ShogunException Will be thrown if an error occurres.
	 */
	virtual void load_serializable_post() throw (ShogunException);

	/** compute the distance between features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return computed the distance
	 *
	 * Note that in GaussianKernel,
	 * kernel(idx_a, idx_b)=exp(-distance(idx_a, idx_b))
	 * \f[
	 * 	distance({\bf x},{\bf y})= \frac{||{\bf x}-{\bf y}||^2}{\tau}
	 * \f]
	 */
	virtual float64_t distance(int32_t idx_a, int32_t idx_b) const;

private:
	/** register parameters and initialize with defaults */
	void register_params();
};

}
#endif /* _GAUSSIANKERNEL_H__ */
