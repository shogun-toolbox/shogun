/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * (W) 2012 Jacob Walker
 *
 * Adapted from WeightedDegreeRBFKernel.h
 *
 */

#ifndef GAUSSIANARDKERNEL_H_
#define GAUSSIANARDKERNEL_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

class CGaussianARDKernel: public CDotKernel
{

public:
	/** default constructor
	 *
	 */
	CGaussianARDKernel();

	/** constructor
	 *
	 * @param size cache size
	 */
	CGaussianARDKernel(int32_t size, float64_t width);

	/** constructor
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @param size cache size
	 */
	CGaussianARDKernel(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r,
		int32_t size=10, float64_t width = 2.0);

	virtual ~CGaussianARDKernel();

	/** initialize kernel
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if initializing was successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);

	/** return what type of kernel we are
	 *
	 * @return kernel type GAUSSIANARD
	 */
	virtual EKernelType get_kernel_type() { return K_GAUSSIANARD; }

	/** return the kernel's name
	 *
	 * @return name GaussianARDKernel
	 */
	inline virtual const char* get_name() const { return "GaussianARDKernel"; }


	/** return feature class the kernel can deal with
	 *
	 * @return feature class DENSE
	 */
	inline virtual EFeatureClass get_feature_class() { return C_DENSE; }

	/** return feature type the kernel can deal with
	 *
	 * @return float64_t feature type
	 */
	virtual EFeatureType get_feature_type() { return F_DREAL; }

	/*Set weight of particular feature
	 *
	 * @param w weight to set
	 * @param i index of feature
	 */
	virtual void set_weight(float64_t w, index_t i);

	/*Get weight of particular feature
	 *
	 * @param i index of feature
	 *
	 * @return weight of feature
	 */
	virtual float64_t get_weight(index_t i);

	/** set the kernel's width
	 *
	 * @param w kernel width
	 */
	inline virtual void set_width(float64_t w)
	{
		m_width = w;
	}

	/** return the kernel's width
	 *
	 * @return kernel width
	 */
	inline virtual float64_t get_width() const
	{
		return m_width;
	}

	/** return derivative with respect to specified parameter
	 *
	 * @param  param the parameter
	 * @param obj the object that owns the parameter
	 * @index index the index of the element if parameter is a vector
	 *
	 * @return gradient with respect to parameter
	 */
	virtual SGMatrix<float64_t> get_parameter_gradient(TParameter* param,
			CSGObject* obj, index_t index = -1);

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

	/** init feature weights
	 *
	 */
	void init_ft_weights();

private:

	void init();

protected:

	/** weights */
	SGVector<float64_t> m_weights;

	/* kernel width */
	float64_t m_width;
};

} /* namespace shogun */
#endif /* GAUSSIANARDKERNEL_H_ */
