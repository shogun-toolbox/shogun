/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * (W) 2012 Jacob Walker
 *
 * Adapted from WeightedDegreeRBFKernel.h
 */

#ifndef LINEARARDKERNEL_H_
#define LINEARARDKERNEL_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

/** @brief Linear Kernel with Automatic Relevance Detection */
class CLinearARDKernel: public CDotKernel
{
public:
	/** default constructor */
	CLinearARDKernel();

	/** constructor
	 *
	 * @param size cache size
	 */
	CLinearARDKernel(int32_t size);

	/** constructor
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @param size cache size
	 */
	CLinearARDKernel(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r,
			int32_t size=10);

	virtual ~CLinearARDKernel();

	/** initialize kernel
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if initializing was successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);

	/** return what type of kernel we are
	 *
	 * @return kernel type LINEARARD
	 */
	virtual EKernelType get_kernel_type() { return K_LINEARARD; }

	/** return the kernel's name
	 *
	 * @return name LinearARDKernel
	 */
	virtual const char* get_name() const { return "LinearARDKernel"; }

	/** return feature class the kernel can deal with
	 *
	 * @return feature class DENSE
	 */
	virtual EFeatureClass get_feature_class() { return C_DENSE; }

	/** return feature type the kernel can deal with
	 *
	 * @return float64_t feature type
	 */
	virtual EFeatureType get_feature_type() { return F_DREAL; }

	/** set weight of particular feature/dimension
	 *
	 * @param w weight to set
	 * @param i index of feature
	 */
	virtual void set_weight(float64_t w, index_t i);

	/** get weight of particular feature/dimension
	 *
	 * @param i index of feature
	 *
	 * @return weight of feature
	 */
	virtual float64_t get_weight(index_t i);

	/** @return Current feature/dimension weights */
	virtual SGVector<float64_t> get_weights() { return m_weights; }

	/** setter for feature/dimension weights
	 *
	 * @param weights weights to set
	 */
	virtual void set_weights(SGVector<float64_t> weights) { m_weights=weights; }

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
	 * @return if initialization was successful
	 */
	void init_ft_weights();

	/** return derivative with respect to specified parameter
	 *
	 * @param  param the parameter
	 * @param index the index of the element if parameter is a vector
	 *
	 * @return gradient with respect to parameter
	 */
	virtual SGMatrix<float64_t> get_parameter_gradient(TParameter* param,
			index_t index=-1);

private:
	void init();

protected:
	/** weights */
	SGVector<float64_t> m_weights;
};
}
#endif /* LINEARARDKERNEL_H_ */
