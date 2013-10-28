/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Jacob Walker
 *
 * Adapted from WeightedDegreeRBFKernel.h
 */

#ifndef GAUSSIANARDKERNEL_H_
#define GAUSSIANARDKERNEL_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/LinearARDKernel.h>

namespace shogun
{

/** @brief Gaussian Kernel with Automatic Relevance Detection */
class CGaussianARDKernel: public CLinearARDKernel
{
public:
	/** default constructor */
	CGaussianARDKernel();

	/** constructor
	 *
	 * @param size cache size
	 * @param width kernel width
	 */
	CGaussianARDKernel(int32_t size, float64_t width);

	/** constructor
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @param size cache size
	 * @param width kernel width
	 */
	CGaussianARDKernel(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r,
		int32_t size=10, float64_t width=2.0);

	/** destructor */
	virtual ~CGaussianARDKernel();

	/** @param kernel is casted to CGaussianARDKernel, error if not possible
	 * is SG_REF'ed
	 * @return casted CGaussianARDKernel object
	 */
	static CGaussianARDKernel* obtain_from_generic(CKernel* kernel);

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
	virtual const char* get_name() const { return "GaussianARDKernel"; }

	/** return derivative with respect to specified parameter
	 *
	 * @param param the parameter
	 * @param index the index of the element if parameter is a vector
	 *
	 * @return gradient with respect to parameter
	 */
	virtual SGMatrix<float64_t> get_parameter_gradient(const TParameter* param,
			index_t index=-1);

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

private:
	void init();

protected:
	/** kernel width */
	float64_t m_width;
};
}
#endif /* GAUSSIANARDKERNEL_H_ */
