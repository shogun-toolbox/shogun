/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Wu Lin
 * Written (W) 2012 Jacob Walker
 *
 * Adapted from WeightedDegreeRBFKernel.h
 */

#ifndef GAUSSIANARDKERNEL_H
#define GAUSSIANARDKERNEL_H

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/kernel/LinearARDKernel.h>

namespace shogun
{

/** @brief Gaussian Kernel with Automatic Relevance Detection computed
 * on CDotFeatures.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf y})= exp(-\frac{||{\bf x}-{\bf y}||}{\tau})
 * \f]
 *
 * where \f$\tau\f$ is the kernel width.
 *
 * There are three variants based on \f$||\cdot||\f$.
 * The default case is
 * \f$\sum_{i=1}^{p}{{[\lambda \times ({\bf x_i}-{\bf y_i})] }^2}\f$
 * where \f$\lambda\f$ is a scalar and \f$p\f$ is # of features
 * To use this case,  please call set_scalar_weights(\f$\lambda\f$),
 * where \f$\lambda\f$ is a 1-by-1 matrix
 *
 * The second case is
 * \f$\sum_{i=1}^{p} {{[\lambda_i \times ({\bf x_i}-{\bf y_i})] }^2}\f$
 * where \f$\lambda\f$ is a vector (we use \f$\lambda\f$ as a column vector)
 * and \f$p\f$ is # of features
 * To use this case,  please call set_vector_weights(\f$\lambda\f$),
 * where \f$\lambda\f$ is a \f$p\f$-by-1 matrix
 *
 * The last case is
 * \f$({\bf x}-{\bf y})^T \Lambda^T \Lambda ({\bf x}-{\bf y})\f$
 * where \f$\Lambda\f$ is a \f$d\f$-by-\f$p\f$ matrix,
 * \f$p\f$ is # of features and \f$d\f$ can be \f$ d \ge p\f$ or \f$ d \le p\f$
 * To use this case,  please call set_matrix_weights(\f$\Lambda\f$)
 * where \f$\Lambda\f$ is a \f$d\f$-by-\f$p\f$ matrix
 *
 * Indeed, the last case is more general than the first two cases.
 * When \f$\Lambda=\lambda I\f$ is, the last case becomes the first case.
 * When \f$\Lambda=\textbf{diag}(\lambda) \f$ is, the last case becomes the second case.
 */
class CGaussianARDKernel: public CLinearARDKernel
{
public:
	/** default constructor */
	CGaussianARDKernel();

	/** set the kernel's width
	 *
	 * @param w kernel width
	 */
	virtual void set_width(float64_t w)	{ m_width=w; }

	/** destructor */
	virtual ~CGaussianARDKernel();

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
private:
	void initialize();

protected:
	/** kernel width */
	float64_t m_width;

#ifdef HAVE_LINALG_LIB
public:
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
	CGaussianARDKernel(CDotFeatures* l, CDotFeatures* r,
		int32_t size=10, float64_t width=2.0);

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

	/** return derivative with respect to specified parameter
	 *
	 * @param param the parameter
	 * @param index the index of the element if parameter is a vector or matrix
	 * if the parameter is a matrix, index is the linearized index
	 * of the matrix (column-major)
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

	/** compute the distance between features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return computed the distance
	 *
	 * Note that in GaussianARDKernel,
	 * kernel(idx_a, idx_b)=exp(-distance(idx_a, idx_b))
	 */
	virtual float64_t distance(int32_t idx_a, int32_t idx_b);

#endif /* HAVE_LINALG_LIB */
};
}
#endif /* _GAUSSIANARDKERNEL_H_ */
