/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * (W) 2015 Wu Lin
 * (W) 2012 Jacob Walker
 *
 * Adapted from WeightedDegreeRBFKernel.h
 */

#ifndef LINEARARDKERNEL_H
#define LINEARARDKERNEL_H
#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>

namespace shogun
{

enum EARDKernelType
{
	KT_SCALAR=10,
	KT_DIAG=20,
	KT_FULL=30
};

/** @brief Linear Kernel with Automatic Relevance Detection computed
 * on CDotFeatures.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf y})= ||{\bf x}-{\bf y}||
 * \f]
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
 * To use this case,  please call set_matrix_weights(\f$\Lambda\f$),
 * where \f$\Lambda\f$ is a \f$d\f$-by-\f$p\f$ matrix
 *
 *
 * Indeed, the last case is more general than the first two cases.
 * When \f$\Lambda=\lambda I\f$ is, the last case becomes the first case.
 * When \f$\Lambda=\textbf{diag}(\lambda) \f$ is, the last case becomes the second case.
 *
 */
class CLinearARDKernel: public CDotKernel
{
public:
	/** default constructor */
	CLinearARDKernel();

	virtual ~CLinearARDKernel();

	/** return what type of kernel we are
	 *
	 * @return kernel type LINEARARD
	 */
	virtual EKernelType get_kernel_type() { return K_LINEARARD; }

	/** return the kernel's name
	 *
	 * @return name 
	 */
	virtual const char* get_name() const=0;

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

private:
	void initialize();

protected:
	/** ARD weights */
	SGMatrix<float64_t> m_weights;

	/** type of ARD kernel */
	EARDKernelType m_ARD_type;

	/** get features vector given idx
	 *
	 * @param idx index of CFeatures
	 * @param hs features
	 * @return the features vector
	 */
	virtual SGVector<float64_t> get_feature_vector(int32_t idx, CFeatures* hs);

#ifdef HAVE_LINALG_LIB
public:
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
	CLinearARDKernel(CDotFeatures* l, CDotFeatures* r,
			int32_t size=10);


	/** initialize kernel
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if initializing was successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);


	/** return current feature/dimension weights in matrix form
	 * Note that a diagonal matrix weights is considered as a column matrix (p-by-one matrix)
	 * where p is the number of features
	 * Note that a scalar weights is considered as a one-by-one matrix
	 *
	 * @return	weights in matrix form */
	virtual SGMatrix<float64_t> get_weights();

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
			index_t index=-1)=0;

	/** setter for feature/dimension weight (scalar kernel)
	 * @param weight scalar weight
	 */
	virtual void set_scalar_weights(float64_t weight);

	/** setter for feature/dimension weights (vector kernel)
	 * @param weights vector weight
	 */
	virtual void set_vector_weights(SGVector<float64_t> weights);

	/** setter for feature/dimension weights (matrix kernel)
	 * @param weights matrix weight
	 */
	virtual void set_matrix_weights(SGMatrix<float64_t> weights);

protected:

	/** setter for feature/dimension weights
	 * Note that a diagonal matrix weights is considered as a column matrix (p-by-one matrix)
	 * where p is the number of features
	 * Note that a scalar weights is considered as one-by-one matrix
	 *
	 * @param weights weights in vector form to set
	 */
	virtual void set_weights(SGMatrix<float64_t> weights);

	/** compute kernel function for features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return computed kernel function at indices a,b
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b)=0;

	/** helper function used to compute kernel function for features avec and bvec
	 *
	 * @param avec left feature vector
	 * @param bvec right feature vector
	 * @return computed kernel value
	 */
	virtual float64_t compute_helper(SGVector<float64_t> avec,
		SGVector<float64_t>bvec);

	/** helper function used to compute derivative with respect to weights
	 *
	 * @param avec left feature vector
	 * @param bvec right feature vector
	 * @param scale scaling value
	 * @param index the linearized index of a weight matrix (column-major)
	 *
	 * @return gradient with respect to parameter
	 */
	virtual float64_t compute_gradient_helper(SGVector<float64_t> avec, SGVector<float64_t> bvec,
		float64_t scale, index_t index);

	/** helper function used to compute kernel value
	 *
	 * The function is to compute \f$V=\Lambda ({\bf x}-{\bf y})\f$
	 *
	 * @param right_vec right feature vector
	 * @param scalar_weight set the scaling value, which is used in the scalar case (first case).
	 * @return the result of \f$V\f$
	 */
	virtual SGMatrix<float64_t> compute_right_product(SGVector<float64_t>right_vec, float64_t & scalar_weight);

	/** check whether index of gradient wrt weights is valid
	 *
	 * @param index the index of the element if parameter is a vector or matrix
	 * if the parameter is a matrix, index is the linearized index
	 * of the matrix (column-major)
	 *
	 */
	virtual void check_weight_gradient_index(index_t index);
#endif /* HAVE_LINALG_LIB */
};
}
#endif /* _LINEARARDKERNEL_H_ */
