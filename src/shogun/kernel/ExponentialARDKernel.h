/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wu Lin, Pan Deng
 */

#ifndef EXPONENTIALARDKERNEL_H
#define EXPONENTIALARDKERNEL_H
#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>

namespace shogun
{

/** ExponentialARD type */
enum EARDKernelType
{
	KT_SCALAR=10,
	KT_DIAG=20,
	KT_FULL=30
};

/** @brief Exponential Kernel with Automatic Relevance Detection computed
 * on DotFeatures.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf y})= \exp(-\Vert{\bf x}-{\bf y}\Vert)
 * \f]
 *
 * There are three variants based on \f$ \Vert \cdot \Vert \f$.
 * The default case is
 * \f$ \lambda \times \lambda \times \textbf{distance}(x,y)\f$
 * where \f$\lambda\f$ is a positive scalar and \f$p\f$ is # of features
 * To use this case,  please call set_scalar_weights(\f$\lambda\f$),
 * where \f$\lambda\f$ is a scalar.
 *
 * The second case is
 * \f$ \textbf{distance}( \lambda .* x, \lambda .* y)\f$
 * where \f$.*\f$ is element-wise product, 
 * \f$\lambda\f$ is a positive vector (we use \f$\lambda\f$ as a column vector)
 * and \f$p\f$ is # of features
 * To use this case,  please call set_vector_weights(\f$\lambda\f$),
 * where \f$\lambda\f$ is a \f$p\f$-by-1 vector.
 *
 * The last case is
 * \f$ \textbf{distance}( \Lambda^T \times x, \Lambda^T \times y)\f$
 * where \f$\Lambda^T\f$ is a \f$d\f$-by-\f$p\f$ upper triangular matrix with positive diagonal elements,
 * \f$p\f$ is # of features and \f$d\f$ can be \f$ d \le p\f$
 * To use this case,  please call set_matrix_weights(\f$\Lambda\f$),
 * where \f$\Lambda\f$ is a \f$p\f$-by-\f$d\f$ lower triangular matrix.
 * Note that only the lower triangular part of \f$\Lambda\f$ will be used
 *
 *
 * Indeed, the last case is more general than the first two cases.
 * When \f$\Lambda=\lambda \times I\f$ is, the last case becomes the first case.
 * When \f$\Lambda=\textbf{diag}(\lambda) \f$ is, the last case becomes the second case.
 *
 */
class ExponentialARDKernel: public DotKernel
{
public:
	/** default constructor */
	ExponentialARDKernel();

	~ExponentialARDKernel() override;

	/** return what type of kernel we are
	 *
	 * @return kernel type ExponentialARD
	 */
	EKernelType get_kernel_type() override { return K_EXPONENTIALARD; }

	/** return the kernel's name
	 *
	 * @return name 
	 */
	const char* get_name() const override { return "ExponentialARDKernel"; }

	/** return feature class the kernel can deal with
	 *
	 * @return feature class DENSE
	 */
	EFeatureClass get_feature_class() override { return C_DENSE; }

	/** return feature type the kernel can deal with
	 *
	 * @return float64_t feature type
	 */
	EFeatureType get_feature_type() override { return F_DREAL; }

private:
	void init();

protected:
	/** feature weights in standard domain in the matrix layout, which is only used in get_weights()*/
	SGMatrix<float64_t> m_weights_raw;

	/** feature weights in log domain in vector layout*/
	SGVector<float64_t> m_log_weights;

	/** the number of rows of feature weights for vector layout*/
	index_t m_weights_rows;

	/** the number of columns of feature weights for vector layout*/
	index_t m_weights_cols;

	/** type of ARD kernel */
	EARDKernelType m_ARD_type;

	/** get features vector given idx
	 *
	 * @param idx index of Features
	 * @param hs features
	 * @return the features vector
	 */
	virtual SGVector<float64_t> get_feature_vector(int32_t idx, std::shared_ptr<Features> hs);

	/** compute the distance between features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return computed the distance
	 *
	 */
	virtual float64_t distance(int32_t idx_a, int32_t idx_b)=0;

	/** compute kernel function for features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return computed kernel function at indices a,b
	 * kernel(idx_a, idx_b)=exp(-distance(idx_a, idx_b))
	 */
	float64_t compute(int32_t idx_a, int32_t idx_b) override
	{
		return std::exp(-distance(idx_a, idx_b));
	}

public:
	/** constructor
	 *
	 * @param size cache size
	 */
	ExponentialARDKernel(int32_t size);

	/** constructor
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @param size cache size
	 */
	ExponentialARDKernel(const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r,
			int32_t size=10);


	/** initialize kernel
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if initializing was successful
	 */
	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;


	/** return current feature/dimension weights in matrix form
	 * Note that a vector weights is considered as a row vector (one-by-p matrix)
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
	SGMatrix<float64_t> get_parameter_gradient(Parameters::const_reference param,
			index_t index=-1) override =0;

	/** setter for feature/dimension weight (scalar kernel)
	 * @param weight positive scalar weight
	 */
	virtual void set_scalar_weights(float64_t weight);

	/** setter for feature/dimension weights (vector kernel)
	 * @param weights positive vector weight
	 */
	virtual void set_vector_weights(SGVector<float64_t> weights);

	/** setter for feature/dimension weights (matrix kernel)
	 * @param weights a lower triangular matrix weight with positive diagonal elements
	 */
	virtual void set_matrix_weights(SGMatrix<float64_t> weights);

protected:
	/** a general setter for feature/dimension weights (matrix kernel)
	 * @param weights the weights can be scalar/vector/lower triangular matrix
	 */
	virtual void set_weights(SGMatrix<float64_t> weights);

	/** convert the weights in log domain to standard domain when get_weights() is called*/
	void lazy_update_weights();

	/** convert the m_log_weights in vector format to the matrix format in standard domain
	 *
	 * @param vec weights in log domain in vector layout
	 *
	 * @return weights in standard domain in matrix layout
	 * */
	SGMatrix<float64_t> get_weighted_vector(SGVector<float64_t> vec);

	/** helper function used to compute kernel value
	 *
	 * The function is to compute 
	 * for scalar weights: \f$V=\textbf{vec}\f$ and scalar_weight=\f$\lambda\f$
	 * for vector weights: \f$V=\lambda .* \textbf{vec}\f$ 
	 * for matrix weights: \f$V=\Lambda^T * \textbf{vec}\f$ 
	 *
	 * @param vec feature vector
	 * @param scalar_weight set the scaling value, which is used in the scalar case (first case).
	 * @return the result of \f$V\f$
	 */
	virtual SGMatrix<float64_t> compute_right_product(SGVector<float64_t>vec, float64_t & scalar_weight);

	/** check whether index of gradient wrt weights is valid
	 *
	 * @param index the index of the element if parameter is a vector or matrix
	 * if the parameter is a matrix, index is the linearized index
	 * of the matrix (column-major)
	 *
	 */
	virtual void check_weight_gradient_index(index_t index);
};
}
#endif /* EXPONENTIALARDKERNEL_H */
