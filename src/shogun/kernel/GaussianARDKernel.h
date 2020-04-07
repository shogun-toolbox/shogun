/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wu Lin, Jacob Walker, Heiko Strathmann, Roman Votyakov, Yuyu Zhang, 
 *          Sergey Lisitsyn, Soumyajit De, Pan Deng
 */

#ifndef GAUSSIANARDKERNEL_H
#define GAUSSIANARDKERNEL_H

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/kernel/ExponentialARDKernel.h>

namespace shogun
{
/** @brief Gaussian Kernel with Automatic Relevance Detection computed
 * on DotFeatures.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf y})= \exp(-\frac{\Vert {\bf x}-{\bf y} \Vert}{2})
 * \f]
 *
 * There are three variants based on \f$\Vert \cdot \Vert \f$.
 * The default case is
 * \f$\sum_{i=1}^{p}{{[\lambda \times ({\bf x_i}-{\bf y_i})] }^2}\f$
 * where \f$\lambda\f$ is a positive scalar and \f$p\f$ is # of features.
 * To use this case,  please call set_scalar_weights(\f$\lambda\f$),
 * where \f$\lambda\f$ is a positive scalar.
 *
 * The second case is
 * \f$\sum_{i=1}^{p} {{[\lambda_i \times ({\bf x_i}-{\bf y_i})] }^2}\f$
 * where \f$\lambda\f$ is a positive vector (we use \f$\lambda\f$ as a column vector)
 * and \f$p\f$ is # of features.
 * To use this case,  please call set_vector_weights(\f$\lambda\f$),
 * where \f$\lambda\f$ is a positive vector.
 *
 * The last case is
 * \f$({\bf x}-{\bf y})^T \Lambda \Lambda^T ({\bf x}-{\bf y})\f$
 * where \f$\Lambda^T\f$ is a \f$d\f$-by-\f$p\f$ upper triangular matrix with positive diagonal elements,
 * \f$p\f$ is # of features and \f$ d \le p\f$.
 * To use this case, please call set_matrix_weights(\f$\Lambda\f$),
 * where \f$\Lambda\f$ is a \f$p\f$-by-\f$d\f$ lower triangular matrix with positive diagonal elements.
 * Note that only the lower triangular part of \f$\Lambda\f$ will be used
 *
 * Indeed, the last case is more general than the first two cases.
 * When \f$\Lambda=\lambda I\f$ is, the last case becomes the first case.
 * When \f$\Lambda=\textbf{diag}(\lambda) \f$ is, the last case becomes the second case.
 */
class GaussianARDKernel: public ExponentialARDKernel
{
public:
	/** default constructor */
	GaussianARDKernel();

	/** destructor */
	~GaussianARDKernel() override;

	/** return what type of kernel we are
	 *
	 * @return kernel type GAUSSIANARD
	 */
	EKernelType get_kernel_type() override { return K_GAUSSIANARD; }

	/** return the kernel's name
	 *
	 * @return name GaussianARDKernel
	 */
	const char* get_name() const override { return "GaussianARDKernel"; }
private:
	void init();

protected:
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
	float64_t distance(int32_t idx_a, int32_t idx_b) override;

public:
	/** constructor
	 *
	 * @param size cache size
	 * @param width kernel width
	 */
	GaussianARDKernel(int32_t size);

	/** constructor
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @param size cache size
	 * @param width kernel width
	 */
	GaussianARDKernel(const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r,
		int32_t size=10);

	/** @param kernel is casted to GaussianARDKernel, error if not possible
	 * is SG_REF'ed
	 * @return casted GaussianARDKernel object
	 */
	static std::shared_ptr<GaussianARDKernel> obtain_from_generic(const std::shared_ptr<Kernel>& kernel);

	/** initialize kernel
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if initializing was successful
	 */
	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

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
			index_t index=-1) override;

	/** return diagonal part of derivative with respect to specified parameter
	 *
	 * @param param the parameter
	 * @param index the index of the element if parameter is a vector
	 *
	 * @return diagonal part of gradient with respect to parameter
	 */
	SGVector<float64_t> get_parameter_gradient_diagonal(
		Parameters::const_reference param, index_t index=-1) override;

protected:
	/** helper function to compute quadratic terms in
	 * (a-b)^2 (== a^2+b^2-2ab)
	 */
	virtual void precompute_squared();

	/** helper function to compute quadratic terms in
	 * (a-b)^2 (== a^2+b^2-2ab)
	 *
	 * @param buf buffer to store squared terms (will be allocated)
	 * @param df dot feature object based on which k(i,i) is computed
	 * */
	virtual SGVector<float64_t> precompute_squared_helper(std::shared_ptr<DotFeatures> df);

	/** squared left-hand side */
	SGVector<float64_t> m_sq_lhs;
	/** squared right-hand side */
	SGVector<float64_t> m_sq_rhs;

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


	/** helper function to compute derivative with respect to specified parameter
	 *
	 * @param param the parameter
	 * @param index the index of the element if parameter is a vector or matrix
	 * if the parameter is a matrix, index is the linearized index
	 * of the matrix (column-major)
	 * @param idx_a the row index of the gradient matrix
	 * @param idx_b the column index of the gradient matrix
	 * @param avec feature vector corresponding to idx_a
	 * @param bvec feature vector corresponding to idx_b
	 *
	 * @return gradient at row idx_a and column idx_b with respect to parameter
	 */
	virtual float64_t get_parameter_gradient_helper(Parameters::const_reference param,
		index_t index, int32_t idx_a, int32_t idx_b,
		SGVector<float64_t> avec, SGVector<float64_t> bvec);
};
}
#endif /* _GAUSSIANARDKERNEL_H_ */
