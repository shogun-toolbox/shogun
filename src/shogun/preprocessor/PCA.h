/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann, 
 *          Evgeniy Andreev, Marc Zimmermann, Viktor Gal, Evan Shelhamer, 
 *          Bjoern Esser, Sanuj Sharma
 */

#ifndef PCA_H_
#define PCA_H_

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/lib/common.h>
#include <shogun/preprocessor/DensePreprocessor.h>

namespace shogun
{
/** Matrix decomposition method for PCA */
enum EPCAMethod
{
	/** if N>D then EVD is chosen automatically else SVD is chosen
	 * (D-dimensions N-number of vectors)
	 */
	AUTO = 10,
	/** SVD based PCA. Time complexity ~14dn^2 (d-dimensions n-number of vectors) */
	SVD = 20,
	/** Eigenvalue decomposition of covariance matrix.
	 * Time complexity ~10d^3 (d-dimensions n-number of vectors)
	 */
	EVD = 30
};

/** mode of pca */
enum EPCAMode
{
	/** cut by threshold */
	THRESHOLD,
	/** variance explained */
	VARIANCE_EXPLAINED,
	/** keep fixed number of features */
	FIXED_NUMBER
};

/** memory usage by PCA : In-place or through reallocation */
enum EPCAMemoryMode
{
	/** The feature matrix replaced by new matrix with target dims.
	 * This requires memory for old matrix as well as new matrix
	 * at once for a short amount of time initially.
	 */
	MEM_REALLOCATE,
	/** The feature matrix is modified in-place to generate the new matrix.
	 * Output matrix dimensions are changed to target dims, but actual matrix
	 * size remains same internally. Modifies initial data matrix
	 */
	MEM_IN_PLACE
};

/** @brief Preprocessor PCA performs principial component analysis on input
 * feature vectors/matrices. When the init method in PCA is called with proper
 * feature matrix X (with say N number of vectors and D feature dimension), a
 * transformation matrix is computed and stored internally. This transformation
 * matrix is then used to transform all D-dimensional feature vectors or feature
 * matrices (with D feature dimensions) supplied via apply_to_feature_matrix or
 * apply_to_feature_vector methods. This tranformation outputs the T-Dimensional
 * approximation of all these input vectors and matrices (where T<=min(D,N)). The
 * transformation matrix is essentially a DxT matrix, the columns of which correspond
 * to the eigenvectors of the covariance matrix(XX') having top T eigenvalues.
 *
 * This class provides 3 method options to compute the transformation matrix :
 * <em>EVD</em> : Eigen Value Decomposition of Covariance Matrix (\f$XX^T\f$)
 * The covariance matrix \f$XX^T\f$ is first formed internally and then
 * its eigenvectors and eigenvalues are computed.
 * The time complexity of this method is \f$~10D^3\f$ and should be used when N > D.
 *
 * <em>SVD</em> : Singular Value Decomposition of feature matrix X
 * The transpose of feature matrix, \f$X^T\f$, is decomposed using SVD.\f$X^T = UDV^T\f$
 * The matrix V in this decomposition contains the required eigenvectors and
 * the diagonal entries of the diagonal matrix D correspond to the non-negative
 * eigenvalues. Eigenvalue, \f$e_i\f$, is derived from a diagonal element, \f$d_i\f$,
 * using the formula \f$e_i = \frac{\sqrt{d_i}}{N-1}\f$.
 * The time complexity of this method is \f$~14DN^2\f$ and should be used when N < D.
 *
 * <em>AUTO</em> : This mode automagically chooses one of the above modes for the user
 * based on whether N > D (chooses EVD) or N < D (chooses SVD).
 *
 * This class provides 3 modes to determine the value of T :
 *
 * <em>FIXED_NUMBER</em> : T is supplied by user directly using set_target_dims method
 *
 * <em>VARIANCE_EXPLAINED</em> : The user supplies the fractional variance that he
 * wants preserved in the target dimension T. From this supplied fractional variance (thresh),
 * T is calculated as the smallest k such that the ratio of sum of largest k eigenvalues
 * over total sum of all eigenvalues is greater than thresh.
 *
 * <em>THRESH</em> : The user supplies a threshold. All eigenvectors with corresponding eigenvalue
 * greater than the supplied threshold are chosen.
 *
 * An option for whitening the transformation matrix is also given - do_whitening. Setting this
 * option normalizes the eigenvectors (ie. the columns of transformation matrix) by dividing them
 * with the square root of corresponding eigenvalues.
 *
 * Note that vectors/matrices don't have to have zero mean as it is substracted within the class.
 */
class CPCA : public CDensePreprocessor<float64_t>
{
	public:

		/** standard constructor
		 *
		 * @param do_whitening normalize columns(eigenvectors) in transformation matrix
		 * @param mode mode of pca : FIXED_NUMBER/VARIANCE_EXPLAINED/THRESHOLD
		 * @param thresh threshold value for VARIANCE_EXPLAINED or THRESHOLD mode
		 * @param method Matrix decomposition method used : SVD/EVD/AUTO[default]
		 * @param mem_mode memory usage mode of PCA : MEM_REALLOCATE/MEM_IN_PLACE
		 */
		CPCA(bool do_whitening=false, EPCAMode mode=FIXED_NUMBER, float64_t thresh=1e-6,
					EPCAMethod method=AUTO, EPCAMemoryMode mem_mode=MEM_REALLOCATE);

		/** special constructor for FIXED_NUMBER mode
		 *
		 * @param method Matrix decomposition method used : SVD/EVD/AUTO[default]
		 * @param do_whitening normalize columns(eigenvectors) in transformation matrix
		 * @param mem memory usage mode of PCA : MEM_REALLOCATE/MEM_IN_PLACE
		 */
		CPCA(EPCAMethod method, bool do_whitening=false, EPCAMemoryMode mem=MEM_REALLOCATE);

		/** destructor */
		virtual ~CPCA();

		virtual void fit(CFeatures* features);

		/** cleanup */
		virtual void cleanup();

		/** apply preprocessor to feature vector
		 * @param vector feature vector
		 * @return processed feature vector
		 */
		virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

		/** get transformation matrix, i.e. eigenvectors (potentially scaled if
		 * do_whitening is true)
		 */
		SGMatrix<float64_t> get_transformation_matrix();

		/** get eigenvalues of PCA
		 */
		SGVector<float64_t> get_eigenvalues();

		/** get mean vector of original data
		 */
		SGVector<float64_t> get_mean();

		/** @return object name */
		virtual const char* get_name() const { return "PCA"; }

		/** @return a type of preprocessor */
		virtual EPreprocessorType get_type() const { return P_PCA; }

		/** return the PCA memory mode being used */
		EPCAMemoryMode get_memory_mode() const;

		/** set PCA memory mode to be used
		 * @param e hoice between MEM_REALLOCATE and MEM_IN_PLACE
		 */
		void set_memory_mode(EPCAMemoryMode e);

		/** set zero tolerance of eigenvalues during data whitening
		 * @param eigenvalue_zero_tolerance zero tolerance value
		 */
		void set_eigenvalue_zero_tolerance(float64_t eigenvalue_zero_tolerance=1e-15);

		/** get zero tolerance of eigenvalues during data whitening
		 * @return zero tolerance value
		 */
		float64_t get_eigenvalue_zero_tolerance() const;

		/** setter for target dimension
		 * @param dim target dimension
		 */
		void set_target_dim(int32_t dim);

		/** getter for target dimension
		 * @return target dimension
		 */
		int32_t get_target_dim() const;

	protected:

		void init();

		virtual SGMatrix<float64_t>
		    apply_to_matrix(SGMatrix<float64_t>) override;

	protected:

		/** transformation matrix */
		SGMatrix<float64_t> m_transformation_matrix;
		/** num dim */
		int32_t num_dim;
		/** num old dim */
		int32_t num_old_dim;
		/** mean vector */
		SGVector<float64_t> m_mean_vector;
		/** eigenvalues vector */
		SGVector<float64_t> m_eigenvalues_vector;
		/** initialized */
		bool m_initialized;
		/** whitening */
		bool m_whitening;
		/** PCA mode */
		EPCAMode m_mode;
		/** thresh */
		float64_t m_thresh;
		/** PCA memory mode */
		EPCAMemoryMode m_mem_mode;
		/** PCA method */
		EPCAMethod m_method;
		/** eigenvalues within zero tolerance
		 * region are considered 0 while
		 * whitening to tackle numerical issues
		 */
		float64_t m_eigenvalue_zero_tolerance;

		/** target dimension */
		int32_t m_target_dim;

	private:
		/** Computes the transformation matrix using an eigenvalue decomposition. */
		void init_with_evd(const SGMatrix<float64_t>& feature_matrix, int32_t max_dim_allowed);
		/** Computes the transformation matrix using svd */
		void init_with_svd(const SGMatrix<float64_t>& feature_matrix, int32_t max_dim_allowed);
};
}
#endif // PCA_H_
