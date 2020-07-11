/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Bjoern Esser, Evangelos Anagnostopoulos
 */

#ifndef _RANDOMKITCHENSINKS_DOT_FEATURES_H__
#define _RANDOMKITCHENSINKS_DOT_FEATURES_H__

#include <shogun/lib/config.h>

#include <shogun/features/DotFeatures.h>
#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{

/** @brief class that implements the Random Kitchen Sinks (RKS) for the DotFeatures
 * as mentioned in http://books.nips.cc/papers/files/nips21/NIPS2008_0885.pdf.
 *
 * RKS input:
 *		- a dataset \f$\{x_i, y_i\}_{i=1,\dots,m}\f$ of \f$m\f$ points to work on
 *		- \f$\phi(x; w)\f$: a bounded feature function s.t. \f$|\phi(x; w)| \leq 1\f$, where \f$w\f$ is the function parameter
 *		- \f$p(w)\f$: a probability distrubution function, from which to draw the \f$w\f$
 *		- \f$K\f$: the number of samples to draw from \f$p(w)\f$
 * 		- \f$C\f$: a scalar, which is chosen to be large enough in practice.
 *
 * RKS output:
 * 		A function \f$\hat{f}(x) = \sum_{k=1}^{K} \phi(x; w_k)\alpha_k\f$
 *		1. Draw \f$w_1,\dots,w_K\f$ iid from \f$p(w)\f$
 *		2. Featurize the input: \f$z_i = [\phi(x_i; w_1),\dots,\phi(x_i; w_K)]^{\top}\f$
 *		3. With \f$w\f$ fixed, solve the empirical risk minimization problem:
 * 		\f[ \underset{\alpha \in \mathbf{R}^K}{\text{minimize}} \quad \frac{1}{m}\sum_{i=1}^{m} c(\alpha^{\top} z_i, y_i) \f]
 * 		\f[ \text{s.t.} \quad \|\alpha\|_{\infty} \leq C/K. \f]
 *		  for vector \f$\alpha\f$, either through least squares when \f$c(y', y)\f$ is the quadratic loss or through a linear SVM when \f$c(y', y)\f$ is the hinge loss.
 *
 * This class implements the vector transformation on-the-fly whenever it is needed.
 * In order for it to work, the class expects the user to implement a subclass of
 * CRKSFunctions and implement in there the functions \f$\phi\f$ and \f$p\f$ and then pass an
 * instantiated object of that class to the constructor. For example, in the derived class CRandomFourierDotFeatures,
 * random fourier features are implemented as \f$ z(x) = \sqrt{2/K}\cos(w^{\top}x + b)\f$, where \f$w\f$ drawn from a Gaussian distribution and \f$b\f$ from a uniform distribution.
 *
 * Further useful resources, include :
 *	http://www.shloosl.com/~ali/random-features/
 *	https://research.microsoft.com/apps/video/dl.aspx?id=103390&l=i
 */
class RandomKitchenSinksDotFeatures : public RandomMixin<DotFeatures>
{
public:

	/** default constructor */
	RandomKitchenSinksDotFeatures();

	/** constructor
	 * Subclasses should call generate_random_coefficients() on their
	 * own if they choose to use this constructor.
	 *
	 * @param dataset the dataset to work on
	 * @param K the number of samples to draw
	 */
	RandomKitchenSinksDotFeatures(std::shared_ptr<DotFeatures> dataset, int32_t K);

	/** constructor
	 *
	 * @param dataset the dataset to work on
	 * @param K the number of samples to draw
	 * @param coeff the random coefficients to use
	 */
	RandomKitchenSinksDotFeatures(std::shared_ptr<DotFeatures> dataset, int32_t K,
			SGMatrix<float64_t> coeff);

	/** constructor loading features from file
	 *
	 * @param loader File object via which to load data
	 */
	RandomKitchenSinksDotFeatures(const std::shared_ptr<File>& loader);

	/** copy constructor */
	RandomKitchenSinksDotFeatures(const RandomKitchenSinksDotFeatures& orig);

	/** duplicate */
	std::shared_ptr<Features> duplicate() const override;

	/** destructor */
	~RandomKitchenSinksDotFeatures() override;

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	int32_t get_dim_feature_space() const override;

	/** compute dot product between vector1 and vector2,
	 * appointed by their indices
	 *
	 * possible with subset
	 *
	 * @param vec_idx1 index of first vector
	 * @param df DotFeatures (of same kind) to compute dot product with
	 * @param vec_idx2 index of second vector
	 */
	float64_t dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df,
			int32_t vec_idx2) const override;

	/** compute dot product between vector1 and a dense vector
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 dense vector
	 */
	float64_t
	dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const override;

	/** add vector 1 multiplied with alpha to dense vector2
	 *
	 * possible with subset
	 *
	 * @param alpha scalar alpha
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 * @param abs_val if true add the absolute value
	 */
	void add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
			float64_t* vec2, int32_t vec2_len, bool abs_val = false) const override;

	/** get number of non-zero features in vector
	 *
	 * @param num which vector
	 * @return number of non-zero features in vector
	 */
	int32_t get_nnz_features_for_vector(int32_t num) const override;

	/** iterate over the non-zero features
	 *
	 * call get_feature_iterator first, followed by get_next_feature and
	 * free_feature_iterator to cleanup
	 *
	 * possible with subset
	 *
	 * @param vector_index the index of the vector over whose components to
	 *			iterate over
	 * @return feature iterator (to be passed to get_next_feature)
	 */
	void* get_feature_iterator(int32_t vector_index) override;

	/** iterate over the non-zero features
	 *
	 * call this function with the iterator returned by get_first_feature
	 * and call free_feature_iterator to cleanup
	 *
	 * possible with subset
	 *
	 * @param index is returned by reference (-1 when not available)
	 * @param value is returned by reference
	 * @param iterator as returned by get_first_feature
	 * @return true if a new non-zero feature got returned
	 */
	bool get_next_feature(int32_t& index, float64_t& value,
			void* iterator) override;

	/** clean up iterator
	 * call this function with the iterator returned by get_first_feature
	 *
	 * @param iterator as returned by get_first_feature
	 */
	void free_feature_iterator(void* iterator) override;

	/** get feature type
	 *
	 * @return templated feature type
	 */
	EFeatureType get_feature_type() const override;

	/** get feature class
	 *
	 * @return feature class DENSE
	 */
	EFeatureClass get_feature_class() const override;

	/** get number of feature vectors
	 *
	 * @return number of feature vectors
	 */
	int32_t get_num_vectors() const override;

	/** generate the random coefficients and return them in a
	 * matrix where each column is a parameter vector
	 *
	 * @return the parameter vectors in a matrix
	 */
	SGMatrix<float64_t> generate_random_coefficients();

	/** returns the random function parameters that were generated through the function p
	 *
	 * @return the generated random coefficients
	 */
	SGMatrix<float64_t> get_random_coefficients();

	/** @return object name */
	const char* get_name() const override;

protected:
	/** Method used before computing the dot product between
	 * a feature vector and a parameter vector
	 *
	 * @param vec_idx the feature vector index
	 * @param par_idx the parameter vector index
	 */
	virtual float64_t dot(index_t vec_idx, index_t par_idx) const;

	/** subclass must override this to perform any operations
	 * on the dot result between a feature vector and a parameter vector w
	 *
	 * @param dot_result the result of the dot operation
	 * @param par_idx the idx of the parameter vector
	 * @return the (optionally) modified result
	 */
	virtual float64_t post_dot(float64_t dot_result, index_t par_idx) const;

	/** Generates a random parameter vector, subclasses must override this
	 *
	 * @return a random parameter vector
	 */
	virtual void generate_random_params(SGVector<float64_t>& vec) = 0;
	
	/** @return number of parameters per input dimension */
	virtual int32_t get_num_params() const = 0;
private:
	void init(std::shared_ptr<DotFeatures> dataset, int32_t K);

protected:

	/** the dataset to work on */
	std::shared_ptr<DotFeatures> feats;

	/** the number of samples to use */
	int32_t num_samples;

	/** random coefficients of the function phi, drawn from p */
	SGMatrix<float64_t> random_coeff;
};
}

#endif // _RANDOMKITCHENSINKS_DOT_FEATURES_H__

