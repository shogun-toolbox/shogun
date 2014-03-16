/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _RANDOMKITCHENSINKS_DOT_FEATURES_H__
#define _RANDOMKITCHENSINKS_DOT_FEATURES_H__

#include <shogun/features/DotFeatures.h>

namespace shogun
{

/** @brief class that implements the Random Kitchen Sinks for the DotFeatures
 * as mentioned in http://books.nips.cc/papers/files/nips21/NIPS2008_0885.pdf.
 *
 * The Random Kitchen Sinks algorithm expects:
 *              - Item 1
 *                a dataset to work on
 *              - Item 2
 *                a function phi such that |phi(x; a)| <= 1, the a's are the function parameters
 *              - Item 3
 *                a probability distrubution p, from which to draw the a's
 *              - Item 4
 *                the number of samples K to draw from p.
 *
 * Then:
 *              1. it draws K a's from p
 *              2. it computes for each vector in the dataset
 *                         Zi = [phi(Xi;a1), ..., phi(Xi;aK)]
 *              3. it solves the empirical risk minimization problem for all Zi, either
 *                 through least squares or through a linear SVM.
 *
 * This class implements the vector transformation on-the-fly whenever it is needed.
 * In order for it to work, the class expects the user to implement a subclass of
 * CRKSFunctions and implement in there the functions phi and p and then pass an
 * instantiated object of that class to the constructor.
 *
 * Further useful resources, include :
 *	http://www.shloosl.com/~ali/random-features/
 *	https://research.microsoft.com/apps/video/dl.aspx?id=103390&l=i
 */
class CRandomKitchenSinksDotFeatures : public CDotFeatures
{
public:

	/** default constructor */
	CRandomKitchenSinksDotFeatures();

	/** constructor
	 * Subclasses should call generate_random_coefficients() on their
	 * own if they choose to use this constructor.
	 *
	 * @param dataset the dataset to work on
	 * @param K the number of samples to draw
	 */
	CRandomKitchenSinksDotFeatures(CDotFeatures* dataset, int32_t K);

	/** constructor
	 *
	 * @param dataset the dataset to work on
	 * @param K the number of samples to draw
	 * @param coeff the random coefficients to use
	 */
	CRandomKitchenSinksDotFeatures(CDotFeatures* dataset, int32_t K,
			SGMatrix<float64_t> coeff);

	/** constructor loading features from file
	 *
	 * @param loader File object via which to load data
	 */
	CRandomKitchenSinksDotFeatures(CFile* loader);

	/** copy constructor */
	CRandomKitchenSinksDotFeatures(const CRandomKitchenSinksDotFeatures& orig);

	/** duplicate */
	virtual CFeatures* duplicate() const;

	/** destructor */
	virtual ~CRandomKitchenSinksDotFeatures();

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space() const;

	/** compute dot product between vector1 and vector2,
	 * appointed by their indices
	 *
	 * possible with subset
	 *
	 * @param vec_idx1 index of first vector
	 * @param df DotFeatures (of same kind) to compute dot product with
	 * @param vec_idx2 index of second vector
	 */
	virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df,
			int32_t vec_idx2);

	/** compute dot product between vector1 and a dense vector
	 *
	 * possible with subset
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 */
	virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2,
			int32_t vec2_len);

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
	virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
			float64_t* vec2, int32_t vec2_len, bool abs_val = false);

	/** get number of non-zero features in vector
	 *
	 * @param num which vector
	 * @return number of non-zero features in vector
	 */
	virtual int32_t get_nnz_features_for_vector(int32_t num);

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
	virtual void* get_feature_iterator(int32_t vector_index);

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
	virtual bool get_next_feature(int32_t& index, float64_t& value,
			void* iterator);

	/** clean up iterator
	 * call this function with the iterator returned by get_first_feature
	 *
	 * @param iterator as returned by get_first_feature
	 */
	virtual void free_feature_iterator(void* iterator);

	/** get feature type
	 *
	 * @return templated feature type
	 */
	virtual EFeatureType get_feature_type() const;

	/** get feature class
	 *
	 * @return feature class DENSE
	 */
	virtual EFeatureClass get_feature_class() const;

	/** get number of feature vectors
	 *
	 * @return number of feature vectors
	 */
	virtual int32_t get_num_vectors() const;

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
	const char* get_name() const;

protected:
	/** Method used before computing the dot product between
	 * a feature vector and a parameter vector
	 *
	 * @param vec_idx the feature vector index
	 * @param par_idx the parameter vector index
	 */
	virtual float64_t dot(index_t vec_idx, index_t par_idx);

	/** subclass must override this to perform any operations
	 * on the dot result between a feature vector and a parameter vector w
	 *
	 * @param dot_result the result of the dot operation
	 * @param par_idx the idx of the parameter vector
	 * @return the (optionally) modified result
	 */
	virtual float64_t post_dot(float64_t dot_result, index_t par_idx);

	/** Generates a random parameter vector, subclasses must override this
	 *
	 * @return a random parameter vector
	 */
	virtual SGVector<float64_t> generate_random_parameter_vector()=0;
private:
	void init(CDotFeatures* dataset, int32_t K);

protected:

	/** the dataset to work on */
	CDotFeatures* feats;

	/** the number of samples to use */
	int32_t num_samples;

	/** random coefficients of the function phi, drawn from p */
	SGMatrix<float64_t> random_coeff;
};
}

#endif // _RANDOMKITCHENSINKS_DOT_FEATURES_H__

