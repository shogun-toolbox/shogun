/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _RANDOMFOURIER_DOTFEATURES__H__
#define _RANDOMFOURIER_DOTFEATURES__H__


#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{
template <class ST> class CDenseFeatures;
class CDotFeatures;

/** names of kernels that can be approximated currently */
enum KernelName
{
	/** approximate gaussian kernel
	 * 	expects one parameter to be specified :
	 * 		kernel width
	 */
	GAUSSIAN,

	/** not specified */
	NOT_SPECIFIED	
};

/** @brief This class implements the random fourier features for the DotFeatures
 *  framework.
 *  Basically upon the object creation it computes the random coefficients, namely w and b,
 *  that are needed for this method and then every time a vector is required it is computed
 *  based on the following formula z(x) = sqrt(2/D) * cos(w'*x + b), where D is the number
 *  of samples that are used.
 *
 *  For more detailed information you can take a look at this source:
 *  i) Random Features for Large-Scale Kernel Machines - Ali Rahimi and Ben Recht
 */
class CRandomFourierDotFeatures : public CDotFeatures
{
public:

	/** default constructor */
	CRandomFourierDotFeatures();

	/** constructor that creates new random coefficients, basedon the kernel specified and the parameters
	 * of the kernel.
	 *
	 * @param feats	the dense features to use as a base
	 * @param num_samples the number of random fourier samples to draw / dimensionality of new feature space
	 * @param kernel the name of the kernel to approximate
	 * @param params kernel parameters (see kernel's description in KernelName to see what each kernel expects)
	 */
	CRandomFourierDotFeatures(CDotFeatures* feats, int32_t num_samples, KernelName kernel,
			SGVector<float64_t> params);

	/** constructor that uses the specified random coefficients.
	 *
	 * @param feats	the dense features to use as a base
	 * @param num_samples the number of random fourier samples to draw / dimensionality of new feature space
	 * @param ww specify the w (multiplicative) coefficients. See the class description for more details
	 * @param bb specify the b (additive) coefficients. See the class description for more details
	 */
	CRandomFourierDotFeatures(CDotFeatures* feats, int32_t num_samples, 
			SGMatrix<float64_t> ww, SGVector<float64_t> bb);

	/** constructor loading features from file
	 *
	 * @param loader File object via which to load data
	 */
	CRandomFourierDotFeatures(CFile* loader);
	
	/** copy constructor */
	CRandomFourierDotFeatures(const CRandomFourierDotFeatures& orig);

	/** duplicate */
	virtual CFeatures* duplicate() const;

	/** destructor */
	virtual ~CRandomFourierDotFeatures();

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
	 * 			iterate over
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

	/** @return object name */
	virtual const char* get_name() const;

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

	/** get the b vector
	 *
	 * @return the b vector
	 */
	SGVector<float64_t> get_b() const;

	/** get the w matrix
	 *
	 * @return the w matrix
	 */
	SGMatrix<float64_t> get_w() const;

	/** Generates a random b vector uniformly distributed in [0, 2*pi]
	 *
	 * @param num_samples the length of the vector
	 */
	static SGVector<float64_t> generate_random_b(int32_t num_samples);

	/** Generates num_samples vectors of length dim from the probability distribution
	 * computed from the fourier transform of the specified kernel.
	 *
	 * @param num_samples the number of vectors to create
	 * @param dim the dimension of the vectors
	 * @param kernel the kernel to approximate
	 * @param params kernel parameters (see kernel's description in KernelName to see what each kernel expects)
	 *
	 * @return a matrix containing the vectors concatenated
	 */
	static SGMatrix<float64_t> generate_random_w(int32_t num_samples, int32_t dim, KernelName kernel,
			SGVector<float64_t> params);

private:
	void init(CDotFeatures* feats, int32_t num_samples, SGMatrix<float64_t> ww,
			SGVector<float64_t> bb);

protected:

	/** dense features */
	CDotFeatures* dot_feats;

	/** b's */
	SGVector<float64_t> b;

	/** weights */
	SGMatrix<float64_t> w;

	/** num_samples */
	int32_t D;
};
}

#endif // _RANDOMFOURIER_DOTFEATURES__H__
