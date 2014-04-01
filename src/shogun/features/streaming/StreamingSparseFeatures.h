/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Modifications (W) 2013 Thoralf Klein
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef _STREAMING_SPARSEFEATURES__H__
#define _STREAMING_SPARSEFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/streaming/StreamingDotFeatures.h>
#include <shogun/io/streaming/InputParser.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/features/FeatureTypes.h>

namespace shogun
{
class CStreamingFile;

/** @brief This class implements streaming features with sparse feature vectors.
 * The vector is represented as an SGSparseVector<T>. Each entry is of type
 * SGSparseVectorEntry<T> with members `feat_index' and `entry'.
 *
 * This class expects the input from the StreamingFile object to be zero-based,
 * i.e., a feature entered as 1:6.5 would have feat_index=0 and entry=6.5.
 *
 * The current example is stored as a combination of current_vector
 * and current_label.
 * current_num_features stores the highest dimensionality of examples encountered
 * upto the point of the function call.
 * For example, if the first example is '1:6.5 7:10.0', then current_num_features
 * would be 7 after the first function call.
 *
 * Since the dimensionality of the feature space is not immediately known initially,
 * current_num_features may increase as more examples are processed and larger
 * dimensions are seen.
 * For this purpose, `expand_if_required()' is provided which when called with a
 * dynamically allocated float or double array and the length, reallocates that
 * array to the new dimensionality (if necessary), setting the newer dimensions
 * to zero, and updates the length parameter to equal the new length of the array.
 */
template <class T> class CStreamingSparseFeatures : public CStreamingDotFeatures
{
public:

	/**
	 * Default constructor.
	 *
	 * Sets the reading functions to be
	 * CStreamingFile::get_*_vector and get_*_vector_and_label
	 * depending on the type T.
	 */
	CStreamingSparseFeatures();

	/**
	 * Constructor taking args.
	 * Initializes the parser with the given args.
	 *
	 * @param file StreamingFile object, input file.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of example objects to be stored in the parser at a time.
	 */
	CStreamingSparseFeatures(CStreamingFile* file,
				 bool is_labelled,
				 int32_t size);

	/**
	 * Destructor.
	 *
	 * Ends the parsing thread. (Waits for pthread_join to complete)
	 */
	virtual ~CStreamingSparseFeatures();

	/**
	 * Sets the read function (in case the examples are
	 * unlabelled) to get_*_vector() from CStreamingFile.
	 *
	 * The exact function depends on type T.
	 *
	 * The parser uses the function set by this while reading
	 * unlabelled examples.
	 */
	virtual void set_vector_reader();

	/**
	 * Sets the read function (in case the examples are labelled)
	 * to get_*_vector_and_label from CStreamingFile.
	 *
	 * The exact function depends on type T.
	 *
	 * The parser uses the function set by this while reading
	 * labelled examples.
	 */
	virtual void set_vector_and_label_reader();

	/**
	 * Starts the parsing thread.
	 *
	 * To be called before trying to use any feature vectors from this object.
	 */
	virtual void start_parser();

	/**
	 * Ends the parsing thread.
	 *
	 * Waits for the thread to join.
	 */
	virtual void end_parser();

	/**
	 * Instructs the parser to return the next example.
	 *
	 * This example is stored as the current_example in this object.
	 *
	 * @return True on success, false if there are no more
	 * examples, or an error occurred.
	 */
	virtual bool get_next_example();

	/** get a single feature
	 *
	 * @param index index of feature in this vector
	 *
	 * @return sum of features that match dimension index and 0 if none is found
	 */
	T get_feature(int32_t index);

	/**
	 * Return the current feature vector as an SGSparseVector<T>.
	 *
	 * @return The vector as SGSparseVector<T>
	 */
	SGSparseVector<T> get_vector();

	/**
	 * Return the label of the current example as a float.
	 *
	 * Examples must be labelled, otherwise an error occurs.
	 *
	 * @return The label as a float64_t.
	 */
	virtual float64_t get_label();

	/**
	 * Release the current example, indicating to the parser that
	 * it has been processed by the learning algorithm.
	 *
	 * The parser is then free to throw away that example.
	 */
	virtual void release_example();

	/**
	 * Reset the file back to the first example
	 * if possible.
	 */
	virtual void reset_stream();

	/** set number of features
	 *
	 * Sometimes when loading sparse features not all possible dimensions
	 * are used. This may pose a problem to classifiers when being applied
	 * to higher dimensional test-data. This function allows to
	 * artificially explode the feature space
	 *
	 * @param num the number of features, must be larger
	 *        than the current number of features
	 * @return previous number of features
	 */
	int32_t set_num_features(int32_t num);

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space() const;

	/**
	 * Dot product taken with another StreamingDotFeatures object.
	 *
	 * Currently only works if it is a CStreamingSparseFeatures object.
	 * It takes the dot product of the current_vectors of both objects.
	 *
	 * @param df CStreamingDotFeatures object.
	 *
	 * @return Dot product.
	 */
	virtual float32_t dot(CStreamingDotFeatures *df);

	/** compute the dot product between two sparse feature vectors
	 * alpha * vec^T * vec
	 *
	 * @param alpha scalar to multiply with
	 * @param avec first sparse feature vector
	 * @param alen avec's length
	 * @param bvec second sparse feature vector
	 * @param blen bvec's length
	 * @return dot product between the two sparse feature vectors
	 */
	static T sparse_dot(T alpha, SGSparseVectorEntry<T>* avec, int32_t alen, SGSparseVectorEntry<T>* bvec, int32_t blen);

	/** compute the dot product between dense weights and a sparse feature vector
	 * alpha * sparse^T * w + b
	 *
	 * @param alpha scalar to multiply with
	 * @param vec dense vector to compute dot product with
	 * @param dim length of the dense vector
	 * @param b bias
	 * @return dot product between dense weights and a sparse feature vector
	 */
	T dense_dot(T alpha, T* vec, int32_t dim, T b);

	/**
	 * Dot product with another float64_t type dense vector.
	 *
	 * @param vec2 The dense vector with which to take the dot product.
	 * @param vec2_len length of vector
	 *
	 * @return Dot product as a float64_t.
	 */
	virtual float64_t dense_dot(const float64_t* vec2, int32_t vec2_len);

	/**
	 * Dot product with another dense vector.
	 *
	 * @param vec2 The dense vector with which to take the dot product.
	 * @param vec2_len length of vector
	 *
	 * @return Dot product as a float32_t.
	 */
	virtual float32_t dense_dot(const float32_t* vec2, int32_t vec2_len);

	/**
	 * Add alpha*current_vector to another float64_t type dense vector.
	 * Takes the absolute value of current_vector if specified.
	 *
	 * @param alpha alpha
	 * @param vec2 vector to add to, float64_t*
	 * @param vec2_len length of vector
	 * @param abs_val true if abs of current_vector should be taken
	 */
	virtual void add_to_dense_vec(float64_t alpha, float64_t* vec2, int32_t vec2_len, bool abs_val=false);

	/**
	 * Add alpha*current_vector to another dense vector.
	 * Takes the absolute value of current_vector if specified.
	 *
	 * @param alpha alpha
	 * @param vec2 vector to add to
	 * @param vec2_len length of vector
	 * @param abs_val true if abs of current_vector should be taken
	 */
	virtual void add_to_dense_vec(float32_t alpha, float32_t* vec2, int32_t vec2_len, bool abs_val=false);

	/**
	 * Get number of non-zero entries in current sparse vector
	 *
	 * @return number of features explicity set in the sparse vector
	 */
	int64_t get_num_nonzero_entries();

	/**
	 * Compute sum of squares of features on current vector.
	 *
	 * @return sum of squares for current vector
	 */
	float32_t compute_squared();

	/**
	 * Ensure features of the current vector are in ascending order.
	 * It modifies the current_sgvector in-place and does not change
	 * the reference in current_sgvector.features.
	 */
	void sort_features();

	/**
	 * Return the number of features in the current example.
	 *
	 * @return number of features as int
	 */
	virtual int32_t get_num_features();

	/**
	 * Return the number of non-zero features in vector
	 *
	 * @return number of sparse features in vector
	 */
	virtual int32_t get_nnz_features_for_vector();

	/**
	 * Return the feature type, depending on T.
	 *
	 * @return Feature type as EFeatureType
	 */
	virtual EFeatureType get_feature_type() const;

	/**
	 * Return the feature class
	 *
	 * @return C_STREAMING_SPARSE
	 */
	virtual EFeatureClass get_feature_class() const;

	/**
	 * Duplicate the object.
	 *
	 * @return a duplicate object as CFeatures*
	 */
	virtual CFeatures* duplicate() const;

	/**
	 * Return the name.
	 *
	 * @return StreamingSparseFeatures
	 */
	virtual const char* get_name() const { return "StreamingSparseFeatures"; }

	/**
	 * Return the number of vectors stored in this object.
	 *
	 * @return 1 if current_vector exists, else 0.
	 */
	virtual int32_t get_num_vectors() const;

private:
	/**
	 * Initializes members to null values.
	 * current_length is set to -1.
	 */
	virtual void init();

	/**
	 * Calls init, and also initializes the parser with the given args.
	 *
	 * @param file StreamingFile to read from
	 * @param is_labelled whether labelled or not
	 * @param size number of examples in the parser's ring
	 */
	virtual void init(CStreamingFile *file, bool is_labelled, int32_t size);

protected:
	/// The parser object, which reads from input and returns parsed example objects.
	CInputParser< SGSparseVectorEntry<T> > parser;

	/// The current example's feature vector as an SGVector<T>
	SGSparseVector<T> current_sgvector;

	/// The current vector index
	index_t current_vec_index;

	/// The current example's label.
	float64_t current_label;

	/// Number of features in current vector (as seen so far upto the current vector)
	int32_t current_num_features;
};

}
#endif // _STREAMING_SPARSEFEATURES__H__
