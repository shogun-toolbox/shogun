/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _STREAMING_HASHED_SPARSEFEATURES__H__
#define _STREAMING_HASHED_SPARSEFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/features/SparseFeatures.h>
#include <shogun/features/streaming/StreamingDotFeatures.h>
#include <shogun/io/streaming/InputParser.h>

namespace shogun
{
class CStreamingDotFeatures;

/** @brief This class acts as an alternative to the CStreamingSparseFeatures class
 * and their difference is that the current example in this class is hashed into
 * a smaller dimension dim.
 *
 * The current example is stored as a combination of current_vector
 * and current_label. Call get_next_example() followed by get_current_vector()
 * to iterate through the stream.
 */
template <class ST> class CStreamingHashedSparseFeatures : public CStreamingDotFeatures
{
public:
	/** Constructor */
	CStreamingHashedSparseFeatures();

	/**
	 * Constructor with input information passed.
	 *
	 * @param file CStreamingFile to take input from.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of examples to be held in the parser's "ring".
	 * @param d the dimensionality of the target feature space
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 */
	CStreamingHashedSparseFeatures (CStreamingFile* file, bool is_labelled, int32_t size,
				int32_t d = 512, bool use_quadr = false, bool keep_lin_terms = true);

	/**
	 * Constructor taking a CDotFeatures object and optionally,
	 * labels, as args.
	 *
	 * The derived class should implement it so that the
	 * Streaming*Features class uses the DotFeatures object as the
	 * input, getting examples one by one from the DotFeatures
	 * object (and labels, if applicable).
	 *
	 * @param dot_features CDotFeatures object
	 * @param d the dimensionality of the target feature space
	 * @param use_quadr whether to use quadratic features or not
	 * @param keep_lin_terms whether to maintain the linear terms in the computations
	 * @param lab labels (optional)
	 */
	CStreamingHashedSparseFeatures (CSparseFeatures<ST>* dot_features, int32_t d = 512,
				bool use_quadr = false, bool keep_lin_terms = true, float64_t* lab = NULL);

	/** Destructor */
	virtual ~CStreamingHashedSparseFeatures ();

	/** compute dot product between vectors of two
	 * StreamingDotFeatures objects.
	 *
	 * @param df StreamingDotFeatures (of same kind) to compute
	 * dot product with
	 */
	virtual float32_t dot(CStreamingDotFeatures* df);

	/** compute dot product between current vector and a dense vector
	 *
	 * @param vec2 real valued vector
	 * @param vec2_len length of vector
	 */
	virtual float32_t dense_dot(const float32_t* vec2, int32_t vec2_len);

	/** add current vector multiplied with alpha to dense vector, 'vec'
	 *
	 * @param alpha scalar alpha
	 * @param vec2 real valued vector to add to
	 * @param vec2_len length of vector
	 * @param abs_val if true add the absolute value
	 */
	virtual void add_to_dense_vec(float32_t alpha, float32_t* vec2,
			int32_t vec2_len, bool abs_val = false);

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space() const;

	/**
	 * Return the name.
	 *
	 * @return the name of the class
	 */
	virtual const char* get_name() const;

	/**
	 * Return the number of vectors stored in this object.
	 *
	 * @return 1 if current_vector exists, else 0.
	 */
	virtual int32_t get_num_vectors() const;

	/**
	 * Duplicate the object.
	 *
	 * @return a duplicate object as CFeatures*
	 */
	virtual CFeatures* duplicate() const;

	/**
	 * Sets the read function (in case the examples are
	 * unlabelled) to get_*_vector() from CStreamingFile.
	 *
	 * The exact function depends on type ST.
	 *
	 * The parser uses the function set by this while reading
	 * unlabelled examples.
	 */
	virtual void set_vector_reader();

	/**
	 * Sets the read function (in case the examples are labelled)
	 * to get_*_vector_and_label from CStreamingFile.
	 *
	 * The exact function depends on type ST.
	 *
	 * The parser uses the function set by this while reading
	 * labelled examples.
	 */
	virtual void set_vector_and_label_reader();

	/**
	 * Return the feature type, depending on ST.
	 *
	 * @return Feature type as EFeatureType
	 */
	virtual EFeatureType get_feature_type() const;

	/**
	 * Return the feature class
	 *
	 * @return C_STREAMING_DENSE
	 */
	virtual EFeatureClass get_feature_class() const;

	/**
	 * Start the parser.
	 * It stores parsed examples from the input in a separate thread.
	 */
	virtual void start_parser();

	/**
	 * End the parser. Wait for the parsing thread to complete.
	 */
	virtual void end_parser();

	/**
	 * Return the label of the current example.
	 *
	 * Raise an error if the input has been specified as unlabelled.
	 *
	 * @return Label (if labelled example)
	 */
	virtual float64_t get_label();

	/**
	 * Indicate to the parser that it must fetch the next example.
	 *
	 * @return true on success, false on failure (i.e., no more examples).
	 */
	virtual bool get_next_example();

	/**
	 * Indicate that processing of the current example is done.
	 * The parser then considers it safe to dispose of that example
	 * and replace it with another one.
	 */
	virtual void release_example();

	/**
	 * Get the number of features in the current example.
	 *
	 * @return number of features in current example
	 */
	virtual int32_t get_num_features();

	/** Get the current example
	 *
	 * @return a SGSparseVector representing the hashed version of the string last read
	 */
	SGSparseVector<ST> get_vector();

private:
	void init(CStreamingFile* file, bool is_labelled, int32_t size,
		int32_t d, bool use_quadr, bool keep_lin_terms);

protected:

	/** dimensionality of new feature space */
	int32_t dim;

	/** Current example */
	SGSparseVector<ST> current_vector;

	/** The parser */
	CInputParser<SGSparseVectorEntry<ST> > parser;

	/** The current example's label */
	float64_t current_label;

	/** use quadratic feature or not */
	bool use_quadratic;

	/** keep linear terms or not */
	bool keep_linear_terms;
};
}

#endif // _STREAMING_HASHED_SPARSEFEATURES__H__
