/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */
#ifndef _STREAMING_HASHEDDOCDOTFEATURES__H__
#define _STREAMING_HASHEDDOCDOTFEATURES__H__

#include <shogun/features/StringFeatures.h>
#include <shogun/features/streaming/StreamingDotFeatures.h>
#include <shogun/lib/Tokenizer.h>
#include <shogun/converter/HashedDocConverter.h>
#include <shogun/io/streaming/InputParser.h>
#include <shogun/io/streaming/StreamingFileFromStringFeatures.h>

namespace shogun
{
class CStreamingDotFeatures;
class CTokenizer;
class CHashedDocConverter;
 
/** @brief This class implements streaming features for a document collection.
 * Like in the standard Bag-of-Words representation, this class considers each document as a collection of tokens,
 * which are then hashed into a new feature space of a specified dimension.
 * This class is very flexible and allows the user to specify the tokenizer used to tokenize each document,
 * specify whether the results should be normalized with regards to the sqrt of the document size, as well
 * as to specify whether he wants to combine different tokens.
 * The latter implements a k-skip n-grams approach, meaning that you can combine up to n tokens, while skipping up to k.
 * Eg. for the tokens ["a", "b", "c", "d"], with n_grams = 2 and skips = 2, one would get the following combinations :
 * ["a", "ab", "ac" (skipped 1), "ad" (skipped 2), "b", "bc", "bd" (skipped 1), "c", "cd", "d"].
 *
 * The current example is stored as a combination of current_vector
 * and current_label. Call get_next_example() followed by get_current_vector()
 * to iterate through the stream.
 */
class CStreamingHashedDocDotFeatures : public CStreamingDotFeatures
{
public:
	/** Constructor */
	CStreamingHashedDocDotFeatures();

	/**
	 * Constructor with input information passed.
	 * Will use normalization and no quadratic features by default, user should
	 * use the set_normalization() and set_k_skip_n_gram() methods to change that.
	 *
	 * @param file CStreamingFile to take input from.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of examples to be held in the parser's "ring".
	 * @param tzer the tokenizer to use on the document collection
	 * @param bits the number of bits of the new dimension (means a dimension of size 2^bits)
	 */
	CStreamingHashedDocDotFeatures(CStreamingFile* file, bool is_labelled, int32_t size,
			CTokenizer* tzer, int32_t bits=20);

	/**
	 * Constructor taking a CDotFeatures object and optionally,
	 * labels, as args.
	 * Will use normalization and no quadratic features by default, user should
	 * use the set_normalization() and set_k_skip_n_gram() methods to change that.
	 *
	 * The derived class should implement it so that the
	 * Streaming*Features class uses the DotFeatures object as the
	 * input, getting examples one by one from the DotFeatures
	 * object (and labels, if applicable).
	 *
	 * @param dot_features CDotFeatures object
	 * @param tzer the tokenizer to use on the document collection
	 * @param bits the number of bits of the new dimension (means a dimension of size 2^bits)
	 * @param lab labels (optional)
	 */
	CStreamingHashedDocDotFeatures(CStringFeatures<char>* dot_features,CTokenizer* tzer,
			int32_t bits=20, float64_t* lab=NULL);

	/** Destructor */
	virtual ~CStreamingHashedDocDotFeatures();

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
			int32_t vec2_len, bool abs_val=false);

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
	 * Return the feature type, depending on T.
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
	SGSparseVector<float64_t> get_vector();

	/** specify whether hashed vector should be normalized or not
	 *
	 * @param normalize  whether to normalize
	 */
	void set_normalization(bool normalize);

	/** Method used to specify the parameters for the quadratic
	 * approach of k-skip n-grams. See class description for more 
	 * details and an example.
	 *
	 * @param k the max number of allowed skips
	 * @param n the max number of tokens to combine
	 */
	void set_k_skip_n_grams(int32_t k, int32_t n);

private:
	void init(CStreamingFile* file, bool is_labelled, int32_t size, CTokenizer* tzer,
		int32_t bits, bool normalize, int32_t n_grams, int32_t skips);

protected:
	
	/** number of bits for the target dimension */
	int32_t num_bits;

	/** Current example */
	SGSparseVector<float64_t> current_vector;

	/** CTokenizer */
	CTokenizer *tokenizer;

	/** Converter */
	CHashedDocConverter* converter;

	/** The parser */
	CInputParser<char> parser;

	/** The current example's label */
	float64_t current_label;
};
}

#endif // _STREAMING_HASHEDDOCDOTFEATURES__H__
