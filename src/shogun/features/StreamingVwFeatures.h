/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Adaptation of Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#ifndef _STREAMING_VWFEATURES__H__
#define _STREAMING_VWFEATURES__H__

#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/mathematics/Math.h>

#include <shogun/io/InputParser.h>
#include <shogun/io/StreamingVwFile.h>
#include <shogun/io/StreamingVwCacheFile.h>
#include <shogun/features/StreamingDotFeatures.h>
#include <shogun/classifier/vw/vw_common.h>
#include <shogun/classifier/vw/vw_math.h>

namespace shogun
{
/** @brief This class implements streaming features for use with VW.
 *
 * Each example is stored in a VwExample object, which also
 * contains label and other information.
 * Features are hashed and are supposed to be used with a weight
 * vector of preallocated dimensions.
 */
class CStreamingVwFeatures : public CStreamingDotFeatures
{
public:

	/**
	 * Default constructor.
	 *
	 * Sets the reading functions to be
	 * CStreamingFile::get_*_vector and get_*_vector_and_label
	 * depending on the type T.
	 */
	CStreamingVwFeatures()
		: CStreamingDotFeatures()
	{
		init();
		set_read_functions();
	}

	/**
	 * Constructor taking args.
	 * Initializes the parser with the given args.
	 *
	 * @param file StreamingFile object, input file.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of example objects to be stored in the parser at a time.
	 */
	CStreamingVwFeatures(CStreamingVwFile* file,
			     bool is_labelled,
			     int32_t size)
		: CStreamingDotFeatures()
	{
		init(file, is_labelled, size);
		set_read_functions();
	}

	/**
	 * Constructor used when initialized
	 * with a cache file.
	 *
	 * @param file StreamingVwCacheFile object
	 * @param is_labelled Whether examples are labelled or not
	 * @param size Number of example objects to be stored in the parser at a time
	 */
	CStreamingVwFeatures(CStreamingVwCacheFile* file,
			     bool is_labelled,
			     int32_t size)
		: CStreamingDotFeatures()
	{
		init(file, is_labelled, size);
		set_read_functions();
	}

	/**
	 * Destructor.
	 *
	 * Ends the parsing thread. (Waits for pthread_join to complete)
	 */
	~CStreamingVwFeatures()
	{
		parser.end_parser();
		SG_UNREF(env);
	}

	/**
	 * Duplicate this object
	 *
	 * @return a copy of this object
	 */
	CFeatures* duplicate() const
	{
		return new CStreamingVwFeatures(*this);
	}

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
	 * Reset the file back to the first example.
	 * Only works for cache files.
	 */
	virtual void reset_stream()
	{
		if (working_file->is_seekable())
		{
			working_file->reset_stream();
			parser.exit_parser();
			parser.init(working_file, has_labels, parser.get_ring_size());
			parser.set_free_vector_after_release(false);
			parser.start_parser();
		}
		else
			SG_ERROR("The input cannot be reset! Please use 1 pass.\n");
	}

	/**
	 * Get the environment
	 * @return environment
	 */
	virtual CVwEnvironment* get_env()
	{
		SG_REF(env);
		return env;
	}

	/**
	 * Set the environment
	 *
	 * @param vw_env environment
	 */
	virtual void set_env(CVwEnvironment* vw_env)
	{
		env = vw_env;
		SG_REF(env);
	}

	/**
	 * Instructs the parser to return the next example.
	 *
	 * This example is stored as the current_example in this object.
	 *
	 * @return True on success, false if there are no more
	 * examples, or an error occurred.
	 */
	virtual bool get_next_example();

	/**
	 * Returns the current example.
	 *
	 * @return current example as VwExample*
	 */
	virtual VwExample* get_example();

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

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space() const;

	/**
	 * Reduce element 'w' to max(w-gravity, 0)
	 *
	 * @param w value to truncate
	 * @param gravity value to truncate using
	 *
	 * @return truncated value
	 */
	inline virtual float32_t real_weight(float32_t w, float32_t gravity)
	{
		float32_t wprime = 0;
		if (gravity < fabsf(w))
			wprime = CMath::sign(w)*(fabsf(w) - gravity);
		return wprime;
	}

	/**
	 * Dot product taken with another StreamingDotFeatures object.
	 *
	 * Currently only works if it is a CStreamingVwFeatures object.
	 * It takes the dot product of the current_vectors of both objects.
	 *
	 * @param df CStreamingDotFeatures object.
	 *
	 * @return Dot product.
	 */
	virtual float32_t dot(CStreamingDotFeatures *df);

	/**
	 * Dot product of an example with a vector
	 *
	 * @param ex example, as VwExample
	 * @param vec2 vector to take dot product with
	 *
	 * @return dot product
	 */
	virtual float32_t dense_dot(VwExample* &ex, const float32_t* vec2);

	/**
	 * Dot product of current feature vector with a dense vector
	 * which stores weights in hashed indices
	 *
	 * @param vec2 dense weight vector
	 * @param vec2_len length of weight vector (not used)
	 *
	 * @return dot product
	 */
	virtual float32_t dense_dot(const float32_t* vec2, int32_t vec2_len);

	/**
	 * Dot product between a dense weight vector and a sparse feature vector.
	 * Assumes the features to belong to the constant namespace.
	 *
	 * @param vec1 sparse feature vector
	 * @param vec2 weight vector
	 *
	 * @return dot product between dense weights and a sparse feature vector
	 */
	virtual float32_t dense_dot(SGSparseVector<float32_t>* vec1, const float32_t* vec2);

	/**
	 * Calculate dot product of features with another vector, truncating the elements
	 * of that vector by magnitude 'gravity' to a minimum final magnitude of zero.
	 *
	 * @param vec2 vector to take dot product with
	 * @param ex example whose features have to be taken
	 * @param gravity value to use for truncating
	 *
	 * @return dot product
	 */
	virtual float32_t dense_dot_truncated(const float32_t* vec2, VwExample* &ex, float32_t gravity);

	/**
	 * Add alpha*an example's feature vector to another dense vector.
	 * Takes the absolute value of current_vector if specified
	 *
	 * @param alpha alpha
	 * @param ex example whose vector should be used
	 * @param vec2 vector to add to
	 * @param vec2_len length of vector
	 * @param abs_val true if abs of example's vector should be taken
	 */
	virtual void add_to_dense_vec(float32_t alpha, VwExample* &ex, float32_t* vec2, int32_t vec2_len, bool abs_val = false);

	/**
	 * Add alpha*current_vector to another dense vector.
	 * Takes the absolute value of current_vector if specified
	 *
	 * @param alpha alpha
	 * @param vec2 vector to add to
	 * @param vec2_len length of vector
	 * @param abs_val true if abs of current_vector should be taken
	 */
	virtual void add_to_dense_vec(float32_t alpha, float32_t* vec2, int32_t vec2_len, bool abs_val = false);

	/** get number of non-zero features in vector
	 *
	 * @return number of non-zero features in vector
	 */
	virtual inline int32_t get_nnz_features_for_vector()
	{
		return current_length;
	}

	/**
	 * Return the number of features in the current example.
	 *
	 * @return number of features as int
	 */
	virtual int32_t get_num_features();

	/**
	 * Return the feature type, depending on T.
	 *
	 * @return Feature type as EFeatureType
	 */
	virtual inline EFeatureType get_feature_type();

	/**
	 * Return the feature class
	 *
	 * @return C_STREAMING_VW
	 */
	virtual EFeatureClass get_feature_class();

	/**
	 * Return the name.
	 *
	 * @return StreamingVwFeatures
	 */
	inline virtual const char* get_name() const { return "StreamingVwFeatures"; }

	/**
	 * Return the number of vectors stored in this object.
	 *
	 * @return 1 if current_example exists, else 0.
	 */
	inline virtual int32_t get_num_vectors() const
	{
		if (current_example)
			return 1;
		else
			return 0;
	}

	/**
	 * Return the size of one T object.
	 *
	 * @return Size of T.
	 */
	virtual int32_t get_size() { return sizeof(VwExample); }

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
	virtual void init(CStreamingVwFile *file, bool is_labelled, int32_t size);

	/**
	 * Init function when input is from a cache file
	 *
	 * @param file StreamingVwCacheFile to read from
	 * @param is_labelled whether labelled or not
	 * @param size number of examples in the parser's ring
	 */
	virtual void init(CStreamingVwCacheFile *file, bool is_labelled, int32_t size);

	/**
	 * Setup the example obtained from the parser so it
	 * can be directly updated by the learner.
	 *
	 * @param ae example object
	 */
	virtual void setup_example(VwExample* ae);

protected:

	/// The parser object, which reads from input and returns parsed example objects.
	CInputParser<VwExample> parser;

	/// Number of examples processed at a point of time
	size_t example_count;

	/// The current example's label.
	float64_t current_label;

	/// Number of features in current example.
	int32_t current_length;

	/// Environment for VW
	CVwEnvironment* env;

	/// Example currently being processed
	VwExample* current_example;
};
}
#endif // _STREAMING_VWFEATURES__H__
