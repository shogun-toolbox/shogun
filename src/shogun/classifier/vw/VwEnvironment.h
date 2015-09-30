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
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#ifndef _VW_ENV_H__
#define _VW_ENV_H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/base/DynArray.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/common.h>
#include <shogun/lib/v_array.h>
#include <shogun/classifier/vw/vw_constants.h>

namespace shogun
{

/** @brief Class CVwEnvironment is the environment used by VW.
 *
 * Contains global constants and settings which change the behaviour
 * of Vowpal Wabbit.
 *
 * It is used while parsing input, and also while learning.
 *
 * One CVwEnvironment object should be bound to the CStreamingVwFile or
 * CStreamingVwCacheFile, and the pointer to it propagated upwards
 * to CStreamingVwFeatures and finally to CVowpalWabbit.
 */
class CVwEnvironment: public CSGObject
{
public:
	/**
	 * Default constructor
	 * Should initialize with reasonable default values
	 */
	CVwEnvironment();

	/**
	 * Destructor
	 */
	virtual ~CVwEnvironment() { }

	/**
	 * Set number of bits used for the weight vector
	 * @param bits number of bits
	 */
	inline void set_num_bits(vw_size_t bits) { num_bits = bits; }

	/**
	 * Return number of bits used for weight vector
	 * @return number of bits
	 */
	inline vw_size_t get_num_bits() { return num_bits; }

	/**
	 * Set mask used while accessing features
	 * @param m mask
	 */
	inline void set_mask(vw_size_t m) { mask = m; }

	/**
	 * Return the mask used
	 * @return mask
	 */
	inline vw_size_t get_mask() { return mask; }

	/**
	 * Return minimum label encountered
	 * @return min label
	 */
	inline float64_t get_min_label() { return min_label; }

	/**
	 * Return maximum label encountered
	 * @return max label
	 */
	inline float64_t get_max_label() { return max_label; }

	/**
	 * Return number of threads used for learning
	 * @return number of threads
	 */
	inline vw_size_t num_threads() { return 1 << thread_bits; }

	/**
	 * Return length of weight vector
	 * @return length of weight vector
	 */
	inline vw_size_t length() { return 1 << num_bits; }

	/**
	 * Set a new stride value.
	 * Also changes thread_mask.
	 *
	 * @param new_stride new value of stride
	 */
	void set_stride(vw_size_t new_stride);

	/**
	 * Return the name of the object
	 *
	 * @return VwEnvironment
	 */
	virtual const char* get_name() const { return "VwEnvironment"; }

private:
	/**
	 * Initialize to default values
	 */
	virtual void init();

public:
	/// log_2 of the number of features
	vw_size_t num_bits;
	/// log_2 of the number of threads
	vw_size_t thread_bits;
	/// Mask used for hashing
	vw_size_t mask;
	/// Mask used by regressor for learning
	vw_size_t thread_mask;
	/// Number of elements in weight vector per feature
	vw_size_t stride;

	/// Smallest label seen
	float64_t min_label;
	/// Largest label seen
	float64_t max_label;

	/// Learning rate
	float32_t eta;
	/// Decay rate of eta per pass
	float32_t eta_decay_rate;

	/// Whether adaptive learning is used
	bool adaptive;
	/// Whether exact norm is used for adaptive learning
	bool exact_adaptive_norm;
	/// Level of L1 regularization
	float32_t l1_regularization;

	/// Whether to use random weights
	bool random_weights;
	/// Initial value of all elements in weight vector
	float32_t initial_weight;

	/// Sum of updates
	float32_t update_sum;

	/// Value of t
	float32_t t;
	/// Initial value of t
	float64_t initial_t;
	/// t power value while updating
	float32_t power_t;

	/// Example number
	int64_t example_number;
	/// Weighted examples
	float64_t weighted_examples;
	/// Weighted unlabelled examples
	float64_t weighted_unlabeled_examples;
	/// Weighted labels
	float64_t weighted_labels;
	/// Total number of features
	vw_size_t total_features;
	/// Sum of losses
	float64_t sum_loss;
	/// Number of passes complete
	vw_size_t passes_complete;
	/// Number of passes
	vw_size_t num_passes;

	/// ngrams to generate
	vw_size_t ngram;
	/// Skips in ngrams
	vw_size_t skips;

	/// Whether some namespaces are ignored
	bool ignore_some;
	/// Which namespaces to ignore
	bool ignore[256];

	/// Pairs of features to cross for quadratic updates
	DynArray<char*> pairs;

	/// VW version
	const char* vw_version;
	/// Length of version string
	vw_size_t v_length;
};

}
#endif // _VW_ENV_H__
