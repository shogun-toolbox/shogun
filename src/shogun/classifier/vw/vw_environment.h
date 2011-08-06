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

#include <shogun/lib/DataType.h>
#include <shogun/lib/common.h>
#include <shogun/lib/v_array.h>

#include <string>
#include <vector>

namespace shogun
{

using std::string;

/** @brief Class VwEnvironment is the environment used by VW.
 *
 * Contains global constants and settings which change the behaviour
 * of Vowpal Wabbit.
 *
 * It is used while parsing input, and also while learning.
 *
 * One VwEnvironment object should be bound to the CStreamingVwFile or
 * CStreamingVwCacheFile, and the pointer to it propagated upwards
 * to CStreamingVwFeatures and finally to CVowpalWabbit.
 */
class VwEnvironment
{
public:
	/**
	 * Default constructor
	 * Should initialize with reasonable default values
	 */
	VwEnvironment();

	/**
	 * Destructor
	 */
	~VwEnvironment() { }

	/**
	 * Set number of bits used for the weight vector
	 * @param bits number of bits
	 */
	inline void set_num_bits(index_t bits) { num_bits = bits; }

	/**
	 * Return number of bits used for weight vector
	 * @return number of bits
	 */
	inline index_t get_num_bits() { return num_bits; }

	/**
	 * Set mask used while accessing features
	 * @param m mask
	 */
	inline void set_mask(index_t m) { mask = m; }

	/**
	 * Return the mask used
	 * @return mask
	 */
	inline index_t get_mask() { return mask; }

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
	inline index_t num_threads() { return 1 << thread_bits; }

	/**
	 * Return length of weight vector
	 * @return length of weight vector
	 */
	inline index_t length() { return 1 << num_bits; }

private:
	/**
	 * Initialize to default values
	 */
	void init();

public:
	/// log_2 of the number of features
	index_t num_bits;
	/// log_2 of the number of threads
	index_t thread_bits;
	/// Mask used for hashing
	index_t mask;
	/// Mask used by regressor for learning
	index_t thread_mask;
	/// Number of elements in weight vector per feature
	index_t stride;

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
	index_t total_features;
	/// Sum of losses
	float64_t sum_loss;
	/// Number of passes complete
	index_t passes_complete;

	/// Whether some namespaces are ignored
	bool ignore_some;
	/// Which namespaces to ignore
	bool ignore[256];

	/// Pairs of features to cross for quadratic updates
	std::vector<string> pairs;
};

}
#endif // _VW_ENV_H__
