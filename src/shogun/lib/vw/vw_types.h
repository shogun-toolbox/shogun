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
#ifndef _VW_TYPES_H__
#define _VW_TYPES_H__

#include <shogun/lib/common.h>
#include <shogun/lib/v_array.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/vw/substring.h>
#include <shogun/lib/vw/parse_primitives.h>
#include <shogun/loss/SquaredLoss.h>

#include <math.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

namespace shogun
{

using std::string;

typedef size_t (*hash_func_t)(substring, unsigned long);

const int quadratic_constant = 27942141;
const int constant = 11650396;

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
	VwEnvironment() { init(); }

	/**
	 * Destructor
	 */
	~VwEnvironment() { }

	/**
	 * Set number of bits used for the weight vector
	 * @param bits number of bits
	 */
	inline void set_num_bits(size_t bits) { num_bits = bits; }

	/**
	 * Return number of bits used for weight vector
	 * @return number of bits
	 */
	inline size_t get_num_bits() { return num_bits; }

	/**
	 * Set mask used while accessing features
	 * @param m mask
	 */
	inline void set_mask(size_t m) { mask = m; }

	/**
	 * Return the mask used
	 * @return mask
	 */
	inline size_t get_mask() { return mask; }

	/**
	 * Return minimum label encountered
	 * @return min label
	 */
	inline double get_min_label() { return min_label; }

	/**
	 * Return maximum label encountered
	 * @return max label
	 */
	inline double get_max_label() { return max_label; }

	/**
	 * Return number of threads used for learning
	 * @return number of threads
	 */
	inline size_t num_threads() { return 1 << thread_bits; }

	/**
	 * Return length of weight vector
	 * @return length of weight vector
	 */
	inline size_t length() { return 1 << num_bits; }

private:
	/**
	 * Initialize to default values
	 */
	void init()
	{
		num_bits = 18;
		thread_bits = 0;
		mask = (1 << num_bits) - 1;
		stride = 1;

		min_label = 0.;
		max_label = 1.;

		eta = 10.;
		eta_decay_rate = 1.;

		adaptive = false;
		l1_regularization = 0.;

		random_weights = false;
		initial_weight = 0.;

		update_sum = 0.;

		t = 1.;
		initial_t = 1.;
		power_t = 0.5;

		example_number = 0;
		weighted_examples = 0.;
		weighted_unlabeled_examples = 0.;
		weighted_labels = 0.;
		total_features = 0;
		sum_loss = 0.;
		passes_complete = 0;

		ignore_some = false;
	}

public:
	/// log_2 of the number of features
	size_t num_bits;
	/// log_2 of the number of threads
	size_t thread_bits;
	/// Mask used for hashing
	size_t mask;
	/// Mask used by regressor for learning
	size_t thread_mask;
	/// Number of elements in weight vector per feature
	size_t stride;

	/// Smallest label seen
	double min_label;
	/// Largest label seen
	double max_label;

	/// Learning rate
	float eta;
	/// Decay rate of eta per pass
	float eta_decay_rate;

	/// Whether adaptive learning is used
	bool adaptive;
	/// Level of L1 regularization
	float l1_regularization;

	/// Whether to use random weights
	bool random_weights;
	/// Initial value of all elements in weight vector
	float initial_weight;

	/// Sum of updates
	float update_sum;

	/// Value of t
	float t;
	/// Initial value of t
	double initial_t;
	/// t power value while updating
	float power_t;

	/// Example number
	long long int example_number;
	/// Weighted examples
	double weighted_examples;
	/// Weighted unlabelled examples
	double weighted_unlabeled_examples;
	/// Weighted labels
	double weighted_labels;
	/// Total number of features
	size_t total_features;
	/// Sum of losses
	double sum_loss;
	/// Number of passes complete
	size_t passes_complete;

	/// Whether some namespaces are ignored
	bool ignore_some;
	/// Which namespaces to ignore
	bool ignore[256];

	/// Pairs of features to cross for quadratic updates
	std::vector<string> pairs;
};

/** @brief Class VwLabel holds a label object used by VW.
 *
 * Has 3 members: the label value, weight of the example and
 * initial value of the label.
 */
class VwLabel
{
public:
	/**
	 * Default Constructor
	 */
	VwLabel(): label(FLT_MAX), weight(1.), initial(0.) { }

	/**
	 * Destructor
	 */
	~VwLabel() { }

	/**
	 * Get label value
	 * @return label
	 */
	inline float get_label() { return label; }

	/**
	 * Set label value
	 * @param l label value
	 */
	inline void set_label(float l) { label = l; }

	/**
	 * Get weight
	 * @return example weight
	 */
	inline float get_weight() { return weight; }

	/**
	 * Set weight
	 * @param w example weight
	 */
	inline void set_weight(float w) { weight = w; }

	/**
	 * Get initial value
	 * @return initial value
	 */
	inline float get_initial() { return initial; }

	/**
	 * Set initial value
	 * @param i initial value
	 */
	inline void set_initial(float i) { initial = i; }

	/**
	 * Parse a substring to get a label
	 *
	 * @param words substrings, each representing a token in the label data of the format
	 */
	void parse_label(v_array<substring>& words)
	{
		switch(words.index())
		{
		case 0:
			break;
		case 1:
			label = float_of_substring(words[0]);
			break;
		case 2:
			label = float_of_substring(words[0]);
			weight = float_of_substring(words[1]);
			break;
		case 3:
			label = float_of_substring(words[0]);
			weight = float_of_substring(words[1]);
			initial = float_of_substring(words[2]);
			break;
		default:
			SG_SERROR("malformed example!\n"
				  "words.index() = %d\n", words.index());
		}
	}

public:
	/// Label value
	float label;
	/// Weight of example
	float weight;
	/// Initial approximation
	float initial;
};

/** @brief One feature in VW
 *
 * Has the value of the feature as a float, and the hashed index
 * of the feature in the weight vector.
 */
class VwFeature
{
public:
	/// Feature value
	float x;

	/// Hashed index in weight vector
	uint32_t weight_index;

	/**
	 * Overloaded equals operator
	 *
	 * @param j another feature object
	 *
	 * @return whether the index of the two features is the same
	 */
	bool operator==(VwFeature j) { return weight_index == j.weight_index; }
};

/** @brief Example class for VW
 *
 * It contains a label object pointer.
 * These objects should be returned by the parser.
 */
class VwExample
{
public:
	/**
	 * Constructor
	 */
	VwExample(): tag(), indices(), atomics(),
		num_features(0), pass(0), final_prediction(0.),
		global_prediction(0), loss(0), eta_round(0.),
		eta_global(0), global_weight(0),
		example_t(0), total_sum_feat_sq(1), revert_weight(0)
	{
		ld = new VwLabel();
	}

	/**
	 * Destructor
	 */
	~VwExample()
	{
		if (ld)
			delete ld;
		if (tag.end_array != tag.begin)
		{
			free(tag.begin);
			tag.end_array = tag.begin;
		}

		for (size_t j = 0; j < 256; j++)
		{
			if (atomics[j].begin != atomics[j].end_array)
				free(atomics[j].begin);
		}
		free(indices.begin);
	}

	/**
	 * Resets the members so the values can be updated
	 * directly without constructing another object.
	 */
	void reset_members()
	{
		num_features = 0;
		total_sum_feat_sq = 1;
		example_counter = 0;
		global_weight = 0;
		example_t = 0;
		eta_round = 0;
		final_prediction = 0;
		loss = 0;

		for (size_t* i = indices.begin; i != indices.end; i++)
		{
			atomics[*i].erase();
			sum_feat_sq[*i]=0;
		}

		indices.erase();
		tag.erase();
		sorted = false;
	}

public:
	/// Label object
	VwLabel* ld;

	/// Tag
	v_array<char> tag;
	/// Array of namespaces
	v_array<size_t> indices;
	/// Array of features
	v_array<VwFeature> atomics[256];

	/// Number of features
	size_t num_features;
	/// Pass
	size_t pass;
	/// Final prediction
	float final_prediction;
	/// Loss
	float loss;
	/// Learning rate for this round
	float eta_round;
	/// Global weight
	float global_weight;
	/// t value for this example
	float example_t;

	/// Sum of square of features
	float64_t sum_feat_sq[256];
	/// Total sum of square of features
	float total_sum_feat_sq;

	/// Example counter
	size_t example_counter;
};

/** @brief Regressor used by VW
 *
 * Stores the weight vectors and loss object.
 */
class VwRegressor
{
public:
	/**
	 * Default constructor, optionally taking an environment object
	 *
	 * @param env vw environment
	 */
	VwRegressor(VwEnvironment* env = NULL)
	{
		weight_vectors = NULL;
		loss = new CSquaredLoss();
		init(env);
	}

	/**
	 * Destructor
	 */
	~VwRegressor()
	{
		SG_FREE(weight_vectors);
		SG_UNREF(loss);
	}

	/**
	 * Get loss for a label-prediction set
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return loss
	 */
	inline float64_t get_loss(float64_t prediction, float64_t label)
	{
		return loss->loss(prediction, label);
	}

	/**
	 * Get weight update for a prediction-label set
	 *
	 * @param prediction prediction
	 * @param label label
	 * @param eta_t learning rate
	 * @param norm scaling norm
	 *
	 * @return update
	 */
	inline float64_t get_update(float64_t prediction, float64_t label,
				    float64_t eta_t, float64_t norm)
	{
		return loss->get_update(prediction, label, eta_t, norm);
	}

	/**
	 * Initialize weight vectors
	 *
	 * @param env environment object
	 */
	void init(VwEnvironment* env = NULL)
	{
		/* For each feature, there should be 'stride' number of elements in the weight vector */
		size_t length = ((size_t) 1) << env->num_bits;
		env->thread_mask = (env->stride * (length >> env->thread_bits)) - 1;

		/* Only one learning thread for now */
		size_t num_threads = 1;
		weight_vectors = SG_CALLOC(float*, num_threads);

		for (size_t i=0; i<num_threads; i++)
		{
			weight_vectors[i] = SG_CALLOC(float, env->stride * length / num_threads);

			if (env->random_weights)
			{
				if (env->rank > 0)
					for (size_t j = 0; j < env->stride*length/num_threads; j++)
						weight_vectors[i][j] = (double) 0.1 * rand() / ((double) RAND_MAX + 1.0);
				else
					for (size_t j = 0; j < length/num_threads; j++)
						weight_vectors[i][j] = drand48() - 0.5;
			}

			if (env->initial_weight != 0.)
				for (size_t j = 0; j < env->stride*length/num_threads; j+=env->stride)
					weight_vectors[i][j] = env->initial_weight;

			if (env->adaptive)
				for (size_t j = 1; j < env->stride*length/num_threads; j+=env->stride)
					weight_vectors[i][j] = 1;
		}
	}


public:
	/// Weight vectors, one array for each thread
	float** weight_vectors;
	/// Loss function
	CLossFunction* loss;
};
}
#endif // _VW_TYPES_H__
