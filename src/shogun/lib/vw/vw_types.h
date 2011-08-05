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
#include <shogun/io/IOBuffer.h>
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
	void set_num_bits(size_t bits) { num_bits = bits; }

	/**
	 * Return number of bits used for weight vector
	 * @return number of bits
	 */
	size_t get_num_bits() { return num_bits; }

	/**
	 * Set mask used while accessing features
	 * @param m mask
	 */
	void set_mask(size_t m) { mask = m; }

	/**
	 * Return the mask used
	 * @return mask
	 */
	size_t get_mask() { return mask; }

	/**
	 * Return minimum label encountered
	 * @return min label
	 */
	double get_min_label() { return min_label; }

	/**
	 * Return maximum label encountered
	 * @return max label
	 */
	double get_max_label() { return max_label; }

	/**
	 * Return number of threads used for learning
	 * @return number of threads
	 */
	size_t num_threads() { return 1 << thread_bits; }

	/**
	 * Return length of weight vector
	 * @return length of weight vector
	 */
	size_t length() { return 1 << num_bits; }

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
		rank = 0;

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
	double min_label;
	double max_label;

	/// log_2 of the number of features
	size_t num_bits;
	/// log_2 of the number of threads
	size_t thread_bits;
	/// Mask used for hashing
	size_t mask;
	/// Mask used for computation of dot products
	size_t thread_mask;

	/// Whether some namespaces are ignored
	bool ignore_some;
	/// Which namespaces to ignore
	bool ignore[256];
	/// Pairs of features to cross
	std::vector<string> pairs;

	size_t stride;

	size_t passes_complete;

	bool sort_features;

	size_t rank;

	float eta;
	float eta_decay_rate;

	float l1_regularization;
	float update_sum;

	bool adaptive;

	bool random_weights;
	float initial_weight;

	float power_t;
	float t;

	double initial_t;
	long long int example_number;
	double weighted_examples;
	double weighted_unlabeled_examples;
	double weighted_labels;
	size_t total_features;
	double sum_loss;

};

class VwLabel
{

public:
	/**
	 * Parse a substring to get a label
	 *
	 * @param words substrings, each representing a token in the label data of the format
	 */
	VwLabel(): label(FLT_MAX), weight(1.), initial(0.)
	{
	}

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

	float get_weight()
	{
		return weight;
	}

	float get_initial()
	{
		return initial;
	}


public:
	/// Label value
	float label;

	/// Weight of example
	float weight;

	/// Initial approximation
	float initial;

};

class VwFeature
{
public:
	float64_t x;
	uint32_t weight_index;
	bool operator==(VwFeature j) { return weight_index == j.weight_index; }
};

class VwExample
{
public:
	/**
	 * Constructor, taking environment as optional argument
	 */
	VwExample(): tag(), indices(), atomics(),
		num_features(0), pass(0), final_prediction(0.),
		global_prediction(0), loss(0), eta_round(0.),
		eta_global(0), global_weight(0),
		example_t(0), total_sum_feat_sq(1), revert_weight(0)
	{
		ld = new VwLabel();
	}

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
	 * directly.
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
	VwLabel* ld;
	v_array<char> tag;
	size_t example_counter;

	v_array<size_t> indices;
	v_array<VwFeature> atomics[256];

	size_t num_features;
	size_t pass;
	float final_prediction;
	float global_prediction;
	float loss;
	float eta_round;
	float eta_global;
	float global_weight;
	float example_t;
	float64_t sum_feat_sq[256];
	float total_sum_feat_sq;
	float revert_weight;

	size_t ngram;
	size_t skips;

	bool sorted;
};

class VwRegressor
{
public:
	VwRegressor(VwEnvironment* env = NULL)
	{
		weight_vectors = NULL;
		loss = new CSquaredLoss();
		init(env);
	}

	~VwRegressor()
	{
		delete[] weight_vectors;
		SG_UNREF(loss);
	}

	float64_t get_loss(float64_t prediction, float64_t label)
	{
		return loss->loss(prediction, label);
	}

	float64_t get_update(float64_t prediction, float64_t label,
			     float64_t eta_t, float64_t norm)
	{
		return loss->get_update(prediction, label, eta_t, norm);
	}

	void init(VwEnvironment* env = NULL)
	{
		size_t length = ((size_t) 1) << env->num_bits;
		env->thread_mask = (env->stride * (length >> env->thread_bits)) - 1;

		size_t num_threads = 1;
		weight_vectors = new float*[num_threads];

		for (size_t i=0; i<num_threads; i++)
		{
			/* Initialize vectors to zero */
			weight_vectors[i] = new float[env->stride * length / num_threads]();

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
				for (size_t j = 0; j< env->stride*length/num_threads; j+=env->stride)
					weight_vectors[i][j] = env->initial_weight;

			if (env->adaptive)
				for (size_t j = 1; j< env->stride*length/num_threads; j+=env->stride)
					weight_vectors[i][j] = 1;
		}
	}


public:
	float** weight_vectors;

	CLossFunction* loss;
};

}
#endif
