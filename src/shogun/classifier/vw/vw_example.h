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

#ifndef _VW_EXAMPLE_H__
#define _VW_EXAMPLE_H__

#include <lib/DataType.h>
#include <lib/common.h>
#include <lib/v_array.h>
#include <classifier/vw/vw_constants.h>
#include <classifier/vw/vw_label.h>

namespace shogun
{

/** @brief One feature in VW
 *
 * Has the value of the feature as a float, and the hashed index
 * of the feature in the weight vector.
 */
class VwFeature
{
public:
	/// Feature value
	float32_t x;

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
	VwExample();

	/**
	 * Destructor
	 */
	~VwExample();

	/**
	 * Resets the members so the values can be updated
	 * directly without constructing another object.
	 */
	void reset_members();

public:
	/// Label object
	VwLabel* ld;

	/// Tag
	v_array<char> tag;
	/// Array of namespaces
	v_array<vw_size_t> indices;
	/// Array of features
	v_array<VwFeature> atomics[256];

	/// Number of features
	vw_size_t num_features;
	/// Pass
	vw_size_t pass;
	/// Final prediction
	float32_t final_prediction;
	/// Loss
	float32_t loss;
	/// Learning rate for this round
	float32_t eta_round;
	/// Global weight
	float32_t global_weight;
	/// t value for this example
	float32_t example_t;

	/// Sum of square of features
	float64_t sum_feat_sq[256];
	/// Total sum of square of features
	float32_t total_sum_feat_sq;

	/// Example counter
	vw_size_t example_counter;
	/// Whether features are sorted by weight index
	bool sorted;
};

}
#endif // _VW_EXAMPLE_H__
