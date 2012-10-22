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

#ifndef _VW_LABEL_H__
#define _VW_LABEL_H__

#include <shogun/lib/DataType.h>
#include <shogun/lib/common.h>
#include <shogun/lib/v_array.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

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
	inline float32_t get_label() { return label; }

	/**
	 * Set label value
	 * @param l label value
	 */
	inline void set_label(float32_t l) { label = l; }

	/**
	 * Get weight
	 * @return example weight
	 */
	inline float32_t get_weight() { return weight; }

	/**
	 * Set weight
	 * @param w example weight
	 */
	inline void set_weight(float32_t w) { weight = w; }

	/**
	 * Get initial value
	 * @return initial value
	 */
	inline float32_t get_initial() { return initial; }

	/**
	 * Set initial value
	 * @param i initial value
	 */
	inline void set_initial(float32_t i) { initial = i; }

	/**
	 * Parse a substring to get a label
	 *
	 * @param words substrings, each representing a token in the label data of the format
	 */
	void label_from_substring(v_array<substring>& words);

public:
	/// Label value
	float32_t label;
	/// Weight of example
	float32_t weight;
	/// Initial approximation
	float32_t initial;
};

}
#endif // _VW_LABEL_H__
