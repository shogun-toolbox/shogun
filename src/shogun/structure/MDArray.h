/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Copyright (C) 2014 Jiaolong Xu
 */
#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/io/SGIO.h>

#include <vector>

using namespace std;

#ifndef _MD_ARRAY_
#define _MD_ARRAY_
namespace shogun
{
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CMDArray : public SGNDArray<float64_t>
{
public:
	/** Default constructor */
	CMDArray();

	/** Constructor, initialize to all zero */
	CMDArray(vector<int32_t> &base_sizes);

	/** Copy constructor */
	CMDArray(const CMDArray &v);

	CMDArray &operator=(const CMDArray &v);
	CMDArray &operator=(float64_t val);
	CMDArray &operator*=(float64_t val);
	CMDArray &operator+=(CMDArray &v);
	CMDArray &operator-=(CMDArray &v);

	/** Find the maximum element in the array and return its index.
	 *
	 * @param max_at index in the array
	 *
	 * @return maximum element
	 */
	float64_t max_element(int32_t &max_at) const;

	/** Expand to a big size array
	 *
	 * @param big_array the big aray
	 * @param dims_in_big dimention index in the big array
	 */
	void expand(CMDArray &big_array, vector<int32_t> dims_in_big);

	/** Get the value at the index
	 *
	 * @param indecies the indecis in the array
	 *
	 * @return the value
	 */
	float64_t get_value(vector<int32_t> &indices) const;

	/** Get the next index from current one
	 *
	 * @param inds the index at current place
	 */
	void next_index(vector<int32_t> &inds) const;

public:
	vector<int32_t> m_base_sizes; // the size of the base
	int32_t m_flat_length; // the length of the array (in a flat 1-d vector)
};
}
#endif
