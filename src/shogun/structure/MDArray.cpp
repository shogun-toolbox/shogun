/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Copyright (C) 2014 Jiaolong Xu
 */
#include <shogun/structure/MDArray.h>
#include <math.h>
#include <float.h>

using namespace shogun;

CMDArray::CMDArray():
	SGNDArray<float64_t>()
{
	init_data();
	m_flat_length = 0;
}

CMDArray::CMDArray(vector<int32_t> &base_sizes)
{
	init_data();

	m_base_sizes = base_sizes;
	m_flat_length = 1;

	for (uint32_t i = 0; i < base_sizes.size(); i++)
	{
		m_flat_length *= base_sizes[i];
	}

	num_dims = base_sizes.size();
	dims = SG_MALLOC(index_t, num_dims);

	for (int32_t i = 0; i < num_dims; i++)
	{
		dims[i] = base_sizes[i];
	}

	array = SG_MALLOC(float64_t, m_flat_length);

	// Initialize array to zero
	for (int32_t i = 0; i < m_flat_length; i++)
	{
		array[i] = 0.0;
	}
}

CMDArray::CMDArray(const CMDArray &v)
{
	m_base_sizes = v.m_base_sizes;
	m_flat_length = v.m_flat_length;
	array = SG_MALLOC(float64_t, m_flat_length);
	memcpy(array, v.array, m_flat_length * sizeof(float64_t));

	num_dims = m_base_sizes.size();
	dims = SG_MALLOC(index_t, num_dims);

	for (int32_t i = 0; i < num_dims; i++)
	{
		dims[i] = m_base_sizes[i];
	}
}

CMDArray &CMDArray::operator=(const CMDArray &v)
{
	free_data();

	m_base_sizes = v.m_base_sizes;
	m_flat_length = v.m_flat_length;

	array = SG_MALLOC(float64_t, m_flat_length);
	memcpy(array, v.array, m_flat_length * sizeof(float64_t));

	num_dims = m_base_sizes.size();
	dims = SG_MALLOC(index_t, num_dims);

	for (int32_t i = 0; i < num_dims; i++)
	{
		dims[i] = m_base_sizes[i];
	}

	return (*this);
}

CMDArray &CMDArray::operator*=(float64_t val)
{
	for (int32_t i = 0; i < m_flat_length; i++)
	{
		array[i] *= val;
	}
	return (*this);
}

CMDArray &CMDArray::operator=(float64_t val)
{
	for (int32_t i = 0; i < m_flat_length; i++)
	{
		array[i] = val;
	}
	return (*this);
}

CMDArray &CMDArray::operator+=(CMDArray &v)
{
	ASSERT(m_base_sizes.size() == v.m_base_sizes.size());

	for (int32_t i = 0; i < m_flat_length; i++)
	{
		array[i] += v.array[i];
	}

	return (*this);
}

CMDArray &CMDArray::operator-=(CMDArray &v)
{
	for (int32_t i = 0; i < m_flat_length; i++)
	{
		array[i] -= v.array[i];
	}
	return (*this);
}

float64_t CMDArray::max_element(int32_t &max_at) const
{
	float64_t m = array[0];
	max_at = 0;
	for (int32_t i = 1; i < m_flat_length; i++)
	{
		if (array[i] >= m)
		{
			max_at = i;
			m = array[i];
		}
	}
	return m;
}

float64_t CMDArray::get_value(vector<int32_t> &indices) const
{
	int32_t y = 0;
	int32_t fact = 1;
	int32_t nx = m_base_sizes.size();

	for (int32_t i = nx - 1; i >= 0; i--)
	{
		y += indices[i] * fact;
		fact *= m_base_sizes[i];
	}

	return array[y];
}

void CMDArray::next_index(vector<int32_t> &inds) const
{
	for (int32_t i = inds.size() - 1; i >= 0; i--)
	{
		inds[i]++;
		if (inds[i] < m_base_sizes[i])
		{
			break;
		}
		inds[i] = 0;
	}
}

void CMDArray::expand(CMDArray &big_array, vector<int32_t> dim_in_big)
{
	// TODO: A nice implementation would be a function like repmat in matlab
	REQUIRE(dim_in_big.size() <= 2, "Only 1-d and 2-d array can be expanded currently.");
	// Initialize indices in big array to zeros
	vector<int32_t> inds_big(big_array.m_base_sizes.size(), 0);

	// Replicate the small array to the big one.
	// Go over the big one by one and take the corresponding value
	float64_t* data_big = &big_array.array[0];
	for (int32_t vi = 0; vi < big_array.m_flat_length; vi++)
	{
		int32_t y = 0;

		if (dim_in_big.size() == 1)
		{
			y = inds_big[dim_in_big[0]];
		}
		else if (dim_in_big.size() == 2)
		{
			int32_t ind1 = dim_in_big[0];
			int32_t ind2 = dim_in_big[1];
			y = inds_big[ind1] * m_base_sizes[1] + inds_big[ind2];
		}

		*data_big = array[y];
		data_big++;

		// Move to the next index
		big_array.next_index(inds_big);
	}
}
