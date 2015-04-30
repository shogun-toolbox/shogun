/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#include <shogun/lib/common.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/lib/SGReferencedData.h>

namespace shogun
{

template<class T> SGNDArray<T>::SGNDArray() :
	SGReferencedData()
{
	init_data();
}

template<class T> SGNDArray<T>::SGNDArray(T* a, index_t* d, index_t nd, bool ref_counting) :
	SGReferencedData(ref_counting)
{
	array = a;
	dims = d;
	num_dims = nd;
	len_array = 1;
	for (int32_t i=0; i<num_dims; i++)
		len_array *= dims[i];

	REQUIRE(len_array>0, "Length of array (%d) must be greater than 0\n", len_array);
}

template<class T> SGNDArray<T>::SGNDArray(index_t* d, index_t nd, bool ref_counting) :
	SGReferencedData(ref_counting), dims(d), num_dims(nd)
{
	len_array = 1;
	for (int32_t i=0; i<num_dims; i++)
		len_array *= dims[i];

	REQUIRE(len_array>0, "Length of array (%d) must be greater than 0\n", len_array);
	array = SG_MALLOC(T, len_array);
}

template<class T> SGNDArray<T>::SGNDArray(const SGVector<index_t> dimensions, bool ref_counting) :
	SGReferencedData(ref_counting)
{
	num_dims = dimensions.size();
	dims = SG_MALLOC(index_t, num_dims);

	len_array = 1;
	for (int32_t i=0; i<num_dims; i++)
	{
		dims[i] = dimensions[i];
		len_array *= dims[i];
	}

	REQUIRE(len_array>0, "Length of array (%d) must be greater than 0\n", len_array);
	array = SG_MALLOC(T, len_array);
}

template<class T> SGNDArray<T>::SGNDArray(const SGNDArray &orig) :
	SGReferencedData(orig)
{
	copy_data(orig);
}

template<class T> SGNDArray<T>::~SGNDArray()
{
	unref();
}

template<class T> void SGNDArray<T>::copy_data(const SGReferencedData &orig)
{
	array = ((SGNDArray*)(&orig))->array;
	dims = ((SGNDArray*)(&orig))->dims;
	num_dims = ((SGNDArray*)(&orig))->num_dims;
	len_array = ((SGNDArray*)(&orig))->len_array;
}

template<class T> void SGNDArray<T>::init_data()
{
	array = NULL;
	dims = NULL;
	num_dims = 0;
	len_array = 0;
}

template<class T> void SGNDArray<T>::free_data()
{
	SG_FREE(array);
	SG_FREE(dims);

	array     = NULL;
	dims      = NULL;
	num_dims  = 0;
	len_array = 0;
}

template<class T> SGNDArray<T> SGNDArray<T>::clone() const
{
	SGNDArray<T> array_clone(get_dimensions());
	memcpy(array_clone.array, array, sizeof(T)*len_array);
	return array_clone;
}

template<class T> SGVector<index_t> SGNDArray<T>::get_dimensions() const
{
	SGVector<index_t> dimensions(num_dims);

	for (int32_t i = 0; i < num_dims; i++)
		dimensions[i] = dims[i];

	return dimensions;
}

template<class T> void SGNDArray<T>::transpose_matrix(index_t matIdx) const
{
	REQUIRE(array && dims, "Array is empty.\n");
	REQUIRE(num_dims > 2, "Number of dimensions (%d) must be greater than 2.\n", num_dims);
	REQUIRE(dims[2] > matIdx, "Provided index (%d) is out of range, must be smaller than %d\n", matIdx, dims[2]);

	T aux;
	// Index to acces directly the elements of the matrix of interest
	int64_t idx = int64_t(matIdx)*int64_t(dims[0])*dims[1];

	for (int64_t i=0; i<dims[0]; i++)
		for (int64_t j=0; j<i-1; j++)
		{
			aux = array[idx + i + j*dims[0]];
			array[idx + i + j*dims[0]] = array[idx + j + i*dims[0]];
			array[idx + j + i*dims[1]] = aux;
		}

	// Swap the sizes of the two first dimensions
	index_t auxDim = dims[0];
	dims[0] = dims[1];
	dims[1] = auxDim;
}

template<class T> void SGNDArray<T>::set_const(T const_elem)
{
	for (index_t i = 0; i < len_array; i++)
		array[i] = const_elem;
}

template<class T>
SGNDArray<T>& SGNDArray<T>::operator*=(T val)
{
	for (index_t i = 0; i < len_array; i++)
		array[i] *= val;

	return (*this);
}

template<>
SGNDArray<bool>& SGNDArray<bool>::operator*=(bool val)
{
	SG_SNOTIMPLEMENTED;
	return (*this);
}

template<>
SGNDArray<char>& SGNDArray<char>::operator*=(char val)
{
	SG_SNOTIMPLEMENTED;
	return (*this);
}

template<class T>
SGNDArray<T>& SGNDArray<T>::operator+=(SGNDArray& ndarray)
{
	REQUIRE(len_array == ndarray.len_array,
			"The length of the given array (%d) does not match the length of internal array (%d).\n", ndarray.len_array, len_array);
	REQUIRE(num_dims == ndarray.num_dims,
			"The provided number of dimensions (%d) does not match the internal number of dimensions (%d).\n", ndarray.num_dims, num_dims);

	for (index_t i = 0; i < len_array; i++)
		array[i] += ndarray.array[i];

	return (*this);
}

template<>
SGNDArray<bool>& SGNDArray<bool>::operator+=(SGNDArray& ndarray)
{
	SG_SNOTIMPLEMENTED;
	return (*this);
}

template<>
SGNDArray<char>& SGNDArray<char>::operator+=(SGNDArray& ndarray)
{
	SG_SNOTIMPLEMENTED;
	return (*this);
}

template<class T>
SGNDArray<T>& SGNDArray<T>::operator-=(SGNDArray& ndarray)
{
	REQUIRE(len_array == ndarray.len_array,
			"The length of the given array (%d) does not match the length of internal array (%d).\n", ndarray.len_array, len_array);
	REQUIRE(num_dims == ndarray.num_dims,
			"The provided number of dimensions (%d) does not match the internal number of dimensions (%d).\n", ndarray.num_dims, num_dims);

	for (index_t i = 0; i < len_array; i++)
		array[i] -= ndarray.array[i];

	return (*this);
}

template<>
SGNDArray<bool>& SGNDArray<bool>::operator-=(SGNDArray& ndarray)
{
	SG_SNOTIMPLEMENTED;
	return (*this);
}

template<>
SGNDArray<char>& SGNDArray<char>::operator-=(SGNDArray& ndarray)
{
	SG_SNOTIMPLEMENTED;
	return (*this);
}

template<class T>
T SGNDArray<T>::max_element(int32_t &max_at)
{
	REQUIRE(len_array > 0, "Length of the array (%d) must be greater than 0.\n", len_array);

	T m = array[0];
	max_at = 0;

	for (int32_t i = 1; i < len_array; i++)
	{
		if (array[i] >= m)
		{
			max_at = i;
			m = array[i];
		}
	}

	return m;
}

template<>
bool SGNDArray<bool>::max_element(int32_t &max_at)
{
	SG_SNOTIMPLEMENTED;
	return false;
}

template<>
char SGNDArray<char>::max_element(int32_t &max_at)
{
	SG_SNOTIMPLEMENTED;
	return '\0';
}

template<class T>
T SGNDArray<T>::get_value(SGVector<index_t> index) const
{
	int32_t y = 0;
	int32_t fact = 1;

	REQUIRE(index.size() == num_dims,
			"Provided number of dimensions (%d) does not match internal number of dimensions (%d).\n", index.size(), num_dims);

	for (int32_t i = num_dims - 1; i >= 0; i--)
	{
		REQUIRE(index[i] < dims[i], "Provided index (%d) on dimension %d must be smaller than %d. \n", index[i], i, dims[i]);

		y += index[i] * fact;
		fact *= dims[i];
	}

	return array[y];
}

template<class T>
void SGNDArray<T>::next_index(SGVector<index_t>& curr_index) const
{
	REQUIRE(curr_index.size() == num_dims,
			"The provided number of dimensions (%d) does not match the internal number of dimensions (%d).\n", curr_index.size(), num_dims);

	for (int32_t i = num_dims - 1; i >= 0; i--)
	{
		curr_index[i]++;

		if (curr_index[i] < dims[i])
			break;

		curr_index[i] = 0;
	}
}

template<class T>
void SGNDArray<T>::expand(SGNDArray &big_array, SGVector<index_t>& axes)
{
	// TODO: A nice implementation would be a function like repmat in matlab
	REQUIRE(axes.size() <= 2,
			"Provided axes size (%d) must be smaller than 2.\n", axes.size());
	REQUIRE(num_dims <= 2,
			"Number of dimensions (%d) must be smaller than 2. Only 1-d and 2-d array can be expanded currently.\n", num_dims);

	// Initialize indices in big array to zeros
	SGVector<index_t> inds_big(big_array.num_dims);
	inds_big.zero();

	// Replicate the small array to the big one.
	// Go over the big one by one and take the corresponding value
	T* data_big = &big_array.array[0];
	for (int32_t vi = 0; vi < big_array.len_array; vi++)
	{
		int32_t y = 0;

		if (axes.size() == 1)
		{
			y = inds_big[axes[0]];
		}
		else if (axes.size() == 2)
		{
			int32_t ind1 = axes[0];
			int32_t ind2 = axes[1];
			y = inds_big[ind1] * dims[1] + inds_big[ind2];
		}

		*data_big = array[y];
		data_big++;

		// Move to the next index
		big_array.next_index(inds_big);
	}
}

template class SGNDArray<bool>;
template class SGNDArray<char>;
template class SGNDArray<int8_t>;
template class SGNDArray<uint8_t>;
template class SGNDArray<int16_t>;
template class SGNDArray<uint16_t>;
template class SGNDArray<int32_t>;
template class SGNDArray<uint32_t>;
template class SGNDArray<int64_t>;
template class SGNDArray<uint64_t>;
template class SGNDArray<float32_t>;
template class SGNDArray<float64_t>;
template class SGNDArray<floatmax_t>;
}
