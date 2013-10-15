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
}

template<class T> SGNDArray<T>::SGNDArray(index_t* d, index_t nd, bool ref_counting) :
	SGReferencedData(ref_counting), dims(d), num_dims(nd)
{
	index_t total = 1;
	for (int32_t i=0; i<num_dims; i++)
		total *= dims[i];
	ASSERT(total>0)
	array = SG_MALLOC(T, total);
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
}

template<class T> void SGNDArray<T>::init_data()
{
	array = NULL;
	dims = NULL;
	num_dims = 0;
}

template<class T> void SGNDArray<T>::free_data()
{
	SG_FREE(array);
	SG_FREE(dims);

	array     = NULL;
	dims      = NULL;
	num_dims  = 0;
}

template<class T> void SGNDArray<T>::transpose_matrix(index_t matIdx) const
{
	ASSERT(array && dims && num_dims > 2 && dims[2] > matIdx)

	T aux;
	// Index to acces directly the elements of the matrix of interest
	int32_t idx = matIdx*dims[0]*dims[1];

	for (int32_t i=0; i<dims[0]; i++)
		for (int32_t j=0; j<i-1; j++)
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
