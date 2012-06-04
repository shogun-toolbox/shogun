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
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGReferencedData.h>

using namespace shogun;

template<class T> SGVector<T>::SGVector() : SGReferencedData(false)
{
	init_data();
}

template<class T> SGVector<T>::SGVector(T* v, index_t len, bool ref_counting)
: SGReferencedData(ref_counting), vector(v), vlen(len)
{
}

template<class T> SGVector<T>::SGVector(index_t len, bool ref_counting)
: SGReferencedData(ref_counting), vlen(len)
{
	vector=SG_MALLOC(T, len);
}

template<class T> SGVector<T>::SGVector(const SGVector &orig) : SGReferencedData(orig)
{
	copy_data(orig);
}

template<class T> SGVector<T>::~SGVector()
{
	unref();
}

template<class T> void SGVector<T>::zero()
{
	if (vector && vlen)
		set_const(0);
}

template<class T> void SGVector<T>::set_const(T const_elem)
{
	for (index_t i=0; i<vlen; i++)
		vector[i]=const_elem ;
}

template<class T> void SGVector<T>::range_fill(T start)
{
	range_fill_vector(vector, vlen, start);
}

template<class T> void SGVector<T>::random(T min_value, T max_value)
{
	random_vector(vector, vlen, min_value, max_value);
}

template<class T> void SGVector<T>::randperm()
{
	/* this does not work. Heiko Strathmann */
	SG_SNOTIMPLEMENTED;
	randperm(vector, vlen);
}

template<class T> SGVector<T> SGVector<T>::clone() const
{
	return SGVector<T>(clone_vector(vector, vlen), vlen);
}

template<class T> const T& SGVector<T>::get_element(index_t index)
{
	ASSERT(vector && (index>=0) && (index<vlen));
	return vector[index];
}

template<class T> void SGVector<T>::set_element(const T& p_element, index_t index)
{
	ASSERT(vector && (index>=0) && (index<vlen));
	vector[index]=p_element;
}

template<class T> void SGVector<T>::resize_vector(int32_t n)
{
	vector=SG_REALLOC(T, vector, n);

	if (n > vlen)
		memset(&vector[vlen], 0, (n-vlen)*sizeof(T));
	vlen=n;
}

template<class T> void SGVector<T>::add(const SGVector<T> x)
{
	ASSERT(x.vector && vector);
	ASSERT(x.vlen == vlen);

	for (int32_t i=0; i<vlen; i++)
		vector[i]+=x.vector[i];
}

template<class T> void SGVector<T>::display_size() const
{
	SG_SPRINT("SGVector '%p' of size: %d\n", vector, vlen);
}

template<class T> void SGVector<T>::display_vector() const
{
	display_size();
	for (int32_t i=0; i<vlen; i++)
		SG_SPRINT("%10.10g,", (float64_t) vector[i]);
	SG_SPRINT("\n");
}

template<class T> void SGVector<T>::copy_data(const SGReferencedData &orig)
{
	vector=((SGVector*)(&orig))->vector;
	vlen=((SGVector*)(&orig))->vlen;
}

template<class T> void SGVector<T>::init_data()
{
	vector=NULL;
	vlen=0;
}

template<class T> void SGVector<T>::free_data()
{
	SG_FREE(vector);
	vector=NULL;
	vlen=0;
}

template class SGVector<bool>;
template class SGVector<char>;
template class SGVector<int8_t>;
template class SGVector<uint8_t>;
template class SGVector<int16_t>;
template class SGVector<uint16_t>;
template class SGVector<int32_t>;
template class SGVector<uint32_t>;
template class SGVector<int64_t>;
template class SGVector<uint64_t>;
template class SGVector<float32_t>;
template class SGVector<float64_t>;
template class SGVector<floatmax_t>;
