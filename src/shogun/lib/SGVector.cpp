/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Thoralf Klein
 * Written (W) 2011-2013 Heiko Strathmann
 * Written (W) 2013 Soumyajit De
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGReferencedData.h>
#include <shogun/io/File.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>
#include <algorithm>

#include <shogun/mathematics/eigen3.h>

#define COMPLEX128_ERROR_NOARG(function) \
template <> \
void SGVector<complex128_t>::function() \
{ \
	SG_SERROR("SGVector::%s():: Not supported for complex128_t\n",\
		#function);\
}

#define BOOL_ERROR_ONEARG(function) \
template <> \
void SGVector<bool>::function(bool a) \
{ \
	SG_SERROR("SGVector::%s():: Not supported for bool\n",\
		#function);\
}

#define COMPLEX128_ERROR_ONEARG(function) \
template <> \
void SGVector<complex128_t>::function(complex128_t a) \
{ \
	SG_SERROR("SGVector::%s():: Not supported for complex128_t\n",\
		#function);\
}

#define COMPLEX128_ERROR_TWOARGS(function) \
template <> \
void SGVector<complex128_t>::function(complex128_t a, complex128_t b) \
{ \
	SG_SERROR("SGVector::%s():: Not supported for complex128_t\n",\
		#function);\
}

#define COMPLEX128_ERROR_THREEARGS(function) \
template <> \
void SGVector<complex128_t>::function(complex128_t a, complex128_t b,\
	complex128_t c) \
{ \
	SG_SERROR("SGVector::%s():: Not supported for complex128_t\n",\
		#function);\
}

namespace shogun {

template<class T>
SGVector<T>::SGVector() : SGReferencedData()
{
	init_data();
}

template<class T>
SGVector<T>::SGVector(T* v, index_t len, bool ref_counting)
: SGReferencedData(ref_counting), vector(v), vlen(len)
{
}

template<class T>
SGVector<T>::SGVector(index_t len, bool ref_counting)
: SGReferencedData(ref_counting), vlen(len)
{
	vector=SG_MALLOC(T, len);
}

template<class T>
SGVector<T>::SGVector(const SGVector &orig) : SGReferencedData(orig)
{
	copy_data(orig);
}

template<class T>
void SGVector<T>::set(SGVector<T> orig)
{
	*this = SGVector<T>(orig);
}

template<class T>
SGVector<T>::~SGVector()
{
	unref();
}

#ifdef HAVE_EIGEN3
template <class T> 
SGVector<T>::SGVector(EigenVectorXt& vec)
: SGReferencedData(false), vector(vec.data()), vlen(vec.size())
{
	
}

template <class T> 
SGVector<T>::SGVector(EigenRowVectorXt& vec)
: SGReferencedData(false), vector(vec.data()), vlen(vec.size())
{
	
}

template <class T> 
SGVector<T>::operator EigenVectorXtMap() const
{
	return EigenVectorXtMap(vector, vlen);
}

template <class T> 
SGVector<T>::operator EigenRowVectorXtMap() const
{
	return EigenRowVectorXtMap(vector, vlen);
}
#endif

template<class T>
void SGVector<T>::zero()
{
	if (vector && vlen)
		set_const(0);
}

template <>
void SGVector<complex128_t>::zero()
{
	if (vector && vlen)
		set_const(complex128_t(0.0));
}

template<class T>
void SGVector<T>::set_const(T const_elem)
{
	for (index_t i=0; i<vlen; i++)
		vector[i]=const_elem ;
}

#if HAVE_CATLAS
template<>
void SGVector<float64_t>::set_const(float64_t const_elem)
{
	catlas_dset(vlen, const_elem, vector, 1);
}

template<>
void SGVector<float32_t>::set_const(float32_t const_elem)
{
	catlas_sset(vlen, const_elem, vector, 1);
}
#endif // HAVE_CATLAS

template<class T>
void SGVector<T>::range_fill(T start)
{
	range_fill_vector(vector, vlen, start);
}

COMPLEX128_ERROR_ONEARG(range_fill)

template<class T>
void SGVector<T>::random(T min_value, T max_value)
{
	random_vector(vector, vlen, min_value, max_value);
}

COMPLEX128_ERROR_TWOARGS(random)

template <class T>
index_t SGVector<T>::find_position_to_insert(T element)
{
	index_t i;
	for (i=0; i<vlen; ++i)
	{
		if (vector[i]>element)
			break;
	}
	return i;
}

template <>
index_t SGVector<complex128_t>::find_position_to_insert(complex128_t element)
{
	SG_SERROR("SGVector::find_position_to_insert():: \
		Not supported for complex128_t\n");
	return index_t(-1);
}

template<class T>
SGVector<T> SGVector<T>::clone() const
{
	return SGVector<T>(clone_vector(vector, vlen), vlen);
}

template<class T>
T* SGVector<T>::clone_vector(const T* vec, int32_t len)
{
	T* result = SG_MALLOC(T, len);
	memcpy(result, vec, sizeof(T)*len);
	return result;
}

template<class T>
void SGVector<T>::fill_vector(T* vec, int32_t len, T value)
{
	for (int32_t i=0; i<len; i++)
		vec[i]=value;
}

template<class T>
void SGVector<T>::range_fill_vector(T* vec, int32_t len, T start)
{
	for (int32_t i=0; i<len; i++)
		vec[i]=i+start;
}

template <>
void SGVector<complex128_t>::range_fill_vector(complex128_t* vec,
	int32_t len, complex128_t start)
{
	SG_SERROR("SGVector::range_fill_vector():: \
		Not supported for complex128_t\n");
}

template<class T>
const T& SGVector<T>::get_element(index_t index)
{
	ASSERT(vector && (index>=0) && (index<vlen))
	return vector[index];
}

template<class T>
void SGVector<T>::set_element(const T& p_element, index_t index)
{
	ASSERT(vector && (index>=0) && (index<vlen))
	vector[index]=p_element;
}

template<class T>
void SGVector<T>::resize_vector(int32_t n)
{
	vector=SG_REALLOC(T, vector, vlen, n);

	if (n > vlen)
		memset(&vector[vlen], 0, (n-vlen)*sizeof(T));
	vlen=n;
}

/** addition operator */
template<class T>
SGVector<T> SGVector<T>::operator+ (SGVector<T> x)
{
	ASSERT(x.vector && vector)
	ASSERT(x.vlen == vlen)

	SGVector<T> result=clone();
	result.add(x);
	return result;
}

template<class T>
void SGVector<T>::add(const SGVector<T> x)
{
	ASSERT(x.vector && vector)
	ASSERT(x.vlen == vlen)

	for (int32_t i=0; i<vlen; i++)
		vector[i]+=x.vector[i];
}

template<class T>
void SGVector<T>::add(const T x)
{
	ASSERT(vector)

	for (int32_t i=0; i<vlen; i++)
		vector[i]+=x;
}

template<class T>
void SGVector<T>::add(const SGSparseVector<T>& x)
{
	if (x.features)
	{
		for (int32_t i=0; i < x.num_feat_entries; i++)
		{
			index_t idx = x.features[i].feat_index;
			ASSERT(idx < vlen)
			vector[idx] += x.features[i].entry;
		}
	}
}

template<class T>
void SGVector<T>::display_size() const
{
	SG_SPRINT("SGVector '%p' of size: %d\n", vector, vlen)
}

template<class T>
void SGVector<T>::copy_data(const SGReferencedData &orig)
{
	vector=((SGVector*)(&orig))->vector;
	vlen=((SGVector*)(&orig))->vlen;
}

template<class T>
void SGVector<T>::init_data()
{
	vector=NULL;
	vlen=0;
}

template<class T>
void SGVector<T>::free_data()
{
	SG_FREE(vector);
	vector=NULL;
	vlen=0;
}

template<class T>
bool SGVector<T>::equals(SGVector<T>& other)
{
	if (other.vlen!=vlen)
		return false;

	for (index_t i=0; i<vlen; ++i)
	{
		if (other.vector[i]!=vector[i])
			return false;
	}

	return true;
}

template<class T>
void SGVector<T>::display_vector(const char* name,
		const char* prefix) const
{
	display_vector(vector, vlen, name, prefix);
}

template <class T>
void SGVector<T>::display_vector(const SGVector<T> vector, const char* name,
		const char* prefix)
{
	vector.display_vector(prefix);
}

template <>
void SGVector<bool>::display_vector(const bool* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%d%s", prefix, vector[i] ? 1 : 0, i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<char>::display_vector(const char* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%c%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<uint8_t>::display_vector(const uint8_t* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%u%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<int8_t>::display_vector(const int8_t* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%d%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<uint16_t>::display_vector(const uint16_t* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%u%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<int16_t>::display_vector(const int16_t* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%d%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<int32_t>::display_vector(const int32_t* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%d%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<uint32_t>::display_vector(const uint32_t* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%u%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}


template <>
void SGVector<int64_t>::display_vector(const int64_t* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%lld%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<uint64_t>::display_vector(const uint64_t* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%llu%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<float32_t>::display_vector(const float32_t* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%g%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<float64_t>::display_vector(const float64_t* vector, int32_t n, const char* name,
		const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%s%.18g%s", prefix, vector[i], i==n-1? "" : ",")
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<floatmax_t>::display_vector(const floatmax_t* vector, int32_t n,
		const char* name, const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
	{
		SG_SPRINT("%s%.36Lg%s", prefix, (long double) vector[i],
				i==n-1? "" : ",");
	}
	SG_SPRINT("%s]\n", prefix)
}

template <>
void SGVector<complex128_t>::display_vector(const complex128_t* vector, int32_t n,
		const char* name, const char* prefix)
{
	ASSERT(n>=0)
	SG_SPRINT("%s%s=[", prefix, name)
	for (int32_t i=0; i<n; i++)
	{
		SG_SPRINT("%s(%.36lg+i%.36lg)%s", prefix, vector[i].real(),
				vector[i].imag(), i==n-1? "" : ",");
	}
	SG_SPRINT("%s]\n", prefix)
}

template <class T>
void SGVector<T>::vec1_plus_scalar_times_vec2(T* vec1,
		const T scalar, const T* vec2, int32_t n)
{
	for (int32_t i=0; i<n; i++)
		vec1[i]+=scalar*vec2[i];
}

template <>
void SGVector<float64_t>::vec1_plus_scalar_times_vec2(float64_t* vec1,
		float64_t scalar, const float64_t* vec2, int32_t n)
{
#ifdef HAVE_LAPACK
	int32_t skip=1;
	cblas_daxpy(n, scalar, vec2, skip, vec1, skip);
#else
	for (int32_t i=0; i<n; i++)
		vec1[i]+=scalar*vec2[i];
#endif
}

template <>
void SGVector<float32_t>::vec1_plus_scalar_times_vec2(float32_t* vec1,
		float32_t scalar, const float32_t* vec2, int32_t n)
{
#ifdef HAVE_LAPACK
	int32_t skip=1;
	cblas_saxpy(n, scalar, vec2, skip, vec1, skip);
#else
	for (int32_t i=0; i<n; i++)
		vec1[i]+=scalar*vec2[i];
#endif
}

template <class T>
	void SGVector<T>::random_vector(T* vec, int32_t len, T min_value, T max_value)
	{
		for (int32_t i=0; i<len; i++)
			vec[i]=CMath::random(min_value, max_value);
	}

template <>
void SGVector<complex128_t>::random_vector(complex128_t* vec, int32_t len,
	complex128_t min_value, complex128_t max_value)
{
	SG_SNOTIMPLEMENTED
}

template <>
bool SGVector<bool>::twonorm(const bool* x, int32_t len)
{
	SG_SNOTIMPLEMENTED
	return false;
}

template <>
char SGVector<char>::twonorm(const char* x, int32_t len)
{
	SG_SNOTIMPLEMENTED
	return '\0';
}

template <>
int8_t SGVector<int8_t>::twonorm(const int8_t* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <>
uint8_t SGVector<uint8_t>::twonorm(const uint8_t* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <>
int16_t SGVector<int16_t>::twonorm(const int16_t* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <>
uint16_t SGVector<uint16_t>::twonorm(const uint16_t* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <>
int32_t SGVector<int32_t>::twonorm(const int32_t* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <>
uint32_t SGVector<uint32_t>::twonorm(const uint32_t* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <>
int64_t SGVector<int64_t>::twonorm(const int64_t* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <>
uint64_t SGVector<uint64_t>::twonorm(const uint64_t* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <>
float32_t SGVector<float32_t>::twonorm(const float32_t* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <>
float64_t SGVector<float64_t>::twonorm(const float64_t* v, int32_t n)
{
	float64_t norm = 0.0;
#ifdef HAVE_LAPACK
	norm = cblas_dnrm2(n, v, 1);
#else
	norm = CMath::sqrt(CMath::dot(v, v, n));
#endif
	return norm;
}

template <>
floatmax_t SGVector<floatmax_t>::twonorm(const floatmax_t* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <>
complex128_t SGVector<complex128_t>::twonorm(const complex128_t* x, int32_t len)
{
	complex128_t result(0.0);
	for (int32_t i=0; i<len; i++)
		result+=x[i]*x[i];

	return CMath::sqrt(result);
}

template <class T>
float64_t SGVector<T>::onenorm(T* x, int32_t len)
{
	float64_t result=0;
	for (int32_t i=0;i<len; ++i)
		result+=CMath::abs(x[i]);

	return result;
}

/// || x ||_q^q
template <class T>
T SGVector<T>::qsq(T* x, int32_t len, float64_t q)
{
	float64_t result=0;
	for (int32_t i=0; i<len; i++)
		result+=CMath::pow(fabs(x[i]), q);

	return result;
}

template <>
complex128_t SGVector<complex128_t>::qsq(complex128_t* x, int32_t len, float64_t q)
{
	SG_SNOTIMPLEMENTED
	return complex128_t(0.0);
}

/// || x ||_q
template <class T>
T SGVector<T>::qnorm(T* x, int32_t len, float64_t q)
{
	ASSERT(q!=0)
	return CMath::pow((float64_t) qsq(x, len, q), 1.0/q);
}

template <>
complex128_t SGVector<complex128_t>::qnorm(complex128_t* x, int32_t len, float64_t q)
{
	SG_SNOTIMPLEMENTED
	return complex128_t(0.0);
}

/// return sum(abs(vec))
template <class T>
T SGVector<T>::sum_abs(T* vec, int32_t len)
{
	T result=0;
	for (int32_t i=0; i<len; i++)
		result+=CMath::abs(vec[i]);

	return result;
}

#if HAVE_LAPACK
template <>
float64_t SGVector<float64_t>::sum_abs(float64_t* vec, int32_t len)
{
	float64_t result=0;
	result = cblas_dasum(len, vec, 1);
	return result;
}

template <>
float32_t SGVector<float32_t>::sum_abs(float32_t* vec, int32_t len)
{
	float32_t result=0;
	result = cblas_sasum(len, vec, 1);
	return result;
}
#endif

/// return sum(abs(vec))
template <class T>
bool SGVector<T>::fequal(T x, T y, float64_t precision)
{
	return CMath::abs(x-y)<precision;
}

template <class T>
int32_t SGVector<T>::unique(T* output, int32_t size)
{
	CMath::qsort<T>(output, size);
	int32_t j=0;

	for (int32_t i=0; i<size; i++)
	{
		if (i==0 || output[i]!=output[i-1])
			output[j++]=output[i];
	}
	return j;
}

template <>
int32_t SGVector<complex128_t>::unique(complex128_t* output, int32_t size)
{
	int32_t j=0;
	SG_SERROR("SGVector::unique():: Not supported for complex128_t\n");
	return j;
}

template <class T>
SGVector<index_t> SGVector<T>::find(T elem)
{
	SGVector<index_t> idx(vlen);
	index_t k=0;

	for (index_t i=0; i < vlen; ++i)
		if (vector[i] == elem)
			idx[k++] = i;
	idx.vlen = k;
	return idx;
}

template<class T>
void SGVector<T>::scale_vector(T alpha, T* vec, int32_t len)
{
	for (int32_t i=0; i<len; i++)
		vec[i]*=alpha;
}

#ifdef HAVE_LAPACK
template<>
void SGVector<float64_t>::scale_vector(float64_t alpha, float64_t* vec, int32_t len)
{
	cblas_dscal(len, alpha, vec, 1);
}

template<>
void SGVector<float32_t>::scale_vector(float32_t alpha, float32_t* vec, int32_t len)
{
	cblas_sscal(len, alpha, vec, 1);
}
#endif

template<class T>
void SGVector<T>::scale(T alpha)
{
	scale_vector(alpha, vector, vlen);
}

template<class T> void SGVector<T>::load(CFile* loader)
{
	ASSERT(loader)
	unref();

	SG_SET_LOCALE_C;
	SGVector<T> vec;
	loader->get_vector(vec.vector, vec.vlen);
	copy_data(vec);
	copy_refcount(vec);
	ref();
	SG_RESET_LOCALE;
}

template<>
void SGVector<complex128_t>::load(CFile* loader)
{
	SG_SERROR("SGVector::load():: Not supported for complex128_t\n");
}

template<class T> void SGVector<T>::save(CFile* saver)
{
	ASSERT(saver)

	SG_SET_LOCALE_C;
	saver->set_vector(vector, vlen);
	SG_RESET_LOCALE;
}

template<>
void SGVector<complex128_t>::save(CFile* saver)
{
	SG_SERROR("SGVector::save():: Not supported for complex128_t\n");
}

template <class T> SGVector<float64_t> SGVector<T>::get_real()
{
	SGVector<float64_t> real(vlen);
	for (int32_t i=0; i<vlen; i++)
		real[i]=CMath::real(vector[i]);
	return real;
}

template <class T> SGVector<float64_t> SGVector<T>::get_imag()
{
	SGVector<float64_t> imag(vlen);
	for (int32_t i=0; i<vlen; i++)
		imag[i]=CMath::imag(vector[i]);
	return imag;
}

template <class T>
SGMatrix<T> SGVector<T>::convert_to_matrix(SGVector<T> vector,
	index_t nrows, index_t ncols, bool fortran_order)
{
	if (nrows*ncols>vector.size())
		SG_SERROR("SGVector::convert_to_matrix():: Dimensions mismatch\n");

	T* data=NULL;
	SGVector<T>::convert_to_matrix(data, nrows, ncols, vector.vector, vector.vlen, fortran_order);

	SGMatrix<T> matrix=SGMatrix<T>(data, nrows, ncols);
	return matrix;
}

template <class T>
void SGVector<T>::convert_to_matrix(T*& matrix, index_t nrows, index_t ncols, const T* vector, int32_t vlen, bool fortran_order)
{
	if (nrows*ncols>vlen)
		SG_SERROR("SGVector::convert_to_matrix():: Dimensions mismatch\n");

	if (matrix!=NULL)
		SG_FREE(matrix);
	matrix=SG_MALLOC(T, nrows*ncols);

	if (fortran_order)
	{
		for (index_t i=0; i<ncols*nrows; i++)
			matrix[i]=vector[i];
	}
	else
	{
		for (index_t i=0; i<nrows; i++)
		{
			for (index_t j=0; j<ncols; j++)
				matrix[i+j*nrows]=vector[j+i*ncols];
		}
	}
}

#define UNDEFINED(function, type)	\
template <>	\
SGVector<float64_t> SGVector<type>::function()	\
{	\
	SG_SERROR("SGVector::%s():: Not supported for %s\n",	\
		#function, #type);	\
	SGVector<float64_t> ret(vlen);	\
	return ret;	\
}

UNDEFINED(get_real, bool)
UNDEFINED(get_real, char)
UNDEFINED(get_real, int8_t)
UNDEFINED(get_real, uint8_t)
UNDEFINED(get_real, int16_t)
UNDEFINED(get_real, uint16_t)
UNDEFINED(get_real, int32_t)
UNDEFINED(get_real, uint32_t)
UNDEFINED(get_real, int64_t)
UNDEFINED(get_real, uint64_t)
UNDEFINED(get_real, float32_t)
UNDEFINED(get_real, float64_t)
UNDEFINED(get_real, floatmax_t)
UNDEFINED(get_imag, bool)
UNDEFINED(get_imag, char)
UNDEFINED(get_imag, int8_t)
UNDEFINED(get_imag, uint8_t)
UNDEFINED(get_imag, int16_t)
UNDEFINED(get_imag, uint16_t)
UNDEFINED(get_imag, int32_t)
UNDEFINED(get_imag, uint32_t)
UNDEFINED(get_imag, int64_t)
UNDEFINED(get_imag, uint64_t)
UNDEFINED(get_imag, float32_t)
UNDEFINED(get_imag, float64_t)
UNDEFINED(get_imag, floatmax_t)
#undef UNDEFINED

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
template class SGVector<complex128_t>;
}

#undef COMPLEX128_ERROR_NOARG
#undef COMPLEX128_ERROR_ONEARG
#undef COMPLEX128_ERROR_TWOARGS
#undef COMPLEX128_ERROR_THREEARGS
