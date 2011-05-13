/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef __DATATYPE_H__
#define __DATATYPE_H__

#include "lib/common.h"

#define PT_NOT_GENERIC	PT_SGOBJECT
#define PT_LONGEST		floatmax_t

namespace shogun
{

typedef int32_t index_t;

template<class T> struct SGVector
{
	/** default constructor */
	SGVector() : vector(NULL), length(0) { }

	/** constructor for setting params */
	SGVector(T* v, index_t len) : vector(v), length(len) { }

	/** copy constructor */
	SGVector(const SGVector &orig)
	: vector(orig.vector), length(orig.length) { }

	/** vector  */
	T* vector;
	/** length of vector  */
	index_t length;
};

template<class T> struct SGMatrix
{
	/** matrix  */
	T* matrix;
	/** number of rows of matrix  */
	index_t num_rows;
	/** number of columns of matrix  */
	index_t num_cols;
};

template<class T> struct SGString
{
	/** string  */
	T* string;
	/** length of string  */
	index_t length;
};

/** template class SGSparseMatrixEntry */
template <class T> struct SGSparseMatrixEntry
{
	/** feature index  */
	index_t feat_index;
	/** entry ... */
	T entry;
};

/** template class SGSparseMatrix */
template <class T> struct SGSparseMatrix
{
	/** vector index */
	index_t vec_index;
	/** number of feature entries */
	index_t num_feat_entries;
	/** features */
	SGSparseMatrixEntry<T>* features;
};

enum EContainerType
{
	CT_SCALAR,
	CT_VECTOR,
	CT_MATRIX,
	CT_NDARRAY
};

enum EStructType
{
	ST_NONE,
	ST_STRING,
	ST_SPARSE
};

enum EPrimitiveType
{
	PT_BOOL,
	PT_CHAR,
	PT_INT8,
	PT_UINT8,
	PT_INT16,
	PT_UINT16,
	PT_INT32,
	PT_UINT32,
	PT_INT64,
	PT_UINT64,
	PT_FLOAT32,
	PT_FLOAT64,
	PT_FLOATMAX,
	PT_SGOBJECT
};

/* Datatypes that shogun supports. */
struct TSGDataType
{
	EContainerType m_ctype;
	EStructType m_stype;
	EPrimitiveType m_ptype;
	index_t *m_length_y, *m_length_x;

	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype);
	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length);
	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length_y,
						 index_t* length_x);

	bool operator==(const TSGDataType& a);
	inline bool operator!=(const TSGDataType& a) {
		return !(*this == a);
	}

	void to_string(char* dest, size_t n) const;
	size_t sizeof_stype(void) const;
	size_t sizeof_ptype(void) const;

	static size_t sizeof_sparseentry(EPrimitiveType ptype);
	static size_t offset_sparseentry(EPrimitiveType ptype);

	static void stype_to_string(char* dest, EStructType stype,
								EPrimitiveType ptype, size_t n);
	static void ptype_to_string(char* dest, EPrimitiveType ptype,
								size_t n);
	static bool string_to_ptype(EPrimitiveType* ptype,
								const char* str);

	/**
	 * @return size of type in bytes
	 */
	size_t get_size();

	/**
	 * @return number of (matrix, vector, scalar) elements of type
	 */
	index_t get_num_elements();
};
}
#endif /* __DATATYPE_H__  */
