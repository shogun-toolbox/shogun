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

template<class T> struct TString
{
	/** string  */
	T* string;
	/** length of string  */
	index_t length;
};

/** template class TSparseEntry */
template <class T> struct TSparseEntry
{
	/** feature index  */
	index_t feat_index;
	/** entry ... */
	T entry;
};

/** template class TSparse */
template <class T> struct TSparse
{
	/** vector index */
	index_t vec_index;
	/** number of feature entries */
	index_t num_feat_entries;
	/** features */
	TSparseEntry<T>* features;
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
};
}
#endif /* __DATATYPE_H__  */
