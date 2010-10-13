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

#define PT_NOT_GENERIC             PT_SGSERIALIZABLE_PTR

namespace shogun
{

typedef int32_t                    index_t;

enum EContainerType {
	CT_SCALAR, CT_VECTOR, CT_MATRIX
};

enum EPrimitveType {
	PT_BOOL, PT_CHAR, PT_INT8, PT_UINT8, PT_INT16, PT_UINT16, PT_INT32,
	PT_UINT32, PT_INT64, PT_UINT64, PT_FLOAT32, PT_FLOAT64, PT_FLOATMAX,
	PT_SGSERIALIZABLE_PTR
};

/* Datatypes that shogun supports. */
struct TSGDataType
{
	EContainerType m_ctype;
	EPrimitveType m_ptype;
	index_t *m_length_y, *m_length_x;

	explicit TSGDataType(EContainerType ctype, EPrimitveType ptype);
	explicit TSGDataType(EContainerType ctype, EPrimitveType ptype,
						 index_t* length);
	explicit TSGDataType(EContainerType ctype, EPrimitveType ptype,
						 index_t* length_y, index_t* length_x);

	inline bool operator==(const TSGDataType& a) {
		return m_ctype == a.m_ctype && m_ptype == a.m_ptype
			&& *m_length_y == *a.m_length_y
			&& *m_length_x == *a.m_length_x;
	}

	inline bool operator!=(const TSGDataType& a) {
		return !(*this == a);
	}

	void to_string(char* dest, size_t n) const;
	size_t sizeof_ptype(void) const;

	static void ptype_to_string(char* dest, EPrimitveType ptype,
								size_t n);
	static bool string_to_ptype(EPrimitveType* result,
								const char* str);
};
}
#endif // __DATATYPE_H__
