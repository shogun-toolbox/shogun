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
namespace shogun
{
enum EContainerType {
	CT_SCALAR, CT_VECTOR, CT_MATRIX, CT_STRING, CT_SPARSE
};

enum EPrimitveType {
	PT_BOOL, PT_CHAR, PT_INT16, PT_INT32, PT_INT64, PT_FLOAT32,
	PT_FLOAT64, PT_FLOATMAX, PT_SGOBJECT_PTR
};

/* Datatypes that shogun supports. */
struct TSGDataType
{
	explicit TSGDataType(EContainerType ctype, EPrimitveType ptype) {
		m_ctype = ctype; m_ptype = ptype; m_length = 0;
	}
	explicit TSGDataType(EContainerType ctype, EPrimitveType ptype,
						 uint64_t length) {
		m_ctype = ctype; m_ptype = ptype; m_length = length;
	}
	explicit TSGDataType(EContainerType ctype, EPrimitveType ptype,
						 uint64_t length_y, uint64_t length_x) {
		m_ctype = ctype; m_ptype = ptype;
		m_length = length_x * length_y;
	}

	bool operator==(const TSGDataType& a) {
		return m_ctype == a.m_ctype && m_ptype == a.m_ptype
			&& m_length == a.m_length;
	}

	bool operator!=(const TSGDataType& a) {
		return !(*this == a);
	}

	void to_string(char* dest) const {
		char* p = dest;

		switch (m_ctype) {
		case CT_SCALAR: strcpy(p, ""); break;
		case CT_VECTOR: strcpy(p, "Vector<"); break;
		case CT_MATRIX: strcpy(p, "Matrix<"); break;
		case CT_STRING: strcpy(p, "String<"); break;
		case CT_SPARSE: strcpy(p, "Sparse<"); break;
		}

		switch (m_ptype) {
		case PT_BOOL: strcat(p, "bool"); break;
		case PT_CHAR: strcat(p, "char"); break;
		case PT_INT16: strcat(p, "int16"); break;
		case PT_INT32: strcat(p, "int32"); break;
		case PT_INT64: strcat(p, "int64"); break;
		case PT_FLOAT32: strcat(p, "float32"); break;
		case PT_FLOAT64: strcat(p, "float64"); break;
		case PT_FLOATMAX: strcat(p, "floatmax"); break;
		case PT_SGOBJECT_PTR: strcat(p, "SGObject*"); break;
		}

		switch (m_ctype) {
		case CT_SCALAR: break;
		case CT_VECTOR: case CT_MATRIX: case CT_STRING: case CT_SPARSE:
			strcat(p, ">*"); break;
		}
	} /* TO_STRING()  */

	EContainerType m_ctype;
	EPrimitveType m_ptype;
	uint64_t m_length;
};
}
#endif // __DATATYPE_H__
