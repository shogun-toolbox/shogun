/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <string.h>

#include "base/SGObject.h"
#include "lib/DataType.h"

using namespace shogun;

TSGDataType::TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype)
{
	m_ctype = ctype, m_stype = stype, m_ptype = ptype;
	m_length_y = m_length_x = NULL;
}

TSGDataType::TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length)
{
	m_ctype = ctype, m_stype = stype, m_ptype = ptype;
	m_length_y = length, m_length_x = NULL;
}

TSGDataType::TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length_y,
						 index_t* length_x)
{
	m_ctype = ctype, m_stype = stype, m_ptype = ptype;
	m_length_y = length_y, m_length_x = length_x;
}

bool
TSGDataType::operator==(const TSGDataType& a)
{
	bool result = m_ctype == a.m_ctype && m_stype == a.m_stype
		&& m_ptype == a.m_ptype;

	result &= m_length_y != NULL && a.m_length_y != NULL
		? *m_length_y == *a.m_length_y: m_length_y == a.m_length_y;
	result &= m_length_x != NULL && a.m_length_x != NULL
		? *m_length_x == *a.m_length_x: m_length_x == a.m_length_x;

	return result;
}

void
TSGDataType::to_string(char* dest, size_t n) const
{
	char* p = dest;

	switch (m_ctype) {
	case CT_SCALAR: strncpy(p, "", n); break;
	case CT_VECTOR: strncpy(p, "Vector<", n); break;
	case CT_MATRIX: strncpy(p, "Matrix<", n); break;
	case CT_NDARRAY: strncpy(p, "N-Dimensional Array<", n); break;
	}

	size_t np = strlen(p);
	stype_to_string(p + np, m_stype, m_ptype, n - np - 2);

	switch (m_ctype) {
	case CT_SCALAR: break;
	case CT_VECTOR: case CT_MATRIX: case CT_NDARRAY:
		strcat(p, ">"); break;
	}
}

size_t
TSGDataType::sizeof_stype(void) const
{
	switch (m_stype) {
	case ST_NONE: return sizeof_ptype();
	case ST_STRING:
		switch (m_ptype) {
		case PT_BOOL: return sizeof (TString<bool>);
		case PT_CHAR: return sizeof (TString<char>);
		case PT_INT8: return sizeof (TString<int8_t>);
		case PT_UINT8: return sizeof (TString<uint8_t>);
		case PT_INT16: return sizeof (TString<int16_t>);
		case PT_UINT16: return sizeof (TString<uint16_t>);
		case PT_INT32: return sizeof (TString<int32_t>);
		case PT_UINT32: return sizeof (TString<uint32_t>);
		case PT_INT64: return sizeof (TString<int64_t>);
		case PT_UINT64: return sizeof (TString<uint64_t>);
		case PT_FLOAT32: return sizeof (TString<float32_t>);
		case PT_FLOAT64: return sizeof (TString<float64_t>);
		case PT_FLOATMAX: return sizeof (TString<floatmax_t>);
		case PT_SGOBJECT: return -1;
		}
		break;
	case ST_SPARSE:
		switch (m_ptype) {
		case PT_BOOL: return sizeof (TSparse<bool>);
		case PT_CHAR: return sizeof (TSparse<char>);
		case PT_INT8: return sizeof (TSparse<int8_t>);
		case PT_UINT8: return sizeof (TSparse<uint8_t>);
		case PT_INT16: return sizeof (TSparse<int16_t>);
		case PT_UINT16: return sizeof (TSparse<uint16_t>);
		case PT_INT32: return sizeof (TSparse<int32_t>);
		case PT_UINT32: return sizeof (TSparse<uint32_t>);
		case PT_INT64: return sizeof (TSparse<int64_t>);
		case PT_UINT64: return sizeof (TSparse<uint64_t>);
		case PT_FLOAT32: return sizeof (TSparse<float32_t>);
		case PT_FLOAT64: return sizeof (TSparse<float64_t>);
		case PT_FLOATMAX: return sizeof (TSparse<floatmax_t>);
		case PT_SGOBJECT: return -1;
		}
		break;
	}

	return -1;
}

size_t
TSGDataType::sizeof_ptype(void) const
{
	switch (m_ptype) {
	case PT_BOOL: return sizeof (bool);
	case PT_CHAR: return sizeof (char);
	case PT_INT8: return sizeof (int8_t);
	case PT_UINT8: return sizeof (uint8_t);
	case PT_INT16: return sizeof (int16_t);
	case PT_UINT16: return sizeof (uint16_t);
	case PT_INT32: return sizeof (int32_t);
	case PT_UINT32: return sizeof (uint32_t);
	case PT_INT64: return sizeof (int64_t);
	case PT_UINT64: return sizeof (uint64_t);
	case PT_FLOAT32: return sizeof (float32_t);
	case PT_FLOAT64: return sizeof (float64_t);
	case PT_FLOATMAX: return sizeof (floatmax_t);
	case PT_SGOBJECT: return sizeof (CSGObject*);
	}

	return -1;
}

size_t
TSGDataType::sizeof_sparseentry(EPrimitiveType ptype)
{
	switch (ptype) {
	case PT_BOOL: return sizeof (TSparseEntry<bool>);
	case PT_CHAR: return sizeof (TSparseEntry<char>);
	case PT_INT8: return sizeof (TSparseEntry<int8_t>);
	case PT_UINT8: return sizeof (TSparseEntry<uint8_t>);
	case PT_INT16: return sizeof (TSparseEntry<int16_t>);
	case PT_UINT16: return sizeof (TSparseEntry<uint16_t>);
	case PT_INT32: return sizeof (TSparseEntry<int32_t>);
	case PT_UINT32: return sizeof (TSparseEntry<uint32_t>);
	case PT_INT64: return sizeof (TSparseEntry<int64_t>);
	case PT_UINT64: return sizeof (TSparseEntry<uint64_t>);
	case PT_FLOAT32: return sizeof (TSparseEntry<float32_t>);
	case PT_FLOAT64: return sizeof (TSparseEntry<float64_t>);
	case PT_FLOATMAX: return sizeof (TSparseEntry<floatmax_t>);
	case PT_SGOBJECT: return -1;
	}

	return -1;
}

#define ENTRY_OFFSET(k, type)									\
	((char*) &((TSparseEntry<type>*) (k))->entry - (char*) (k))
size_t
TSGDataType::offset_sparseentry(EPrimitiveType ptype)
{
	size_t result = -1; void* x = &result;

	switch (ptype) {
	case PT_BOOL: result = ENTRY_OFFSET(x, bool); break;
	case PT_CHAR: result = ENTRY_OFFSET(x, char); break;
	case PT_INT8: result = ENTRY_OFFSET(x, int8_t); break;
	case PT_UINT8: result = ENTRY_OFFSET(x, uint8_t); break;
	case PT_INT16: result = ENTRY_OFFSET(x, int16_t); break;
	case PT_UINT16: result = ENTRY_OFFSET(x, uint16_t); break;
	case PT_INT32: result = ENTRY_OFFSET(x, int32_t); break;
	case PT_UINT32: result = ENTRY_OFFSET(x, uint32_t); break;
	case PT_INT64: result = ENTRY_OFFSET(x, int64_t); break;
	case PT_UINT64: result = ENTRY_OFFSET(x, uint64_t); break;
	case PT_FLOAT32: result = ENTRY_OFFSET(x, float32_t); break;
	case PT_FLOAT64: result = ENTRY_OFFSET(x, float64_t); break;
	case PT_FLOATMAX: result = ENTRY_OFFSET(x, floatmax_t); break;
	case PT_SGOBJECT: return -1;
	}

	return result;
}

void
TSGDataType::stype_to_string(char* dest, EStructType stype,
							 EPrimitiveType ptype, size_t n)
{
	char* p = dest;

	switch (stype) {
	case ST_NONE: strncpy(p, "", n); break;
	case ST_STRING: strncpy(p, "String<", n); break;
	case ST_SPARSE: strncpy(p, "Sparse<", n); break;
	}

	size_t np = strlen(p);
	ptype_to_string(p + np, ptype, n - np - 2);

	switch (stype) {
	case ST_NONE: break;
	case ST_STRING: case ST_SPARSE:
		strcat(p, ">"); break;
	}
}

void
TSGDataType::ptype_to_string(char* dest, EPrimitiveType ptype,
							 size_t n)
{
	char* p = dest;

	switch (ptype) {
	case PT_BOOL: strncpy(p, "bool", n); break;
	case PT_CHAR: strncpy(p, "char", n); break;
	case PT_INT8: strncpy(p, "int8", n); break;
	case PT_UINT8: strncpy(p, "uint8", n); break;
	case PT_INT16: strncpy(p, "int16", n); break;
	case PT_UINT16: strncpy(p, "uint16", n); break;
	case PT_INT32: strncpy(p, "int32", n); break;
	case PT_UINT32: strncpy(p, "uint32", n); break;
	case PT_INT64: strncpy(p, "int64", n); break;
	case PT_UINT64: strncpy(p, "uint64", n); break;
	case PT_FLOAT32: strncpy(p, "float32", n); break;
	case PT_FLOAT64: strncpy(p, "float64", n); break;
	case PT_FLOATMAX: strncpy(p, "floatmax", n); break;
	case PT_SGOBJECT: strncpy(p, "SGSerializable*", n); break;
	}
}

bool
TSGDataType::string_to_ptype(EPrimitiveType* ptype, const char* str)
{
	if (strcmp(str, "bool") == 0) {
		*ptype = PT_BOOL; return true; }
	if (strcmp(str, "char") == 0) {
		*ptype = PT_CHAR; return true; }
	if (strcmp(str, "int8") == 0) {
		*ptype = PT_INT8; return true; }
	if (strcmp(str, "uint8") == 0) {
		*ptype = PT_UINT8; return true; }
	if (strcmp(str, "int16") == 0) {
		*ptype = PT_INT16; return true; }
	if (strcmp(str, "uint16") == 0) {
		*ptype = PT_UINT16; return true; }
	if (strcmp(str, "int32") == 0) {
		*ptype = PT_INT32; return true; }
	if (strcmp(str, "uint32") == 0) {
		*ptype = PT_UINT32; return true; }
	if (strcmp(str, "int64") == 0) {
		*ptype = PT_INT64; return true; }
	if (strcmp(str, "uint64") == 0) {
		*ptype = PT_UINT64; return true; }
	if (strcmp(str, "float32") == 0) {
		*ptype = PT_FLOAT32; return true; }
	if (strcmp(str, "float64") == 0) {
		*ptype = PT_FLOAT64; return true; }
	if (strcmp(str, "floatmax") == 0) {
		*ptype = PT_FLOATMAX; return true; }
	if (strcmp(str, "SGSerializable*") == 0) {
		*ptype = PT_SGOBJECT; return true; }

	/* Make sure that the compiler will warn at this position.  */
	switch (*ptype) {
	case PT_BOOL: case PT_CHAR: case PT_INT8: case PT_UINT8:
	case PT_INT16: case PT_UINT16: case PT_INT32: case PT_UINT32:
	case PT_INT64: case PT_UINT64: case PT_FLOAT32: case PT_FLOAT64:
	case PT_FLOATMAX: case PT_SGOBJECT: break;
	}

	return false;
}

size_t TSGDataType::get_size()
{
	size_t size;

	switch (m_stype)
	{
	case ST_NONE:
		size=get_num_elements()*sizeof_ptype();
		break;
	case ST_STRING:
		if (m_ptype==PT_SGOBJECT)
			return -1;

		size=get_num_elements()*sizeof_stype();
		break;
	case ST_SPARSE:
		if (m_ptype==PT_SGOBJECT)
			return -1;

		size=get_num_elements()*sizeof_sparseentry(m_ptype);
		break;
	}

	return size;
}

index_t TSGDataType::get_num_elements()
{
	index_t num_elements;

	switch (m_ctype)
	{
	case CT_SCALAR:
		num_elements=1;
		break;
	case CT_VECTOR:
		/* length_y contains the length for vectors */
		num_elements=*m_length_y;
		break;
	case CT_MATRIX:
		num_elements=(*m_length_y)*(*m_length_x);
		break;
	case CT_NDARRAY:
		SG_SNOTIMPLEMENTED;
		break;
	}

	return num_elements;
}
