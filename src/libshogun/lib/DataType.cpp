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

TSGDataType::TSGDataType(EContainerType ctype, EPrimitveType ptype)
{
	m_ctype = ctype; m_ptype = ptype; m_length_y = m_length_x = NULL;
}

TSGDataType::TSGDataType(EContainerType ctype, EPrimitveType ptype,
						 index_t* length)
{
	m_ctype = ctype; m_ptype = ptype; m_length_y = length;
	m_length_x = NULL;
}

TSGDataType::TSGDataType(EContainerType ctype, EPrimitveType ptype,
						 index_t* length_y, index_t* length_x)
{
	m_ctype = ctype; m_ptype = ptype; m_length_y = length_y;
	m_length_x = length_x;
}

void
TSGDataType::to_string(char* dest, size_t n) const
{
	char* p = dest;

	switch (m_ctype) {
	case CT_SCALAR: strncpy(p, "", n); break;
	case CT_VECTOR: strncpy(p, "Vector<", n); break;
	case CT_MATRIX: strncpy(p, "Matrix<", n); break;
	}

	size_t np = strlen(p);
	ptype_to_string(p + np, m_ptype, n - np - 2);

	switch (m_ctype) {
	case CT_SCALAR: break;
	case CT_VECTOR: case CT_MATRIX:
		strcat(p, ">"); break;
	}
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
	case PT_SGSERIALIZABLE_PTR: return sizeof (CSGSerializable*);
	}

	return -1;
}

void
TSGDataType::ptype_to_string(char* dest, EPrimitveType ptype, size_t n)
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
	case PT_SGSERIALIZABLE_PTR: strncpy(p, "SGSerializable*", n); break;
	}
}

bool
TSGDataType::string_to_ptype(EPrimitveType* result, const char* str)
{
	if (strcmp(str, "bool") == 0) {
		*result = PT_BOOL; return true; }
	if (strcmp(str, "char") == 0) {
		*result = PT_CHAR; return true; }
	if (strcmp(str, "int8") == 0) {
		*result = PT_INT8; return true; }
	if (strcmp(str, "uint8") == 0) {
		*result = PT_UINT8; return true; }
	if (strcmp(str, "int16") == 0) {
		*result = PT_INT16; return true; }
	if (strcmp(str, "uint16") == 0) {
		*result = PT_UINT16; return true; }
	if (strcmp(str, "int32") == 0) {
		*result = PT_INT32; return true; }
	if (strcmp(str, "uint32") == 0) {
		*result = PT_UINT32; return true; }
	if (strcmp(str, "int64") == 0) {
		*result = PT_INT64; return true; }
	if (strcmp(str, "uint64") == 0) {
		*result = PT_UINT64; return true; }
	if (strcmp(str, "float32") == 0) {
		*result = PT_FLOAT32; return true; }
	if (strcmp(str, "float64") == 0) {
		*result = PT_FLOAT64; return true; }
	if (strcmp(str, "floatmax") == 0) {
		*result = PT_FLOATMAX; return true; }
	if (strcmp(str, "SGSerializable*") == 0) {
		*result = PT_SGSERIALIZABLE_PTR; return true; }

	/* Make sure that the compiler will warn at this position.  */
	switch (*result) {
	case PT_BOOL: case PT_CHAR: case PT_INT8: case PT_UINT8:
	case PT_INT16: case PT_UINT16: case PT_INT32: case PT_UINT32:
	case PT_INT64: case PT_UINT64: case PT_FLOAT32: case PT_FLOAT64:
	case PT_FLOATMAX: case PT_SGSERIALIZABLE_PTR: break;
	}

	return false;
}
