/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "lib/SerialAsciiFile.h"

using namespace shogun;

CSerialAsciiFile::CSerialAsciiFile(void)
	:CSerialFile() {}

CSerialAsciiFile::CSerialAsciiFile(FILE* f, char rw)
	:CSerialFile(f, rw) {}

CSerialAsciiFile::CSerialAsciiFile(char* fname, char rw)
	:CSerialFile(fname, rw) {}

CSerialAsciiFile::~CSerialAsciiFile() {}

bool
CSerialAsciiFile::write_scalar(EPrimitveType type, const void* param)
{
	switch (type) {
	case PT_BOOL:
		if (fprintf(file, "%s", *(bool*) param? "true": "false") < 0)
			return false;
		break;
	case PT_CHAR:
		if (fprintf(file, "%"PRIu8, *(char*) param) < 0)
			return false;
		break;
	case PT_INT16:
		if (fprintf(file, "%"PRIi16, *(int16_t*) param) < 0)
			return false;
		break;
	case PT_UINT16:
		if (fprintf(file, "%"PRIu16, *(uint16_t*) param) < 0)
			return false;
		break;
	case PT_INT32:
		if (fprintf(file, "%"PRIi32, *(int32_t*) param) < 0)
			return false;
		break;
	case PT_UINT32:
		if (fprintf(file, "%"PRIu32, *(uint32_t*) param) < 0)
			return false;
		break;
	case PT_INT64:
		if (fprintf(file, "%"PRIi64, *(int64_t*) param) < 0)
			return false;
		break;
	case PT_UINT64:
		if (fprintf(file, "%"PRIu64, *(uint64_t*) param) < 0)
			return false;
		break;
	case PT_FLOAT32:
		if (fprintf(file, "%+10.16e", *(float32_t*) param) < 0)
			return false;
		break;
	case PT_FLOAT64:
		if (fprintf(file, "%+10.16e", *(float64_t*) param) < 0)
			return false;
		break;
	case PT_FLOATMAX:
		if (fprintf(file, "%+10.16Le", *(floatmax_t*) param) < 0)
			return false;
		break;
	case PT_SGOBJECT_PTR:
		SG_ERROR("Implementation error during writing AsciiFile!");
		return false;
	}

	return true;
}

bool
CSerialAsciiFile::write_vector(const TSGDataType* type, const void* param,
							   uint64_t length)
{
	length = *(void**) param == NULL? 0: length;

	if (fprintf(file, "%"PRIu64, length) < 0) return false;

	for (uint64_t i=0; i<length; i++)
		if (fprintf(file, ":") < 0
			|| !write_scalar(type->m_ptype, (*(char**) param)
							 + i*type->sizeof_ptype())) return false;

	return true;
}

bool
CSerialAsciiFile::write_type_wrapped(
	const TSGDataType* type, const void* param, const char* name,
	const char* prefix)
{
	char buf[50];
	type->to_string(buf);

	if (fprintf(file, "%s:%s:%s:", prefix, name, buf) < 0)
		return false;

	switch (type->m_ctype) {
	case CT_SCALAR:
		if (!write_scalar(type->m_ptype, param))
			return false;
		break;
	case CT_VECTOR:
		if (!write_vector(type, param, *type->m_length_y))
			return false;
		break;
	case CT_MATRIX:
		if (!write_vector(type, param,
						  *type->m_length_y **type->m_length_x))
			return false;
		break;
	case CT_STRING: case CT_SPARSE:
		SG_NOTIMPLEMENTED;
		break;
	}

	if (fprintf(file, "\n") < 0) return false;

	return true;
}

bool
CSerialAsciiFile::read_type_wrapped(
	const TSGDataType* type, void* param, const char* name,
	const char* prefix)
{
	SG_PRINT("reading: %s %s\n", prefix, name);

	return true;
}
