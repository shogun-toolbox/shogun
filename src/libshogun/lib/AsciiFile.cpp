/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "lib/AsciiFile.h"

using namespace shogun;

CAsciiFile::CAsciiFile(void) :CFile()
{
}

CAsciiFile::CAsciiFile(FILE* f, char rw) : CFile(f, rw)
{
}

CAsciiFile::CAsciiFile(char* fname, char rw) : CFile(fname, rw)
{
}

CAsciiFile::~CAsciiFile(void)
{
}

bool
CAsciiFile::write_type(const TSGDataType* type, const void* param,
					   const char* name, const char* prefix)
{
	if (!is_task_warn('w')) return false_warn(prefix, name);

	char buf[50];
	type->to_string(buf);

	if (fprintf(file, "%s:%s:%s:", prefix, name, buf) < 0)
		return false_warn(prefix, name);

	switch (type->m_ctype) {
	case CT_SCALAR:
		switch (type->m_ptype) {
		case PT_BOOL:
			if (fprintf(file, "%s", *(bool*) param? "true": "false")
				< 0) return false_warn(prefix, name);
			break;
		case PT_CHAR:
			if (fprintf(file, "%"PRIu8, *(char*) param) < 0)
				return false_warn(prefix, name);
			break;
		case PT_INT16:
			if (fprintf(file, "%"PRIu16, *(uint16_t*) param) < 0)
				return false_warn(prefix, name);
			break;
		case PT_INT32:
			if (fprintf(file, "%"PRIu32, *(uint32_t*) param) < 0)
				return false_warn(prefix, name);
			break;
		case PT_INT64:
			if (fprintf(file, "%"PRIu64, *(uint64_t*) param) < 0)
				return false_warn(prefix, name);
			break;
		case PT_FLOAT32:
			if (fprintf(file, "%+10.16e", *(float32_t*) param) < 0)
				return false_warn(prefix, name);
			break;
		case PT_FLOAT64:
			if (fprintf(file, "%+10.16e", *(float64_t*) param) < 0)
				return false_warn(prefix, name);
			break;
		case PT_FLOATMAX:
			if (fprintf(file, "%+10.16Le", *(floatmax_t*) param) < 0)
				return false_warn(prefix, name);
			break;
		case PT_SGOBJECT_PTR:
			SG_ERROR("Implementation error during writing AsciiFile!");
			return false_warn(prefix, name);
		}
		break;
	case CT_VECTOR: case CT_MATRIX: case CT_STRING:
		SG_NOTIMPLEMENTED;
		break;
	}

	if (fprintf(file, "\n") < 0) return false_warn(prefix, name);

	return true;
}

bool
CAsciiFile::read_type(const TSGDataType* type, void* param,
					  const char* name, const char* prefix)
{
	if (!is_task_warn('r')) return false;

	SG_PRINT("reading: %s %s\n", prefix, name);

	return true;
}
