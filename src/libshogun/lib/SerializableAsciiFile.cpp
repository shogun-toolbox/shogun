/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "lib/SerializableAsciiFile.h"

#define STR_HEADER                 \
	"<<_SHOGON_SERIALIZABLE_ASCII_FILE_V_00_>>"

#define CHAR_CONT_BEGIN            '('
#define CHAR_CONT_END              ')'
#define CHAR_ITEM_BEGIN            '{'
#define CHAR_ITEM_END              '}'
#define CHAR_SGSERIAL_BEGIN        '['
#define CHAR_SGSERIAL_END          ']'

#define CHAR_TYPE_END              '\n'

#define STR_EMPTY_PREFIX           ":"

#define CHAR_SGSERIAL_NULL         'n'
#define STR_SGSERIAL_NULL          "ull"

using namespace shogun;

CSerializableAsciiFile::CSerializableAsciiFile(void)
	:CSerializableFile() { init(); }

CSerializableAsciiFile::CSerializableAsciiFile(FILE* f, char rw)
	:CSerializableFile(f, rw) { init(); }

CSerializableAsciiFile::CSerializableAsciiFile(char* fname, char rw)
	:CSerializableFile(fname, rw) { init(); }

CSerializableAsciiFile::~CSerializableAsciiFile() {}

void
CSerializableAsciiFile::init(void)
{
	if (file == NULL) return;

	switch (task) {
	case 'w':
		if (fprintf(file, STR_HEADER"\n") <= 0) {
			close(); return;
		}
		break;
	case 'r':
		char buf[60];
		if (fscanf(file, "%60s\n", buf) != 1
			|| strcmp(STR_HEADER, buf) != 0) {
			SG_WARNING("`%s' is not an serializable ascii file!\n",
					   filename);
			close(); return;
		}
		break;
	default:
		break;
	}

	stack_fpos.push_back(ftell(file));
}

bool
CSerializableAsciiFile::ignore(void)
{
	for (uint32_t cont_count = 0, item_count = 0,
			 sgserial_count = 0; ;) {
		switch (fgetc(file)) {
		case CHAR_ITEM_BEGIN: item_count++; break;
		case CHAR_CONT_BEGIN: cont_count++; break;
		case CHAR_SGSERIAL_BEGIN: sgserial_count++; break;
		case CHAR_CONT_END:
			if (cont_count-- == 0) return false;
			break;
		case CHAR_ITEM_END:
			if (item_count-- == 0) return false;
			break;
		case CHAR_SGSERIAL_END:
			if (sgserial_count-- == 0) return false;
			break;
		case CHAR_TYPE_END:
			if (cont_count == 0 && item_count == 0
				&& sgserial_count == 0)
				return true;
			break;
		case EOF: return false;
		default: break;
		}
	}

	return false;
}

bool
CSerializableAsciiFile::write_scalar_wrapped(
	const TSGDataType* type, const void* param)
{
	switch (type->m_ptype) {
	case PT_BOOL:
		if (fprintf(file, "%c", *(bool*) param? 't': 'f') <= 0)
			return false;
		break;
	case PT_CHAR:
		if (fprintf(file, "%"PRIu8, *(uint8_t*) param) <= 0)
			return false;
		break;
	case PT_INT8:
		if (fprintf(file, "%"PRIi8, *(int8_t*) param) <= 0)
			return false;
		break;
	case PT_UINT8:
		if (fprintf(file, "%"PRIu8, *(uint8_t*) param) <= 0)
			return false;
		break;
	case PT_INT16:
		if (fprintf(file, "%"PRIi16, *(int16_t*) param) <= 0)
			return false;
		break;
	case PT_UINT16:
		if (fprintf(file, "%"PRIu16, *(uint16_t*) param) <= 0)
			return false;
		break;
	case PT_INT32:
		if (fprintf(file, "%"PRIi32, *(int32_t*) param) <= 0)
			return false;
		break;
	case PT_UINT32:
		if (fprintf(file, "%"PRIu32, *(uint32_t*) param) <= 0)
			return false;
		break;
	case PT_INT64:
		if (fprintf(file, "%"PRIi64, *(int64_t*) param) <= 0)
			return false;
		break;
	case PT_UINT64:
		if (fprintf(file, "%"PRIu64, *(uint64_t*) param) <= 0)
			return false;
		break;
	case PT_FLOAT32:
		if (fprintf(file, "%+10.16e", *(float32_t*) param) <= 0)
			return false;
		break;
	case PT_FLOAT64:
		if (fprintf(file, "%+10.16e", *(float64_t*) param) <= 0)
			return false;
		break;
	case PT_FLOATMAX:
		if (fprintf(file, "%+10.16Le", *(floatmax_t*) param) <= 0)
			return false;
		break;
	case PT_SGSERIALIZABLE_PTR:
		SG_ERROR("write_scalar_wrapped(): Implementation error during"
				 " writing AsciiFile!");
		return false;
	}

	return true;
}

bool
CSerializableAsciiFile::read_scalar_wrapped(
	const TSGDataType* type, void* param)
{
	switch (type->m_ptype) {
	case PT_BOOL:
		char bool_buf;

		if (fscanf(file, "%c", &bool_buf) != 1) return false;

		switch (bool_buf) {
		case 't':
			*(bool*) param = true;
			break;
		case 'f':
			*(bool*) param = false;
			break;
		default:
			return false;
		}

		break;
	case PT_CHAR:
		if (fscanf(file, "%"SCNu8, (uint8_t*) param) != 1)
			return false;
		break;
	case PT_INT8:
		if (fscanf(file, "%"SCNi8, (int8_t*) param) != 1)
			return false;
		break;
	case PT_UINT8:
		if (fscanf(file, "%"SCNu8, (uint8_t*) param) != 1)
			return false;
		break;
	case PT_INT16:
		if (fscanf(file, "%"SCNi16, (int16_t*) param) != 1)
			return false;
		break;
	case PT_UINT16:
		if (fscanf(file, "%"SCNu16, (uint16_t*) param) != 1)
			return false;
		break;
	case PT_INT32:
		if (fscanf(file, "%"SCNi32, (int32_t*) param) != 1)
			return false;
		break;
	case PT_UINT32:
		if (fscanf(file, "%"SCNu32, (uint32_t*) param) != 1)
			return false;
		break;
	case PT_INT64:
		if (fscanf(file, "%"SCNi64, (int64_t*) param) != 1)
			return false;
		break;
	case PT_UINT64:
		if (fscanf(file, "%"SCNu64, (uint64_t*) param) != 1)
			return false;
		break;
	case PT_FLOAT32:
		if (fscanf(file, "%e", (float32_t*) param) != 1)
			return false;
		break;
	case PT_FLOAT64:
		if (fscanf(file, "%le", (float64_t*) param) != 1)
			return false;
		break;
	case PT_FLOATMAX:
		if (fscanf(file, "%Le", (floatmax_t*) param) != 1)
			return false;
		break;
	case PT_SGSERIALIZABLE_PTR:
		SG_ERROR("read_scalar_wrapped(): Implementation error during"
				 " writing AsciiFile!");
		return false;
	}

	return true;
}

bool
CSerializableAsciiFile::write_cont_begin_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	switch (type->m_ctype) {
	case CT_SCALAR:
		SG_ERROR("write_cont_begin_wrapped(): Implementation error "
				 "during writing AsciiFile!");
		return false;
	case CT_VECTOR:
		if (fprintf(file, "%"PRIi32" %c", len_real_y, CHAR_CONT_BEGIN)
			<= 0)
			return false;
		break;
	case CT_MATRIX:
		if (fprintf(file, "%"PRIi32" %"PRIi32" %c",
					len_real_y, len_real_x, CHAR_CONT_BEGIN) <= 0)
			return false;
		break;
	}

	return true;
}

bool
CSerializableAsciiFile::read_cont_begin_wrapped(
	const TSGDataType* type, index_t* len_read_y, index_t* len_read_x)
{
	switch (type->m_ctype) {
	case CT_SCALAR:
		SG_ERROR("read_cont_begin_wrapped(): Implementation error "
				 "during writing AsciiFile!");
		return false;
	case CT_VECTOR:
		if (fscanf(file, "%"SCNi32" ", len_read_y) != 1)
			return false;
		*len_read_x = 1;
		break;
	case CT_MATRIX:
		if (fscanf(file, "%"SCNi32" %"SCNi32" ",
				   len_read_y, len_read_x) <= 0)
			return false;
		break;
	}

	if (fgetc(file) != CHAR_CONT_BEGIN) return false;

	return true;
}

bool
CSerializableAsciiFile::write_cont_end_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	if (fprintf(file, "%c", CHAR_CONT_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_cont_end_wrapped(
	const TSGDataType* type, index_t len_read_y, index_t len_read_x)
{
	if (fgetc(file) != CHAR_CONT_END) return false;

	return true;
}

bool
CSerializableAsciiFile::write_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fprintf(file, "%c", CHAR_ITEM_BEGIN) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fgetc(file) != CHAR_ITEM_BEGIN) return false;

	return true;
}

bool
CSerializableAsciiFile::write_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fprintf(file, "%c", CHAR_ITEM_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fgetc(file) != CHAR_ITEM_END) return false;

	return true;
}

bool
CSerializableAsciiFile::write_sgserializable_begin_wrapped(
	const TSGDataType* type, bool is_null)
{
	if (is_null) {
		if (fprintf(file, "%c%s", CHAR_SGSERIAL_NULL,
					STR_SGSERIAL_NULL) <= 0)
			return false;
	} else {
		if (fprintf(file, "%c%c", CHAR_SGSERIAL_BEGIN, CHAR_TYPE_END)
			<= 0)
			return false;
	}

	return true;
}

bool
CSerializableAsciiFile::read_sgserializable_begin_wrapped(
	const TSGDataType* type, bool* is_null)
{
	switch (fgetc(file)) {
	case CHAR_SGSERIAL_BEGIN:
		if (fgetc(file) != CHAR_TYPE_END) return false;
		*is_null = false;
		break;
	case CHAR_SGSERIAL_NULL:
		char buf[10];
		if (fscanf(file, "%10s", buf) != 1
			|| strcmp(buf, STR_SGSERIAL_NULL) != 0) return false;
		*is_null = true;
		break;
	default:
		return false;
	}

	stack_fpos.push_back(ftell(file));

	return true;
}

bool
CSerializableAsciiFile::write_sgserializable_end_wrapped(
	const TSGDataType* type, bool is_null)
{
	if (!is_null)
		if (fprintf(file, "%c", CHAR_SGSERIAL_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_sgserializable_end_wrapped(
	const TSGDataType* type, bool is_null)
{
	if (!is_null)
		if (fgetc(file) != CHAR_SGSERIAL_END) return false;

	stack_fpos.pop_back();

	return true;
}

bool
CSerializableAsciiFile::write_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	char buf[50];
	type->to_string(buf);

	if (fprintf(file, "%s %s %s ",
				*prefix == '\0'? STR_EMPTY_PREFIX: prefix, name, buf
			) <= 0)
		return false;

	return true;
}

bool
CSerializableAsciiFile::read_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (fseek(file, stack_fpos.back(), SEEK_SET) != 0) return false;

	char type_str[50];
	type->to_string(type_str);

	char r_prefix[256], r_name[50], r_type[50];
	while (true) {
		if (fscanf(file, "%256s %50s %50s ", r_prefix, r_name, r_type)
			!= 3)
			return false;

		if ((strcmp(r_prefix, prefix) == 0 || (
				 *prefix == '\0'
				 && strcmp(r_prefix, STR_EMPTY_PREFIX) == 0))
			&& strcmp(r_name, name) == 0
			&& strcmp(r_type, type_str) == 0)
			return true;

		if (!ignore()) return false;
	}

	return false;
}

bool
CSerializableAsciiFile::write_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (fprintf(file, "%c", CHAR_TYPE_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (fgetc(file) != CHAR_TYPE_END) return false;

	return true;
}
