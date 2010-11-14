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
	"<<_SHOGUN_SERIALIZABLE_ASCII_FILE_V_00_>>"

#define CHAR_CONT_BEGIN            '('
#define CHAR_CONT_END              ')'
#define CHAR_ITEM_BEGIN            '{'
#define CHAR_ITEM_END              '}'
#define CHAR_SGSERIAL_BEGIN        '['
#define CHAR_SGSERIAL_END          ']'
#define CHAR_STRING_BEGIN          CHAR_SGSERIAL_BEGIN
#define CHAR_STRING_END            CHAR_SGSERIAL_END
#define CHAR_SPARSE_BEGIN          CHAR_CONT_BEGIN
#define CHAR_SPARSE_END            CHAR_CONT_END

#define CHAR_TYPE_END              '\n'

#define STR_EMPTY_PREFIX           ":"

#define STR_SGSERIAL_NULL          "null"

using namespace shogun;

CSerializableAsciiFile::CSerializableAsciiFile(void)
	:CSerializableFile() { init(); }

CSerializableAsciiFile::CSerializableAsciiFile(FILE* fstream, char rw)
	:CSerializableFile(fstream, rw) { init(); }

CSerializableAsciiFile::CSerializableAsciiFile(
	const char* fname, char rw)
	:CSerializableFile(fname, rw) { init(); }

CSerializableAsciiFile::~CSerializableAsciiFile() {}

void
CSerializableAsciiFile::init(void)
{
	if (m_fstream == NULL) return;

	switch (m_task) {
	case 'w':
		if (fprintf(m_fstream, STR_HEADER"\n") <= 0) {
			close(); return;
		}
		break;
	case 'r':
		string_t buf;
		if (fscanf(m_fstream, "%"STRING_LEN_STR"s\n", buf) != 1
			|| strcmp(STR_HEADER, buf) != 0) {
			SG_WARNING("`%s' is not an serializable ascii file!\n",
					   m_filename);
			close(); return;
		}
		break;
	default:
		break;
	}

	m_stack_fpos.push_back(ftell(m_fstream));
}

bool
CSerializableAsciiFile::ignore(void)
{
	for (uint32_t cont_count = 0, item_count = 0,
			 sgserial_count = 0; ;) {
		switch (fgetc(m_fstream)) {
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
		if (fprintf(m_fstream, "%c", *(bool*) param? 't': 'f') <= 0)
			return false;
		break;
	case PT_CHAR:
		if (fprintf(m_fstream, "%"PRIu8, *(uint8_t*) param) <= 0)
			return false;
		break;
	case PT_INT8:
		if (fprintf(m_fstream, "%"PRIi8, *(int8_t*) param) <= 0)
			return false;
		break;
	case PT_UINT8:
		if (fprintf(m_fstream, "%"PRIu8, *(uint8_t*) param) <= 0)
			return false;
		break;
	case PT_INT16:
		if (fprintf(m_fstream, "%"PRIi16, *(int16_t*) param) <= 0)
			return false;
		break;
	case PT_UINT16:
		if (fprintf(m_fstream, "%"PRIu16, *(uint16_t*) param) <= 0)
			return false;
		break;
	case PT_INT32:
		if (fprintf(m_fstream, "%"PRIi32, *(int32_t*) param) <= 0)
			return false;
		break;
	case PT_UINT32:
		if (fprintf(m_fstream, "%"PRIu32, *(uint32_t*) param) <= 0)
			return false;
		break;
	case PT_INT64:
		if (fprintf(m_fstream, "%"PRIi64, *(int64_t*) param) <= 0)
			return false;
		break;
	case PT_UINT64:
		if (fprintf(m_fstream, "%"PRIu64, *(uint64_t*) param) <= 0)
			return false;
		break;
	case PT_FLOAT32:
		if (fprintf(m_fstream, "%+10.16e", *(float32_t*) param) <= 0)
			return false;
		break;
	case PT_FLOAT64:
		if (fprintf(m_fstream, "%+10.16e", *(float64_t*) param) <= 0)
			return false;
		break;
	case PT_FLOATMAX:
		if (fprintf(m_fstream, "%+10.16Le", *(floatmax_t*) param) <= 0)
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

		if (fscanf(m_fstream, "%c", &bool_buf) != 1) return false;

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
		if (fscanf(m_fstream, "%"SCNu8, (uint8_t*) param) != 1)
			return false;
		break;
	case PT_INT8:
		if (fscanf(m_fstream, "%"SCNi8, (int8_t*) param) != 1)
			return false;
		break;
	case PT_UINT8:
		if (fscanf(m_fstream, "%"SCNu8, (uint8_t*) param) != 1)
			return false;
		break;
	case PT_INT16:
		if (fscanf(m_fstream, "%"SCNi16, (int16_t*) param) != 1)
			return false;
		break;
	case PT_UINT16:
		if (fscanf(m_fstream, "%"SCNu16, (uint16_t*) param) != 1)
			return false;
		break;
	case PT_INT32:
		if (fscanf(m_fstream, "%"SCNi32, (int32_t*) param) != 1)
			return false;
		break;
	case PT_UINT32:
		if (fscanf(m_fstream, "%"SCNu32, (uint32_t*) param) != 1)
			return false;
		break;
	case PT_INT64:
		if (fscanf(m_fstream, "%"SCNi64, (int64_t*) param) != 1)
			return false;
		break;
	case PT_UINT64:
		if (fscanf(m_fstream, "%"SCNu64, (uint64_t*) param) != 1)
			return false;
		break;
	case PT_FLOAT32:
		if (fscanf(m_fstream, "%e", (float32_t*) param) != 1)
			return false;
		break;
	case PT_FLOAT64:
		if (fscanf(m_fstream, "%le", (float64_t*) param) != 1)
			return false;
		break;
	case PT_FLOATMAX:
		if (fscanf(m_fstream, "%Le", (floatmax_t*) param) != 1)
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
		if (fprintf(m_fstream, "%"PRIi32" %c", len_real_y,
					CHAR_CONT_BEGIN) <= 0)
			return false;
		break;
	case CT_MATRIX:
		if (fprintf(m_fstream, "%"PRIi32" %"PRIi32" %c",
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
		if (fscanf(m_fstream, "%"SCNi32" ", len_read_y) != 1)
			return false;
		*len_read_x = 1;
		break;
	case CT_MATRIX:
		if (fscanf(m_fstream, "%"SCNi32" %"SCNi32" ",
				   len_read_y, len_read_x) != 2)
			return false;
		break;
	}

	if (fgetc(m_fstream) != CHAR_CONT_BEGIN) return false;

	return true;
}

bool
CSerializableAsciiFile::write_cont_end_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	if (fprintf(m_fstream, "%c", CHAR_CONT_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_cont_end_wrapped(
	const TSGDataType* type, index_t len_read_y, index_t len_read_x)
{
	if (fgetc(m_fstream) != CHAR_CONT_END) return false;

	return true;
}

bool
CSerializableAsciiFile::write_string_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	if (fprintf(m_fstream, "%"PRIi32" %c", length,
				CHAR_STRING_BEGIN) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_string_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	if (fscanf(m_fstream, "%"PRIi32, length) != 1) return false;
	if (fgetc(m_fstream) != ' ') return false;
	if (fgetc(m_fstream) != CHAR_STRING_BEGIN) return false;

	return true;
}

bool
CSerializableAsciiFile::write_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	if (fprintf(m_fstream, "%c", CHAR_STRING_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	if (fgetc(m_fstream) != CHAR_STRING_END) return false;

	return true;
}

bool
CSerializableAsciiFile::write_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	if (fprintf(m_fstream, "%c", CHAR_ITEM_BEGIN) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	if (fgetc(m_fstream) != CHAR_ITEM_BEGIN) return false;

	return true;
}

bool
CSerializableAsciiFile::write_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	if (fprintf(m_fstream, "%c", CHAR_ITEM_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	if (fgetc(m_fstream) != CHAR_ITEM_END) return false;

	return true;
}

bool
CSerializableAsciiFile::write_sparse_begin_wrapped(
	const TSGDataType* type, index_t vec_index,
	index_t length)
{
	if (fprintf(m_fstream, "%"PRIi32" %"PRIi32" %c", vec_index, length,
				CHAR_SPARSE_BEGIN) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_sparse_begin_wrapped(
	const TSGDataType* type, index_t* vec_index,
	index_t* length)
{
	if (fscanf(m_fstream, "%"PRIi32" %"PRIi32, vec_index, length)
		!= 2) return false;
	if (fgetc(m_fstream) != ' ') return false;
	if (fgetc(m_fstream) != CHAR_SPARSE_BEGIN) return false;

	return true;
}

bool
CSerializableAsciiFile::write_sparse_end_wrapped(
	const TSGDataType* type, index_t vec_index,
	index_t length)
{
	if (fprintf(m_fstream, "%c", CHAR_SPARSE_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_sparse_end_wrapped(
	const TSGDataType* type, index_t* vec_index,
	index_t length)
{
	if (fgetc(m_fstream) != CHAR_SPARSE_END) return false;

	return true;
}

bool
CSerializableAsciiFile::write_sparseentry_begin_wrapped(
	const TSGDataType* type, index_t feat_index, index_t y)
{
	if (fprintf(m_fstream, " %"PRIi32" %c", feat_index, CHAR_ITEM_BEGIN)
		<= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_sparseentry_begin_wrapped(
	const TSGDataType* type, index_t* feat_index, index_t y)
{
	if (fscanf(m_fstream, "%"PRIi32, feat_index) != 1) return false;
	if (fgetc(m_fstream) != ' ') return false;
	if (fgetc(m_fstream) != CHAR_ITEM_BEGIN) return false;

	return true;
}

bool
CSerializableAsciiFile::write_sparseentry_end_wrapped(
	const TSGDataType* type, index_t feat_index, index_t y)
{
	if (fprintf(m_fstream, "%c", CHAR_ITEM_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_sparseentry_end_wrapped(
	const TSGDataType* type, index_t* feat_index, index_t y)
{
	if (fgetc(m_fstream) != CHAR_ITEM_END) return false;

	return true;
}

bool
CSerializableAsciiFile::write_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fprintf(m_fstream, "%c", CHAR_ITEM_BEGIN) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fgetc(m_fstream) != CHAR_ITEM_BEGIN) return false;

	return true;
}

bool
CSerializableAsciiFile::write_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fprintf(m_fstream, "%c", CHAR_ITEM_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fgetc(m_fstream) != CHAR_ITEM_END) return false;

	return true;
}

bool
CSerializableAsciiFile::write_sgserializable_begin_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (*sgserializable_name == '\0') {
		if (fprintf(m_fstream, "%s %c", STR_SGSERIAL_NULL,
					CHAR_SGSERIAL_BEGIN) <= 0)
			return false;
	} else {
		if (fprintf(m_fstream, "%s ", sgserializable_name) <= 0)
			return false;

		if (generic != PT_NOT_GENERIC) {
			string_t buf;
			TSGDataType::ptype_to_string(buf, generic, STRING_LEN);
			if (fprintf(m_fstream, "%s ", buf) <= 0) return false;
		}

		if (fprintf(m_fstream, "%c%c", CHAR_SGSERIAL_BEGIN,
					CHAR_TYPE_END) <= 0)
			return false;
	}

	return true;
}

bool
CSerializableAsciiFile::read_sgserializable_begin_wrapped(
	const TSGDataType* type, char* sgserializable_name,
	EPrimitiveType* generic)
{
	if (fscanf(m_fstream, "%"STRING_LEN_STR"s ", sgserializable_name)
		!= 1) return false;

	if (strcmp(sgserializable_name, STR_SGSERIAL_NULL) == 0) {
		if (fgetc(m_fstream) != CHAR_SGSERIAL_BEGIN) return false;

		*sgserializable_name = '\0';
	} else {
		string_t buf;
		if (fscanf(m_fstream, "%"STRING_LEN_STR"s ", buf) != 1)
			return false;

		if (buf[0] != CHAR_SGSERIAL_BEGIN) {
			if (!TSGDataType::string_to_ptype(generic, buf))
				return false;

			if (fgetc(m_fstream) != CHAR_SGSERIAL_BEGIN) return false;
			if (fgetc(m_fstream) != CHAR_TYPE_END) return false;
		}
	}

	m_stack_fpos.push_back(ftell(m_fstream));

	return true;
}

bool
CSerializableAsciiFile::write_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (fprintf(m_fstream, "%c", CHAR_SGSERIAL_END) <= 0)
		return false;

	return true;
}

bool
CSerializableAsciiFile::read_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (fgetc(m_fstream) != CHAR_SGSERIAL_END) return false;

	m_stack_fpos.pop_back();

	return true;
}

bool
CSerializableAsciiFile::write_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	string_t buf;
	type->to_string(buf, STRING_LEN);

	if (fprintf(m_fstream, "%s %s %s ",
				*prefix == '\0'? STR_EMPTY_PREFIX: prefix, name, buf
			) <= 0)
		return false;

	return true;
}

bool
CSerializableAsciiFile::read_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (fseek(m_fstream, m_stack_fpos.back(), SEEK_SET) != 0)
		return false;

	string_t type_str;
	type->to_string(type_str, STRING_LEN);

	string_t r_prefix, r_name, r_type;
	while (true) {
		if (fscanf(m_fstream, "%"STRING_LEN_STR"s %"STRING_LEN_STR
				   "s %"STRING_LEN_STR"s ", r_prefix, r_name,
				   r_type) != 3)
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
	if (fprintf(m_fstream, "%c", CHAR_TYPE_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::read_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (fgetc(m_fstream) != CHAR_TYPE_END) return false;

	return true;
}
