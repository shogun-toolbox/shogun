/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/io/SerializableAsciiReader00.h>
#include <shogun/lib/common.h>

using namespace shogun;

SerializableAsciiReader00::SerializableAsciiReader00(
	CSerializableAsciiFile* file) { m_file = file; }

SerializableAsciiReader00::~SerializableAsciiReader00() {}

bool
SerializableAsciiReader00::read_scalar_wrapped(
	const TSGDataType* type, void* param)
{
	switch (type->m_ptype) {
	case PT_BOOL:
		char bool_buf;

		if (fscanf(m_file->m_fstream, "%c", &bool_buf) != 1)
			return false;

		switch (bool_buf) {
		case 't': *(bool*) param = true; break;
		case 'f': *(bool*) param = false; break;
		default: return false;
		}

		break;
	case PT_CHAR:
		if (fscanf(m_file->m_fstream, "%" SCNu8, (uint8_t*) param)
			!= 1) return false;
		break;
	case PT_INT8:
		if (fscanf(m_file->m_fstream, "%" SCNi8, (int8_t*) param)
			!= 1) return false;
		break;
	case PT_UINT8:
		if (fscanf(m_file->m_fstream, "%" SCNu8, (uint8_t*) param)
			!= 1) return false;
		break;
	case PT_INT16:
		if (fscanf(m_file->m_fstream, "%" SCNi16, (int16_t*) param)
			!= 1) return false;
		break;
	case PT_UINT16:
		if (fscanf(m_file->m_fstream, "%" SCNu16, (uint16_t*) param)
			!= 1) return false;
		break;
	case PT_INT32:
		if (fscanf(m_file->m_fstream, "%" SCNi32, (int32_t*) param)
			!= 1) return false;
		break;
	case PT_UINT32:
		if (fscanf(m_file->m_fstream, "%" SCNu32, (uint32_t*) param)
			!= 1) return false;
		break;
	case PT_INT64:
		if (fscanf(m_file->m_fstream, "%" SCNi64, (int64_t*) param)
			!= 1) return false;
		break;
	case PT_UINT64:
		if (fscanf(m_file->m_fstream, "%" SCNu64, (uint64_t*) param)
			!= 1) return false;
		break;
	case PT_FLOAT32:
		if (fscanf(m_file->m_fstream, "%g", (float32_t*) param)
			!= 1) return false;
		break;
	case PT_FLOAT64:
		if (fscanf(m_file->m_fstream, "%lg", (float64_t*) param)
			!= 1) return false;
		break;
	case PT_FLOATMAX:
		if (fscanf(m_file->m_fstream, "%Lg", (floatmax_t*) param)
			!= 1) return false;
		break;
	case PT_COMPLEX128:
		float64_t c_real, c_imag;
		if (fscanf(m_file->m_fstream, "(%lg,%lg)", &c_real, &c_imag) 
			!= 2) return false;
#ifdef HAVE_CXX11
		((complex128_t*) param)->real(c_real);
		((complex128_t*) param)->imag(c_imag);
#else
		((complex128_t*) param)->real()=c_real;
		((complex128_t*) param)->imag()=c_imag;
#endif
		break;
	case PT_SGOBJECT:
		SG_ERROR("read_scalar_wrapped(): Implementation error during"
				 " reading AsciiFile!");
		return false;
	}

	return true;
}

bool
SerializableAsciiReader00::read_cont_begin_wrapped(
	const TSGDataType* type, index_t* len_read_y, index_t* len_read_x)
{
	switch (type->m_ctype) {
	case CT_NDARRAY:
		SG_NOTIMPLEMENTED
	case CT_SCALAR:
		SG_ERROR("read_cont_begin_wrapped(): Implementation error "
				 "during writing AsciiFile!");
		return false;
	case CT_VECTOR: case CT_SGVECTOR:
		if (fscanf(m_file->m_fstream, "%" SCNi32 " ", len_read_y) != 1)
			return false;
		*len_read_x = 1;
		break;
	case CT_MATRIX: case CT_SGMATRIX:
		if (fscanf(m_file->m_fstream, "%" SCNi32 " %" SCNi32 " ",
				   len_read_y, len_read_x) != 2)
			return false;
		break;
	}

	if (fgetc(m_file->m_fstream) != CHAR_CONT_BEGIN) return false;

	return true;
}

bool
SerializableAsciiReader00::read_cont_end_wrapped(
	const TSGDataType* type, index_t len_read_y, index_t len_read_x)
{
	if (fgetc(m_file->m_fstream) != CHAR_CONT_END) return false;

	return true;
}

bool
SerializableAsciiReader00::read_string_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	if (fscanf(m_file->m_fstream, "%" PRIi32, length) != 1)
		return false;
	if (fgetc(m_file->m_fstream) != ' ') return false;
	if (fgetc(m_file->m_fstream) != CHAR_STRING_BEGIN) return false;

	return true;
}

bool
SerializableAsciiReader00::read_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	if (fgetc(m_file->m_fstream) != CHAR_STRING_END) return false;

	return true;
}

bool
SerializableAsciiReader00::read_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	if (fgetc(m_file->m_fstream) != CHAR_ITEM_BEGIN) return false;

	return true;
}

bool
SerializableAsciiReader00::read_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	if (fgetc(m_file->m_fstream) != CHAR_ITEM_END) return false;

	return true;
}

bool
SerializableAsciiReader00::read_sparse_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	if (fscanf(m_file->m_fstream, "%" PRIi32, length) != 1) return false;
	if (fgetc(m_file->m_fstream) != ' ') return false;
	if (fgetc(m_file->m_fstream) != CHAR_SPARSE_BEGIN) return false;

	return true;
}

bool
SerializableAsciiReader00::read_sparse_end_wrapped(
	const TSGDataType* type, index_t length)
{
	if (fgetc(m_file->m_fstream) != CHAR_SPARSE_END) return false;

	return true;
}

bool
SerializableAsciiReader00::read_sparseentry_begin_wrapped(
	const TSGDataType* type, SGSparseVectorEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	if (fscanf(m_file->m_fstream, "%" PRIi32, feat_index) != 1)
		return false;
	if (fgetc(m_file->m_fstream) != ' ') return false;
	if (fgetc(m_file->m_fstream) != CHAR_ITEM_BEGIN) return false;

	return true;
}

bool
SerializableAsciiReader00::read_sparseentry_end_wrapped(
	const TSGDataType* type, SGSparseVectorEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	if (fgetc(m_file->m_fstream) != CHAR_ITEM_END) return false;

	return true;
}

bool
SerializableAsciiReader00::read_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fgetc(m_file->m_fstream) != CHAR_ITEM_BEGIN) return false;

	return true;
}

bool
SerializableAsciiReader00::read_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fgetc(m_file->m_fstream) != CHAR_ITEM_END) return false;

	return true;
}

bool
SerializableAsciiReader00::read_sgserializable_begin_wrapped(
	const TSGDataType* type, char* sgserializable_name,
	EPrimitiveType* generic)
{
	if (fscanf(m_file->m_fstream, "%" STRING_LEN_STR "s ",
			   sgserializable_name) != 1) return false;

	if (strcmp(sgserializable_name, STR_SGSERIAL_NULL) == 0) {
		if (fgetc(m_file->m_fstream) != CHAR_SGSERIAL_BEGIN)
			return false;

		*sgserializable_name = '\0';
	} else {
		string_t buf;
		if (fscanf(m_file->m_fstream, "%" STRING_LEN_STR "s ", buf)
			!= 1) return false;

		if (buf[0] != CHAR_SGSERIAL_BEGIN) {
			if (!TSGDataType::string_to_ptype(generic, buf))
				return false;

			if (fgetc(m_file->m_fstream) != CHAR_SGSERIAL_BEGIN)
				return false;
			if (fgetc(m_file->m_fstream) != CHAR_TYPE_END)
				return false;
		}
	}

	m_file->m_stack_fpos.push_back(ftell(m_file->m_fstream));

	return true;
}

bool
SerializableAsciiReader00::read_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (fgetc(m_file->m_fstream) != CHAR_SGSERIAL_END) return false;

	m_file->m_stack_fpos.pop_back();

	return true;
}

bool
SerializableAsciiReader00::read_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (fseek(m_file->m_fstream, m_file->m_stack_fpos.back(), SEEK_SET
			) != 0) return false;

	SG_SET_LOCALE_C;

	string_t type_str;
	type->to_string(type_str, STRING_LEN);

	string_t r_name, r_type;
	while (true) {
		if (fscanf(m_file->m_fstream, "%" STRING_LEN_STR "s %"
				   STRING_LEN_STR "s ", r_name, r_type) != 2)
			return false;

		if (strcmp(r_name, name) == 0
			&& strcmp(r_type, type_str) == 0) return true;

		if (!m_file->ignore()) return false;
	}

	return false;
}

bool
SerializableAsciiReader00::read_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (fgetc(m_file->m_fstream) != CHAR_TYPE_END) return false;

	SG_RESET_LOCALE;

	return true;
}
