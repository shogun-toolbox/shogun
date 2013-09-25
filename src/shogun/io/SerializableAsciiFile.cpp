/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/io/SerializableAsciiReader00.h>

#define STR_HEADER_00                 \
	"<<_SHOGUN_SERIALIZABLE_ASCII_FILE_V_00_>>"

using namespace shogun;

CSerializableAsciiFile::CSerializableAsciiFile()
	:CSerializableFile() { init(); }

CSerializableAsciiFile::CSerializableAsciiFile(FILE* fstream, char rw)
	:CSerializableFile(fstream, rw) { init(); }

CSerializableAsciiFile::CSerializableAsciiFile(
	const char* fname, char rw)
	:CSerializableFile(fname, rw) { init(); }

CSerializableAsciiFile::~CSerializableAsciiFile() {}

bool
CSerializableAsciiFile::ignore()
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

CSerializableFile::TSerializableReader*
CSerializableAsciiFile::new_reader(char* dest_version, size_t n)
{
	string_t buf;
	if (fscanf(m_fstream, "%" STRING_LEN_STR"s\n", buf) != 1)
		return NULL;

	strncpy(dest_version, buf, n < STRING_LEN? n: STRING_LEN);
	m_stack_fpos.push_back(ftell(m_fstream));

	if (strcmp(STR_HEADER_00, dest_version) == 0)
		return new SerializableAsciiReader00(this);

	return NULL;
}

void
CSerializableAsciiFile::init()
{
	if (m_fstream == NULL) return;

	switch (m_task) {
	case 'w':
		if (fprintf(m_fstream, STR_HEADER_00"\n") <= 0) {
			close(); return;
		}
		m_stack_fpos.push_back(ftell(m_fstream));
		break;
	case 'r': break;
	default:
		SG_WARNING("Could not open file `%s', unknown mode!\n",
				   m_filename);
		close(); return;
	}
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
		if (fprintf(m_fstream, "%" PRIu8, *(uint8_t*) param) <= 0)
			return false;
		break;
	case PT_INT8:
		if (fprintf(m_fstream, "%" PRIi8, *(int8_t*) param) <= 0)
			return false;
		break;
	case PT_UINT8:
		if (fprintf(m_fstream, "%" PRIu8, *(uint8_t*) param) <= 0)
			return false;
		break;
	case PT_INT16:
		if (fprintf(m_fstream, "%" PRIi16, *(int16_t*) param) <= 0)
			return false;
		break;
	case PT_UINT16:
		if (fprintf(m_fstream, "%" PRIu16, *(uint16_t*) param) <= 0)
			return false;
		break;
	case PT_INT32:
		if (fprintf(m_fstream, "%" PRIi32, *(int32_t*) param) <= 0)
			return false;
		break;
	case PT_UINT32:
		if (fprintf(m_fstream, "%" PRIu32, *(uint32_t*) param) <= 0)
			return false;
		break;
	case PT_INT64:
		if (fprintf(m_fstream, "%" PRIi64, *(int64_t*) param) <= 0)
			return false;
		break;
	case PT_UINT64:
		if (fprintf(m_fstream, "%" PRIu64, *(uint64_t*) param) <= 0)
			return false;
		break;
	case PT_FLOAT32:
		if (fprintf(m_fstream, "%.16g", *(float32_t*) param) <= 0)
			return false;
		break;
	case PT_FLOAT64:
		if (fprintf(m_fstream, "%.16lg", *(float64_t*) param) <= 0)
			return false;
		break;
	case PT_FLOATMAX:
		if (fprintf(m_fstream, "%.16Lg", *(floatmax_t*) param) <= 0)
			return false;
		break;
	case PT_COMPLEX128:
		if (fprintf(m_fstream, "(%.16lg,%.16lg)",
			((complex128_t*) param)->real(),((complex128_t*) param)->imag()) <= 0)
			return false;
		break;
	case PT_SGOBJECT:
		SG_ERROR("write_scalar_wrapped(): Implementation error during"
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
	case CT_NDARRAY:
		SG_NOTIMPLEMENTED
		break;
	case CT_SCALAR:
		SG_ERROR("write_cont_begin_wrapped(): Implementation error "
				 "during writing AsciiFile!");
		return false;
	case CT_VECTOR: case CT_SGVECTOR:
		if (fprintf(m_fstream, "%" PRIi32 " %c", len_real_y,
					CHAR_CONT_BEGIN) <= 0)
			return false;
		break;
	case CT_MATRIX: case CT_SGMATRIX:
		if (fprintf(m_fstream, "%" PRIi32" %" PRIi32 " %c",
					len_real_y, len_real_x, CHAR_CONT_BEGIN) <= 0)
			return false;
		break;
	}

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
CSerializableAsciiFile::write_string_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	if (fprintf(m_fstream, "%" PRIi32 " %c", length,
				CHAR_STRING_BEGIN) <= 0) return false;

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
CSerializableAsciiFile::write_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	if (fprintf(m_fstream, "%c", CHAR_ITEM_BEGIN) <= 0) return false;

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
CSerializableAsciiFile::write_sparse_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	if (fprintf(m_fstream, "%" PRIi32" %c", length,
				CHAR_SPARSE_BEGIN) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::write_sparse_end_wrapped(
	const TSGDataType* type, index_t length)
{
	if (fprintf(m_fstream, "%c", CHAR_SPARSE_END) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::write_sparseentry_begin_wrapped(
	const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	if (fprintf(m_fstream, " %" PRIi32 " %c", feat_index, CHAR_ITEM_BEGIN)
		<= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::write_sparseentry_end_wrapped(
	const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	if (fprintf(m_fstream, "%c", CHAR_ITEM_END) <= 0) return false;

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
CSerializableAsciiFile::write_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (fprintf(m_fstream, "%c", CHAR_ITEM_END) <= 0) return false;

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
CSerializableAsciiFile::write_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (fprintf(m_fstream, "%c", CHAR_SGSERIAL_END) <= 0)
		return false;

	return true;
}

bool
CSerializableAsciiFile::write_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	string_t buf;
	type->to_string(buf, STRING_LEN);

	SG_SET_LOCALE_C;

	if (fprintf(m_fstream, "%s %s ", name, buf) <= 0) return false;

	return true;
}

bool
CSerializableAsciiFile::write_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (fprintf(m_fstream, "%c", CHAR_TYPE_END) <= 0) return false;

	SG_RESET_LOCALE;

	return true;
}
