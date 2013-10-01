/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/lib/config.h>
#ifdef HAVE_JSON

#include <shogun/io/SerializableJsonReader00.h>

using namespace shogun;

SerializableJsonReader00::SerializableJsonReader00(
	CSerializableJsonFile* file) { m_file = file; }

SerializableJsonReader00::~SerializableJsonReader00()
{
}

bool
SerializableJsonReader00::read_scalar_wrapped(
	const TSGDataType* type, void* param)
{
	json_object* m = m_file->m_stack_stream.back();

	switch (type->m_ptype) {
	case PT_BOOL:
		if (!json_object_is_type(m, json_type_boolean)) return false;
		*(bool*) param = json_object_get_boolean(m);
		break;
	case PT_CHAR:
		if (!json_object_is_type(m, json_type_int)) return false;
		*(char*) param = json_object_get_int(m);
		break;
	case PT_INT8:
		if (!json_object_is_type(m, json_type_int)) return false;
		*(int8_t*) param = json_object_get_int(m);
		break;
	case PT_UINT8:
		if (!json_object_is_type(m, json_type_int)) return false;
		*(uint8_t*) param = json_object_get_int(m);
		break;
	case PT_INT16:
		if (!json_object_is_type(m, json_type_int)) return false;
		*(int16_t*) param = json_object_get_int(m);
		break;
	case PT_UINT16:
		if (!json_object_is_type(m, json_type_int)) return false;
		*(uint16_t*) param = json_object_get_int(m);
		break;
	case PT_INT32:
		if (!json_object_is_type(m, json_type_int)) return false;
		*(int32_t*) param = json_object_get_int(m);
		break;
	case PT_UINT32:
		if (!json_object_is_type(m, json_type_int)) return false;
		*(uint32_t*) param = json_object_get_int(m);
		break;
	case PT_INT64:
		if (!json_object_is_type(m, json_type_int)) return false;
		*(int64_t*) param = json_object_get_int(m);
		break;
	case PT_UINT64:
		if (!json_object_is_type(m, json_type_int)) return false;
		*(uint64_t*) param = json_object_get_int(m);
		break;
	case PT_FLOAT32:
		if (!json_object_is_type(m, json_type_double)) return false;
		*(float32_t*) param = json_object_get_double(m);
		break;
	case PT_FLOAT64:
		if (!json_object_is_type(m, json_type_double)) return false;
		*(float64_t*) param = json_object_get_double(m);
		break;
	case PT_FLOATMAX:
		if (!json_object_is_type(m, json_type_double)) return false;
		*(floatmax_t*) param = json_object_get_double(m);
		break;
	case PT_COMPLEX128:
		SG_ERROR("read_scalar_wrapped(): Not supported for complex128_t"
				 " for reading from JsonFile!");
		break;
	case PT_SGOBJECT:
	case PT_UNDEFINED:
		SG_ERROR("read_scalar_wrapped(): Implementation error during"
				 " reading JsonFile!");
		return false;
	}

	return true;
}

bool
SerializableJsonReader00::read_cont_begin_wrapped(
	const TSGDataType* type, index_t* len_read_y, index_t* len_read_x)
{
	json_object* m = m_file->m_stack_stream.back();

	if (!json_object_is_type(m, json_type_array)) return false;

	*len_read_y = json_object_array_length(m);

	if (type->m_ctype==CT_MATRIX || type->m_ctype==CT_SGMATRIX) {
		*len_read_x = *len_read_y;
		for (index_t i=0; i<*len_read_x; i++) {
			json_object* buf = json_object_array_get_idx(m, i);
			if (!json_object_is_type(buf, json_type_array))
				return false;

			index_t len = json_object_array_length(buf);
			if (i == 0) *len_read_y = len;
			else if (*len_read_y != len) return false;
		}
	}

	return true;
}

bool
SerializableJsonReader00::read_cont_end_wrapped(
	const TSGDataType* type, index_t len_read_y, index_t len_read_x)
{
	return true;
}

bool
SerializableJsonReader00::read_string_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	json_object* m = m_file->m_stack_stream.back();

	if (!json_object_is_type(m, json_type_array)) return false;

	*length = json_object_array_length(m);

	return true;
}

bool
SerializableJsonReader00::read_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
SerializableJsonReader00::read_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	json_object* m = m_file->m_stack_stream.back();

	json_object* buf = json_object_array_get_idx(m, y);
	if (is_error(buf)) return false;

	m_file->push_object(buf);
	return true;
}

bool
SerializableJsonReader00::read_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	m_file->pop_object();
	return true;
}

bool
SerializableJsonReader00::read_sparse_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	json_object* m = m_file->m_stack_stream.back();

	if (!json_object_is_type(m, json_type_object)) return false;

	json_object* buf;
	if (!m_file->get_object(&buf, m, STR_KEY_SPARSE_FEATURES,
							json_type_array)) return false;
	*length = json_object_array_length(buf);
	m_file->push_object(buf);

	return true;
}

bool
SerializableJsonReader00::read_sparse_end_wrapped(
	const TSGDataType* type, index_t length)
{
	m_file->pop_object();
	return true;
}

bool
SerializableJsonReader00::read_sparseentry_begin_wrapped(
	const TSGDataType* type, SGSparseVectorEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	json_object* m = m_file->m_stack_stream.back();

	json_object* buf_obj
		= json_object_array_get_idx(m, y);
	if (is_error(buf_obj)) return false;
	if (!json_object_is_type(buf_obj, json_type_object)) return false;

	json_object* buf;
	if (!m_file->get_object(&buf, buf_obj, STR_KEY_SPARSE_FEATINDEX,
							json_type_int)) return false;
	*feat_index = json_object_get_int(buf);

	if (!m_file->get_object_any(&buf, buf_obj, STR_KEY_SPARSE_ENTRY))
		return false;
	m_file->push_object(buf);

	return true;
}

bool
SerializableJsonReader00::read_sparseentry_end_wrapped(
	const TSGDataType* type, SGSparseVectorEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	m_file->pop_object();
	return true;
}

bool
SerializableJsonReader00::read_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	json_object* m = m_file->m_stack_stream.back();

	if (type->m_ctype==CT_MATRIX || type->m_ctype==CT_SGMATRIX)
		m = json_object_array_get_idx(m, x);
	m = json_object_array_get_idx(m, y);

	m_file->push_object(m);
	return true;
}

bool
SerializableJsonReader00::read_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	m_file->pop_object();
	return true;
}

bool
SerializableJsonReader00::read_sgserializable_begin_wrapped(
	const TSGDataType* type, char* sgserializable_name,
	EPrimitiveType* generic)
{
	json_object* m = m_file->m_stack_stream.back();

	if (m == NULL || json_object_is_type(m, json_type_null)) {
		*sgserializable_name = '\0'; return true;
	}

	if (!json_object_is_type(m, json_type_object)) return false;

	json_object* buf;
	if (!m_file->get_object(&buf, m, STR_KEY_INSTANCE_NAME,
							json_type_string)) return false;
	strncpy(sgserializable_name, json_object_get_string(buf),
			STRING_LEN);

	if (m_file->get_object(&buf, m, STR_KEY_GENERIC_NAME,
						   json_type_string)) {
		if (!TSGDataType::string_to_ptype(
				generic, json_object_get_string(buf))) return false;
	}

	if (!m_file->get_object(&buf, m, STR_KEY_INSTANCE,
							json_type_object)) return false;
	m_file->push_object(buf);

	return true;
}

bool
SerializableJsonReader00::read_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (*sgserializable_name == '\0') return true;

	m_file->pop_object();
	return true;
}

bool
SerializableJsonReader00::read_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	json_object* m = m_file->m_stack_stream.back();

	if (!json_object_is_type(m, json_type_object)) return false;

	json_object* buf_type;
	if (!m_file->get_object(&buf_type, m, name, json_type_object))
		return false;

	string_t str_buf; json_object* buf;
	type->to_string(str_buf, STRING_LEN);
	if (!m_file->get_object(&buf, buf_type, STR_KEY_TYPE,
							json_type_string)) return false;
	if (strcmp(str_buf, json_object_get_string(buf)) != 0)
		return false;

	// data (and so buf) can be NULL for empty objects
	m_file->get_object_any(&buf, buf_type, STR_KEY_DATA);
	m_file->push_object(buf);

	return true;
}

bool
SerializableJsonReader00::read_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	m_file->pop_object();
	return true;
}

#endif /* HAVE_JSON  */
