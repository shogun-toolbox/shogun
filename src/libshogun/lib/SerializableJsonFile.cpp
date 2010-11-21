/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "lib/config.h"
#ifdef HAVE_JSON

#include "lib/SerializableJsonFile.h"

#define STR_KEY_FILETYPE           "filetype"
#define STR_FILETYPE \
	"_SHOGUN_SERIALIZABLE_JSON_FILE_V_00_"

#define STR_KEY_TYPE               "type"
#define STR_KEY_DATA               "data"
#define STR_KEY_INSTANCE_NAME      "instance_name"
#define STR_KEY_INSTANCE           "instance"
#define STR_KEY_GENERIC_NAME       "generic_name"
#define STR_KEY_SPARSE_VECINDEX    "vec_index"
#define STR_KEY_SPARSE_FEATURES    "features"
#define STR_KEY_SPARSE_FEATINDEX   "feat_index"
#define STR_KEY_SPARSE_ENTRY       "entry"

using namespace shogun;

CSerializableJsonFile::CSerializableJsonFile(void)
	:CSerializableFile() { init(""); }

CSerializableJsonFile::CSerializableJsonFile(const char* fname, char rw)
	:CSerializableFile()
{
	CSerializableFile::init(NULL, rw, fname);
	init(fname);
}

CSerializableJsonFile::~CSerializableJsonFile()
{
	close();
}

void
CSerializableJsonFile::push_object(json_object* o)
{ m_stack_stream.push_back(o); json_object_get(o); }

void
CSerializableJsonFile::pop_object(void)
{ json_object_put(m_stack_stream.back()); m_stack_stream.pop_back(); }

bool
CSerializableJsonFile::get_object_any(
	json_object** dest, json_object* src, const char* key)
{
	*dest = json_object_object_get(src, key);

	return !is_error(*dest);
}

bool
CSerializableJsonFile::get_object(json_object** dest, json_object* src,
								  const char* key, json_type t)
{
	*dest = json_object_object_get(src, key);

	return *dest != NULL && !is_error(*dest)
		&& json_object_is_type(*dest, t);
}

void
CSerializableJsonFile::init(const char* fname)
{
	if (m_filename == NULL || *m_filename == '\0') {
		SG_WARNING("Filename not given for opening file!\n");
		close(); return;
	}

	json_object* buf;
	switch (m_task) {
	case 'r':
		buf = json_object_from_file((char*) fname);
		if (is_error(buf)) {
			SG_WARNING("Could not open file `%s' for reading!\n",
					   fname);
			return;
		}
		push_object(buf);

		const char* ftype;
		if ((buf = json_object_object_get(buf, STR_KEY_FILETYPE))
			== NULL || is_error(buf)
			|| (ftype = json_object_get_string(buf)) == NULL
			|| is_error(buf) || strcmp(ftype, STR_FILETYPE) != 0)
		{
			SG_WARNING("%s: Not a Serializable JSON file!\n", fname);
			close(); return;
		}
		break;
	case 'w':
		push_object(json_object_new_object());

		buf = json_object_new_string(STR_FILETYPE);
		json_object_object_add(m_stack_stream.back(),
							   STR_KEY_FILETYPE, buf);
		break;
	default:
		SG_WARNING("Could not open file `%s', unknown mode!\n",
				   m_filename);
		close(); return;
	}
}

void
CSerializableJsonFile::close(void)
{
	while (m_stack_stream.get_num_elements() > 1)
		pop_object();

	if (m_stack_stream.get_num_elements() == 1) {
		if (m_task == 'w'
			&& is_error(
				json_object_to_file(m_filename, m_stack_stream.back())
				)) {
			SG_WARNING("Could not close file `%s' for writing!\n",
					   m_filename);
		}

		pop_object();
	}
}

bool
CSerializableJsonFile::is_opened(void)
{
	return m_stack_stream.get_num_elements() > 0;
}

bool
CSerializableJsonFile::write_scalar_wrapped(
	const TSGDataType* type, const void* param)
{
	switch (type->m_ptype) {
	case PT_BOOL:
		push_object(json_object_new_boolean(*(bool*) param));
		break;
	case PT_CHAR:
		push_object(json_object_new_int((int) *(char*) param));
		break;
	case PT_INT8:
		push_object(json_object_new_int((int) *(int8_t*) param));
		break;
	case PT_UINT8:
		push_object(json_object_new_int((int) *(uint8_t*) param));
		break;
	case PT_INT16:
		push_object(json_object_new_int((int) *(int16_t*) param));
		break;
	case PT_UINT16:
		push_object(json_object_new_int((int) *(uint16_t*) param));
		break;
	case PT_INT32:
		push_object(json_object_new_int((int) *(int32_t*) param));
		break;
	case PT_UINT32:
		push_object(json_object_new_int((int) *(uint32_t*) param));
		break;
	case PT_INT64:
		push_object(json_object_new_int((int) *(int64_t*) param));
		break;
	case PT_UINT64:
		push_object(json_object_new_int((int) *(uint64_t*) param));
		break;
	case PT_FLOAT32:
		push_object(json_object_new_double(
						(double) *(float32_t*) param));
		break;
	case PT_FLOAT64:
		push_object(json_object_new_double(
						(double) *(float64_t*) param));
		break;
	case PT_FLOATMAX:
		push_object(json_object_new_double(
						(double) *(floatmax_t*) param));
		break;
	case PT_SGSERIALIZABLE_PTR:
		SG_ERROR("write_scalar_wrapped(): Implementation error during"
				 " writing JsonFile!");
		return false;
	}

	if (is_error(m_stack_stream.back())) return false;

	return true;
}

bool
CSerializableJsonFile::read_scalar_wrapped(
	const TSGDataType* type, void* param)
{
	switch (type->m_ptype) {
	case PT_BOOL:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_boolean)) return false;
		*(bool*) param = json_object_get_boolean(
			m_stack_stream.back());
		break;
	case PT_CHAR:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_int)) return false;
		*(char*) param = json_object_get_int(
			m_stack_stream.back());
		break;
	case PT_INT8:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_int)) return false;
		*(int8_t*) param = json_object_get_int(
			m_stack_stream.back());
		break;
	case PT_UINT8:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_int)) return false;
		*(uint8_t*) param = json_object_get_int(
			m_stack_stream.back());
		break;
	case PT_INT16:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_int)) return false;
		*(int16_t*) param = json_object_get_int(
			m_stack_stream.back());
		break;
	case PT_UINT16:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_int)) return false;
		*(uint16_t*) param = json_object_get_int(
			m_stack_stream.back());
		break;
	case PT_INT32:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_int)) return false;
		*(int32_t*) param = json_object_get_int(
			m_stack_stream.back());
		break;
	case PT_UINT32:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_int)) return false;
		*(uint32_t*) param = json_object_get_int(
			m_stack_stream.back());
		break;
	case PT_INT64:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_int)) return false;
		*(int64_t*) param = json_object_get_int(
			m_stack_stream.back());
		break;
	case PT_UINT64:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_int)) return false;
		*(uint64_t*) param = json_object_get_int(
			m_stack_stream.back());
		break;
	case PT_FLOAT32:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_double)) return false;
		*(float32_t*) param = json_object_get_double(
			m_stack_stream.back());
		break;
	case PT_FLOAT64:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_double)) return false;
		*(float64_t*) param = json_object_get_double(
			m_stack_stream.back());
		break;
	case PT_FLOATMAX:
		if (!json_object_is_type(m_stack_stream.back(),
								 json_type_double)) return false;
		*(floatmax_t*) param = json_object_get_double(
			m_stack_stream.back());
		break;
	case PT_SGSERIALIZABLE_PTR:
		SG_ERROR("write_scalar_wrapped(): Implementation error during"
				 " writing JsonFile!");
		return false;
	}

	return true;
}

bool
CSerializableJsonFile::write_cont_begin_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	push_object(json_object_new_array());

	for (index_t i=0; i<len_real_x && type->m_ctype == CT_MATRIX; i++)
		json_object_array_add(m_stack_stream.back(),
							  json_object_new_array());

	return true;
}

bool
CSerializableJsonFile::read_cont_begin_wrapped(
	const TSGDataType* type, index_t* len_read_y, index_t* len_read_x)
{
	if (!json_object_is_type(m_stack_stream.back(), json_type_array))
		return false;

	*len_read_y = json_object_array_length(m_stack_stream.back());

	if (type->m_ctype == CT_MATRIX) {
		*len_read_x = *len_read_y;
		for (index_t i=0; i<*len_read_x; i++) {
			json_object* buf = json_object_array_get_idx(
				m_stack_stream.back(), i);
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
CSerializableJsonFile::write_cont_end_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	return true;
}

bool
CSerializableJsonFile::read_cont_end_wrapped(
	const TSGDataType* type, index_t len_read_y, index_t len_read_x)
{
	return true;
}

bool
CSerializableJsonFile::write_string_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	push_object(json_object_new_array());

	return true;
}

bool
CSerializableJsonFile::read_string_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	if (!json_object_is_type(m_stack_stream.back(), json_type_array))
		return false;

	*length = json_object_array_length(m_stack_stream.back());

	return true;
}

bool
CSerializableJsonFile::write_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableJsonFile::read_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableJsonFile::write_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
CSerializableJsonFile::read_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	json_object* buf = json_object_array_get_idx(
		m_stack_stream.back(), y);
	if (is_error(buf)) return false;

	push_object(buf);
	return true;
}

bool
CSerializableJsonFile::write_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	json_object* array = m_stack_stream.get_element(
		m_stack_stream.get_num_elements() - 2);

	if (is_error(json_object_array_put_idx(
					 array, y, m_stack_stream.back()))) return false;

	pop_object();
	return true;
}

bool
CSerializableJsonFile::read_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	pop_object();
	return true;
}

bool
CSerializableJsonFile::write_sparse_begin_wrapped(
	const TSGDataType* type, index_t vec_index,
	index_t length)
{
	push_object(json_object_new_object());

	json_object* buf = json_object_new_int(vec_index);
	if (is_error(buf)) return false;
	json_object_object_add(m_stack_stream.back(),
						   STR_KEY_SPARSE_VECINDEX, buf);

	buf = json_object_new_array();
	if (is_error(buf)) return false;
	json_object_object_add(m_stack_stream.back(),
						   STR_KEY_SPARSE_FEATURES, buf);

	push_object(buf);
	return true;
}

bool
CSerializableJsonFile::read_sparse_begin_wrapped(
	const TSGDataType* type, index_t* vec_index,
	index_t* length)
{
	if (!json_object_is_type(m_stack_stream.back(), json_type_object))
		return false;

	json_object* buf;
	if (!get_object(&buf, m_stack_stream.back(),
					STR_KEY_SPARSE_VECINDEX, json_type_int))
		return false;
	*vec_index = json_object_get_int(buf);

	if (!get_object(&buf, m_stack_stream.back(),
					STR_KEY_SPARSE_FEATURES, json_type_array))
		return false;
	*length = json_object_array_length(buf);
	push_object(buf);

	return true;
}

bool
CSerializableJsonFile::write_sparse_end_wrapped(
	const TSGDataType* type, index_t vec_index,
	index_t length)
{
	pop_object();
	return true;
}

bool
CSerializableJsonFile::read_sparse_end_wrapped(
	const TSGDataType* type, index_t* vec_index,
	index_t length)
{
	pop_object();
	return true;
}

bool
CSerializableJsonFile::write_sparseentry_begin_wrapped(
	const TSGDataType* type, const TSparseEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	json_object* buf = json_object_new_object();
	if (is_error(json_object_array_put_idx(m_stack_stream.back(), y,
										   buf))) return false;
	push_object(buf);

	buf = json_object_new_int(feat_index);
	if (is_error(buf)) return false;
	json_object_object_add(m_stack_stream.back(),
						   STR_KEY_SPARSE_FEATINDEX, buf);

	return true;
}

bool
CSerializableJsonFile::read_sparseentry_begin_wrapped(
	const TSGDataType* type, TSparseEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	json_object* buf_obj
		= json_object_array_get_idx(m_stack_stream.back(), y);
	if (is_error(buf_obj)) return false;
	if (!json_object_is_type(buf_obj, json_type_object)) return false;

	json_object* buf;
	if (!get_object(&buf, buf_obj, STR_KEY_SPARSE_FEATINDEX,
					json_type_int)) return false;
	*feat_index = json_object_get_int(buf);

	if (!get_object_any(&buf, buf_obj, STR_KEY_SPARSE_ENTRY))
		return false;
	push_object(buf);

	return true;
}

bool
CSerializableJsonFile::write_sparseentry_end_wrapped(
	const TSGDataType* type, const TSparseEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	json_object* o = m_stack_stream.get_element(
		m_stack_stream.get_num_elements() - 2);

	json_object_object_add(o, STR_KEY_SPARSE_ENTRY,
						   m_stack_stream.back());

	pop_object(); pop_object();
	return true;
}

bool
CSerializableJsonFile::read_sparseentry_end_wrapped(
	const TSGDataType* type, TSparseEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	pop_object();
	return true;
}

bool
CSerializableJsonFile::write_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	return true;
}

bool
CSerializableJsonFile::read_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	json_object* buf = m_stack_stream.back();

	if (type->m_ctype == CT_MATRIX)
		buf = json_object_array_get_idx(buf, x);
	buf = json_object_array_get_idx(buf, y);

	push_object(buf);
	return true;
}

bool
CSerializableJsonFile::write_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	json_object* array = m_stack_stream.get_element(
		m_stack_stream.get_num_elements() - 2);

	if (type->m_ctype == CT_MATRIX)
		array = json_object_array_get_idx(array, x);

	json_object_array_put_idx(array, y, m_stack_stream.back());

	pop_object();
	return true;
}

bool
CSerializableJsonFile::read_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	pop_object();
	return true;
}

bool
CSerializableJsonFile::write_sgserializable_begin_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (*sgserializable_name == '\0') {
		push_object(NULL); return true;
	}

	push_object(json_object_new_object());

	json_object* buf;
	buf = json_object_new_string(sgserializable_name);
	if (is_error(buf)) return false;
	json_object_object_add(m_stack_stream.back(),
						   STR_KEY_INSTANCE_NAME, buf);

	if (generic != PT_NOT_GENERIC) {
		string_t buf_str;
		TSGDataType::ptype_to_string(buf_str, generic, STRING_LEN);
		buf = json_object_new_string(buf_str);
		if (is_error(buf)) return false;
		json_object_object_add(m_stack_stream.back(),
							   STR_KEY_GENERIC_NAME, buf);
	}

	buf = json_object_new_object();
	if (is_error(buf)) return false;
	json_object_object_add(m_stack_stream.back(), STR_KEY_INSTANCE,
						   buf);
	push_object(buf);

	return true;
}

bool
CSerializableJsonFile::read_sgserializable_begin_wrapped(
	const TSGDataType* type, char* sgserializable_name,
	EPrimitiveType* generic)
{
	if (m_stack_stream.back() == NULL ||
		json_object_is_type(m_stack_stream.back(), json_type_null)) {
		*sgserializable_name = '\0'; return true;
	}

	if (!json_object_is_type(m_stack_stream.back(), json_type_object))
		return false;

	json_object* buf;
	if (!get_object(&buf, m_stack_stream.back(), STR_KEY_INSTANCE_NAME,
					json_type_string)) return false;
	strncpy(sgserializable_name, json_object_get_string(buf),
			STRING_LEN);

	if (get_object(&buf, m_stack_stream.back(), STR_KEY_GENERIC_NAME,
				   json_type_string)) {
		if (!TSGDataType::string_to_ptype(
				generic, json_object_get_string(buf))) return false;
	}

	if (!get_object(&buf, m_stack_stream.back(), STR_KEY_INSTANCE,
					json_type_object)) return false;
	push_object(buf);

	return true;
}

bool
CSerializableJsonFile::write_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (*sgserializable_name == '\0') return true;

	pop_object();
	return true;
}

bool
CSerializableJsonFile::read_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (*sgserializable_name == '\0') return true;

	pop_object();
	return true;
}

bool
CSerializableJsonFile::write_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	json_object* buf = json_object_new_object();
	if (is_error(buf)) return false;

	json_object_object_add(m_stack_stream.back(), name, buf);
	push_object(buf);

	string_t str_buf;
	type->to_string(str_buf, STRING_LEN);
	buf = json_object_new_string(str_buf);
	if (is_error(buf)) return false;
	json_object_object_add(m_stack_stream.back(), STR_KEY_TYPE, buf);

	return true;
}

bool
CSerializableJsonFile::read_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (!json_object_is_type(m_stack_stream.back(), json_type_object))
		return false;

	json_object* buf_type;
	if (!get_object(&buf_type, m_stack_stream.back(), name,
					json_type_object)) return false;

	string_t str_buf; json_object* buf;
	type->to_string(str_buf, STRING_LEN);
	if (!get_object(&buf, buf_type, STR_KEY_TYPE, json_type_string))
		return false;
	if (strcmp(str_buf, json_object_get_string(buf)) != 0)
		return false;

	if (!get_object_any(&buf, buf_type, STR_KEY_DATA)) return false;
	push_object(buf);

	return true;
}

bool
CSerializableJsonFile::write_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	json_object_object_add(
		m_stack_stream.get_element(
			m_stack_stream.get_num_elements() - 2), STR_KEY_DATA,
		m_stack_stream.back());
	pop_object();

	pop_object();
	return true;
}

bool
CSerializableJsonFile::read_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	pop_object();
	return true;
}

#endif /* HAVE_JSON  */
