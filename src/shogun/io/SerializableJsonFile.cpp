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

#include <shogun/io/SerializableJsonFile.h>
#include <shogun/io/SerializableJsonReader00.h>

#define STR_KEY_FILETYPE           "filetype"
#define STR_FILETYPE_00 \
	"_SHOGUN_SERIALIZABLE_JSON_FILE_V_00_"

using namespace shogun;

CSerializableJsonFile::CSerializableJsonFile()
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

CSerializableFile::TSerializableReader*
CSerializableJsonFile::new_reader(char* dest_version, size_t n)
{
	const char* ftype;
	json_object* buf;

	bool success = json_object_object_get_ex(
                         m_stack_stream.back(), STR_KEY_FILETYPE, &buf);

	if (!success || buf == NULL
		|| is_error(buf)
		|| (ftype = json_object_get_string(buf)) == NULL)
		return NULL;

	strncpy(dest_version, ftype, n);

	if (strcmp(STR_FILETYPE_00, dest_version) == 0)
		return new SerializableJsonReader00(this);

	return NULL;
}

void CSerializableJsonFile::push_object(json_object* o)
{
	m_stack_stream.push_back(o);
	json_object_get(o);
}

void CSerializableJsonFile::pop_object()
{
	json_object_put(m_stack_stream.back());
	m_stack_stream.pop_back();
}

bool
CSerializableJsonFile::get_object_any(
	json_object** dest, json_object* src, const char* key)
{
	return json_object_object_get_ex(src, key, & *dest);
}

bool
CSerializableJsonFile::get_object(json_object** dest, json_object* src,
								  const char* key, json_type t)
{
        bool success = true ;
        success = json_object_object_get_ex(src, key, & *dest);

	return success && *dest != NULL && !is_error(*dest)
		&& json_object_is_type(*dest, t);
}

void
CSerializableJsonFile::init(const char* fname)
{
	if (m_filename == NULL || *m_filename == '\0') {
		SG_WARNING("Filename not given for opening file!\n")
		close();
		return;
	}

	json_object* buf;
	switch (m_task) {
	case 'r':
		buf = json_object_from_file((char*) fname);
		if (is_error(buf)) {
			SG_ERROR("Could not open file `%s' for reading!\n",
					   fname);
			return;
		}
		m_stack_stream.push_back(buf);
		break;
	case 'w':
		m_stack_stream.push_back(json_object_new_object());
		buf = json_object_new_string(STR_FILETYPE_00);
		json_object_object_add(m_stack_stream.back(),
							   STR_KEY_FILETYPE, buf);
		break;
	default:
		SG_WARNING("Could not open file `%s', unknown mode!\n",
				   m_filename);
		close();
		return;
	}
}

void
CSerializableJsonFile::close()
{
	while (m_stack_stream.get_num_elements() > 1)
		pop_object();

	if (m_stack_stream.get_num_elements() == 1) {
		if (m_task == 'w'
			&& json_object_to_file(m_filename, m_stack_stream.back()))
		{
			SG_WARNING("Could not close file `%s' for writing!\n",
					   m_filename);
		}

		pop_object();
	}
}

bool
CSerializableJsonFile::is_opened()
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
	case PT_COMPLEX128:
		SG_ERROR("Not supported for complex128_t for writing into JsonFile!");
		break;
	case PT_SGOBJECT:
		SG_ERROR("Implementation error during writing JsonFile!");
		return false;
	case PT_UNDEFINED: default:
		SG_ERROR("Implementation error: undefined primitive type\n");
		return false;
		break;
	}

	if (is_error(m_stack_stream.back()))
		return false;

	return true;
}

bool
CSerializableJsonFile::write_cont_begin_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	push_object(json_object_new_array());

	for (index_t i=0; i<len_real_x && (type->m_ctype==CT_MATRIX || type->m_ctype==CT_SGMATRIX); i++)
		json_object_array_add(m_stack_stream.back(),
							  json_object_new_array());

	return true;
}

bool
CSerializableJsonFile::write_cont_end_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
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
CSerializableJsonFile::write_string_end_wrapped(
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
CSerializableJsonFile::write_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	json_object* array = m_stack_stream.get_element(
		m_stack_stream.get_num_elements() - 2);

	if (json_object_array_put_idx( array, y, m_stack_stream.back()))
		return false;

	pop_object();
	return true;
}

bool
CSerializableJsonFile::write_sparse_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	push_object(json_object_new_object());

	json_object* buf = json_object_new_array();
	if (is_error(buf))
		return false;

	json_object_object_add(m_stack_stream.back(),
			STR_KEY_SPARSE_FEATURES, buf);

	push_object(buf);
	return true;
}

bool
CSerializableJsonFile::write_sparse_end_wrapped(
	const TSGDataType* type, index_t length)
{
	pop_object();
	return true;
}

bool
CSerializableJsonFile::write_sparseentry_begin_wrapped(
	const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	json_object* buf = json_object_new_object();
	if (json_object_array_put_idx(m_stack_stream.back(), y, buf))
		return false;

	push_object(buf);

	buf = json_object_new_int(feat_index);
	if (is_error(buf))
		return false;

	json_object_object_add(m_stack_stream.back(),
						   STR_KEY_SPARSE_FEATINDEX, buf);

	return true;
}

bool
CSerializableJsonFile::write_sparseentry_end_wrapped(
	const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
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
CSerializableJsonFile::write_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	return true;
}

bool
CSerializableJsonFile::write_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	json_object* array = m_stack_stream.get_element(
		m_stack_stream.get_num_elements() - 2);

	if (type->m_ctype==CT_MATRIX || type->m_ctype==CT_SGMATRIX)
		array = json_object_array_get_idx(array, x);

	json_object_array_put_idx(array, y, m_stack_stream.back());

	pop_object();
	return true;
}

bool
CSerializableJsonFile::write_sgserializable_begin_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (*sgserializable_name == '\0') {
		push_object(NULL);
		return true;
	}

	push_object(json_object_new_object());

	json_object* buf;
	buf = json_object_new_string(sgserializable_name);
	if (is_error(buf))
		return false;

	json_object_object_add(m_stack_stream.back(),
						   STR_KEY_INSTANCE_NAME, buf);

	if (generic != PT_NOT_GENERIC) {
		string_t buf_str;
		TSGDataType::ptype_to_string(buf_str, generic, STRING_LEN);
		buf = json_object_new_string(buf_str);
		if (is_error(buf))
			return false;

		json_object_object_add(m_stack_stream.back(),
							   STR_KEY_GENERIC_NAME, buf);
	}

	buf = json_object_new_object();
	if (is_error(buf))
		return false;
	json_object_object_add(m_stack_stream.back(), STR_KEY_INSTANCE,
						   buf);
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
CSerializableJsonFile::write_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	json_object* buf = json_object_new_object();
	if (is_error(buf))
		return false;

	json_object_object_add(m_stack_stream.back(), name, buf);
	push_object(buf);

	string_t str_buf;
	type->to_string(str_buf, STRING_LEN);
	buf = json_object_new_string(str_buf);
	if (is_error(buf))
		return false;

	json_object_object_add(m_stack_stream.back(), STR_KEY_TYPE, buf);

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

#endif /* HAVE_JSON  */
