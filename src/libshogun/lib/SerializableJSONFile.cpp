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

#include "lib/SerializableJSONFile.h"

using namespace shogun;

CSerializableJSONFile::CSerializableJSONFile(void)
	:CSerializableFile() { init(""); }

CSerializableJSONFile::CSerializableJSONFile(const char* fname, char rw)
	:CSerializableFile()
{
	CSerializableFile::init(NULL, rw, fname);
	init(fname);
}

CSerializableJSONFile::~CSerializableJSONFile()
{
	close();
}

void
CSerializableJSONFile::init(const char* fname)
{
	if (m_filename == NULL || *m_filename == '\0') {
		SG_WARNING("Filename not given for opening file!\n");
		close(); return;
	}

	if (m_task == 'r') {
		json_object* buf = json_object_from_file((char*) fname);
		if (is_error(buf)) {
			SG_WARNING("Could not open file `%s' for reading!\n",
					   fname);
			return;
		}
		m_stack_stream.push_back(buf);
	} else
		m_stack_stream.push_back(json_object_new_object());
}

void
CSerializableJSONFile::close(void)
{
	while (m_stack_stream.get_num_elements() > 1) {
		json_object_put(m_stack_stream.back());
		m_stack_stream.pop_back();
	}

	if (m_stack_stream.get_num_elements() == 1) {
		if (m_task == 'w'
			&& is_error(
				json_object_to_file(m_filename, m_stack_stream.back())
				)) {
			SG_WARNING("Could not close file `%s' for writing!\n",
					   m_filename);
		}

		m_stack_stream.pop_back();
	}
}

bool
CSerializableJSONFile::is_opened(void)
{
	return m_stack_stream.get_num_elements() > 0;
}

bool
CSerializableJSONFile::write_scalar_wrapped(
	const TSGDataType* type, const void* param)
{
	return true;
}

bool
CSerializableJSONFile::read_scalar_wrapped(
	const TSGDataType* type, void* param)
{
	return true;
}

bool
CSerializableJSONFile::write_cont_begin_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	return true;
}

bool
CSerializableJSONFile::read_cont_begin_wrapped(
	const TSGDataType* type, index_t* len_read_y, index_t* len_read_x)
{
	return true;
}

bool
CSerializableJSONFile::write_cont_end_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	return true;
}

bool
CSerializableJSONFile::read_cont_end_wrapped(
	const TSGDataType* type, index_t len_read_y, index_t len_read_x)
{
	return true;
}

bool
CSerializableJSONFile::write_string_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableJSONFile::read_string_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	return true;
}

bool
CSerializableJSONFile::write_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableJSONFile::read_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableJSONFile::write_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
CSerializableJSONFile::read_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
CSerializableJSONFile::write_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
CSerializableJSONFile::read_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
CSerializableJSONFile::write_sparse_begin_wrapped(
	const TSGDataType* type, index_t vec_index,
	index_t length)
{
	return true;
}

bool
CSerializableJSONFile::read_sparse_begin_wrapped(
	const TSGDataType* type, index_t* vec_index,
	index_t* length)
{
	return true;
}

bool
CSerializableJSONFile::write_sparse_end_wrapped(
	const TSGDataType* type, index_t vec_index,
	index_t length)
{
	return true;
}

bool
CSerializableJSONFile::read_sparse_end_wrapped(
	const TSGDataType* type, index_t* vec_index,
	index_t length)
{
	return true;
}

bool
CSerializableJSONFile::write_sparseentry_begin_wrapped(
	const TSGDataType* type, index_t feat_index, index_t y)
{
	return true;
}

bool
CSerializableJSONFile::read_sparseentry_begin_wrapped(
	const TSGDataType* type, index_t* feat_index, index_t y)
{
	return true;
}

bool
CSerializableJSONFile::write_sparseentry_end_wrapped(
	const TSGDataType* type, index_t feat_index, index_t y)
{
	return true;
}

bool
CSerializableJSONFile::read_sparseentry_end_wrapped(
	const TSGDataType* type, index_t* feat_index, index_t y)
{
	return true;
}

bool
CSerializableJSONFile::write_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	return true;
}

bool
CSerializableJSONFile::read_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	return true;
}

bool
CSerializableJSONFile::write_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	return true;
}

bool
CSerializableJSONFile::read_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	return true;
}

bool
CSerializableJSONFile::write_sgserializable_begin_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	return true;
}

bool
CSerializableJSONFile::read_sgserializable_begin_wrapped(
	const TSGDataType* type, char* sgserializable_name,
	EPrimitiveType* generic)
{
	return true;
}

bool
CSerializableJSONFile::write_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	return true;
}

bool
CSerializableJSONFile::read_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	return true;
}

bool
CSerializableJSONFile::write_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	return true;
}

bool
CSerializableJSONFile::read_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	return true;
}

bool
CSerializableJSONFile::write_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	return true;
}

bool
CSerializableJSONFile::read_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	return true;
}

#endif /* HAVE_JSON  */
