/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/io/SerializableFile.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CSerializableFile::CSerializableFile()
	: CSGObject(), m_filename(NULL)
{
	init(NULL, 0, "(file)");
}

CSerializableFile::CSerializableFile(FILE* fstream, char rw)
	:CSGObject(), m_filename(NULL)
{
	REQUIRE(fstream != NULL, "Provided fstream should be != NULL\n");
	init(fstream, rw, "(file)");
}

CSerializableFile::CSerializableFile(const char* fname, char rw)
	:CSGObject(), m_filename(NULL)
{
	const char mode[3] = {rw, 'b', '\0'};

	if (fname == NULL || *fname == '\0') {
		SG_ERROR("Filename not given for opening file!\n")
		close(); return;
	}

	if (rw != 'r' && rw != 'w') {
		SG_ERROR("Unknown mode '%c'!\n", mode[0])
		close(); return;
	}

	FILE* fstream = fopen(fname, mode);
	if (!fstream) {
		SG_ERROR("Error opening file '%s'\n", fname)
		close(); return;
	}

	init(fstream, rw, fname);
}

CSerializableFile::~CSerializableFile()
{
	close();
	SG_FREE(m_filename);
	delete m_reader;
	m_task = 0;
}

void
CSerializableFile::init(FILE* fstream, char task, const char* filename)
{
	m_fstream = fstream;
	m_task = task;
	SG_FREE(m_filename);
	m_filename = SG_MALLOC(char, strlen(filename)+1);
	strcpy(m_filename, filename);
	m_reader = NULL;
}

void
CSerializableFile::close()
{
	if (is_opened()) { fclose(m_fstream); m_fstream = NULL; }
}

bool
CSerializableFile::is_opened()
{
	return m_fstream != NULL;
}

bool
CSerializableFile::is_task_warn(char rw, const char* name,
								const char* prefix)
{
	if (m_task == 'r' && m_reader == NULL) {
		string_t dest_version;
		strncpy(dest_version, "(unkown)", STRING_LEN);
		m_reader = new_reader(dest_version, STRING_LEN);
		if (m_reader == NULL) {
			SG_WARNING("`%s' has file-version `%s', which is not "
					   "supported!\n", m_filename, dest_version);
			close(); return false;
		}
	}

	if (rw == 'w' && (m_task != 'w' || !is_opened())) {
		SG_WARNING("`%s' not opened (for writing) during writing "
				   "`%s%s'!\n", m_filename, prefix, name);
		return false;
	}
	if (rw == 'r' && (m_task != 'r' || !is_opened())) {
		SG_WARNING("`%s' not opened (for reading) during reading "
				   "`%s%s'!\n", m_filename, prefix, name);
		return false;
	}

	return true;
}

bool
CSerializableFile::false_warn(const char* prefix, const char* name)
{
	if (m_task == 'w')
		SG_WARNING("Could not write `%s%s' to `%s'!\n", prefix,
				   name, m_filename);
	if (m_task == 'r')
		SG_WARNING("Could not read `%s%s' from `%s'!\n", prefix,
				   name, m_filename);
	if (m_task != 'w' && m_task != 'r')
		SG_WARNING("Could not read/write `%s%s' from `%s'!\n",
				   prefix, name, m_filename);

	return false;
}

bool
CSerializableFile::write_scalar(
	const TSGDataType* type, const char* name, const char* prefix,
	const void* param)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_scalar_wrapped(type, param))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_scalar(
	const TSGDataType* type, const char* name, const char* prefix,
	void* param)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_scalar_wrapped(type, param))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_cont_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t len_real_y, index_t len_real_x)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_cont_begin_wrapped(type, len_real_y, len_real_x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_cont_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t* len_read_y, index_t* len_read_x)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_cont_begin_wrapped(type, len_read_y,
										   len_read_x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_cont_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t len_real_y, index_t len_real_x)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_cont_end_wrapped(type, len_real_y, len_real_x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_cont_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t len_read_y, index_t len_read_x)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_cont_end_wrapped(type, len_read_y, len_read_x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_string_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t length)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_string_begin_wrapped(type, length))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_string_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t* length)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_string_begin_wrapped(type, length))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_string_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t length)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_string_end_wrapped(type, length))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_string_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t length)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_string_end_wrapped(type, length))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_stringentry_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_stringentry_begin_wrapped(type, y))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_stringentry_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_stringentry_begin_wrapped(type, y))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_stringentry_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_stringentry_end_wrapped(type, y))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_stringentry_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_stringentry_end_wrapped(type, y))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_sparse_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t length)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_sparse_begin_wrapped(type, length))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_sparse_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t* length)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_sparse_begin_wrapped(type, length))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_sparse_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t length)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_sparse_end_wrapped(type, length))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_sparse_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t length)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_sparse_end_wrapped(type, length))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_sparseentry_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	const SGSparseVectorEntry<char>* first_entry, index_t feat_index,
	index_t y)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_sparseentry_begin_wrapped(type, first_entry,
										 feat_index, y))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_sparseentry_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	SGSparseVectorEntry<char>* first_entry, index_t* feat_index, index_t y)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_sparseentry_begin_wrapped(type, first_entry,
												  feat_index, y))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_sparseentry_end(
	const TSGDataType* type, const char* name, const char* prefix,
	const SGSparseVectorEntry<char>* first_entry, index_t feat_index,
	index_t y)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_sparseentry_end_wrapped(type, first_entry, feat_index,
									   y))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_sparseentry_end(
	const TSGDataType* type, const char* name, const char* prefix,
	SGSparseVectorEntry<char>* first_entry, index_t* feat_index,
	index_t y)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_sparseentry_end_wrapped(type, first_entry,
												feat_index, y))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_item_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y, index_t x)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_item_begin_wrapped(type, y, x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_item_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y, index_t x)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_item_begin_wrapped(type, y, x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_item_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y, index_t x)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_item_end_wrapped(type, y, x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_item_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y, index_t x)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_item_end_wrapped(type, y, x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_sgserializable_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	const char* sgserializable_name, EPrimitiveType generic)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_sgserializable_begin_wrapped(type, sgserializable_name,
											generic))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_sgserializable_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	char* sgserializable_name, EPrimitiveType* generic)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_sgserializable_begin_wrapped(
			type, sgserializable_name, generic))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_sgserializable_end(
	const TSGDataType* type, const char* name, const char* prefix,
	const char* sgserializable_name, EPrimitiveType generic)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_sgserializable_end_wrapped(type, sgserializable_name,
										  generic))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_sgserializable_end(
	const TSGDataType* type, const char* name, const char* prefix,
	const char* sgserializable_name, EPrimitiveType generic)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_sgserializable_end_wrapped(
			type, sgserializable_name, generic))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_type_begin(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_type_begin_wrapped(type, name, prefix))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_type_begin(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_type_begin_wrapped(type, name, prefix))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_type_end(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (!is_task_warn('w', name, prefix)) return false;

	if (!write_type_end_wrapped(type, name, prefix))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_type_end(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (!is_task_warn('r', name, prefix)) return false;

	if (!m_reader->read_type_end_wrapped(type, name, prefix))
		return false_warn(prefix, name);

	return true;
}
