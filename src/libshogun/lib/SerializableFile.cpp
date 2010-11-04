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

#include "lib/SerializableFile.h"

using namespace shogun;

CSerializableFile::CSerializableFile(void)
	:CSGObject()
{
	init(NULL, 0, "(file)");
}

CSerializableFile::CSerializableFile(FILE* fstream, char rw)
	:CSGObject()
{
	init(fstream, rw, "(file)");
}

CSerializableFile::CSerializableFile(const char* fname, char rw)
	:CSGObject()
{
	char mode[3] = {rw, 'b', '\0'};

	init(NULL, rw, fname);

	if (m_filename == NULL || *m_filename == '\0') {
		SG_WARNING("Filename not given for opening file!\n");
		close(); return;
	}

	if (rw != 'r' && rw != 'w') {
		SG_WARNING("Unknown mode '%c'!\n", mode[0]);
		close(); return;
	}

	m_fstream = fopen(m_filename, mode);
	if (!is_opened()) {
		SG_WARNING("Error opening file '%s'\n", m_filename);
		close(); return;
	}
}

CSerializableFile::~CSerializableFile(void)
{
	close();
	if (m_filename != NULL) { free(m_filename); m_filename = NULL; }
	m_task = 0;
}

void
CSerializableFile::init(FILE* fstream, char task, const char* filename)
{
	m_fstream = fstream; m_task = task; m_filename = strdup(filename);
}

void
CSerializableFile::close(void)
{
	if (is_opened()) { fclose(m_fstream); m_fstream = NULL; }
}

bool
CSerializableFile::is_opened(void)
{
	return m_fstream != NULL;
}

bool
CSerializableFile::is_task_warn(char rw)
{
	if (rw == 'w' && (m_task != 'w' || !is_opened())) {
		SG_WARNING("`%s' not opened for writing!\n", m_filename);
		return false;
	}
	if (rw == 'r' && (m_task != 'r' || !is_opened())) {
		SG_WARNING("`%s' not opened for reading!\n", m_filename);
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
	if (!is_task_warn('w')) return false;

	if (!write_scalar_wrapped(type, param))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_scalar(
	const TSGDataType* type, const char* name, const char* prefix,
	void* param)
{
	if (!is_task_warn('r')) return false;

	if (!read_scalar_wrapped(type, param))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_cont_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t len_real_y, index_t len_real_x)
{
	if (!is_task_warn('w')) return false;

	if (!write_cont_begin_wrapped(type, len_real_y, len_real_x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_cont_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t* len_read_y, index_t* len_read_x)
{
	if (!is_task_warn('r')) return false;

	if (!read_cont_begin_wrapped(type, len_read_y, len_read_x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_cont_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t len_real_y, index_t len_real_x)
{
	if (!is_task_warn('w')) return false;

	if (!write_cont_end_wrapped(type, len_real_y, len_real_x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_cont_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t len_read_y, index_t len_read_x)
{
	if (!is_task_warn('r')) return false;

	if (!read_cont_end_wrapped(type, len_read_y, len_read_x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_item_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y, index_t x)
{
	if (!is_task_warn('w')) return false;

	if (!write_item_begin_wrapped(type, y, x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_item_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y, index_t x)
{
	if (!is_task_warn('r')) return false;

	if (!read_item_begin_wrapped(type, y, x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_item_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y, index_t x)
{
	if (!is_task_warn('w')) return false;

	if (!write_item_end_wrapped(type, y, x))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_item_end(
	const TSGDataType* type, const char* name, const char* prefix,
	index_t y, index_t x)
{
	if (!is_task_warn('r')) return false;

	if (!read_item_end_wrapped(type, y, x))
		return false_warn(prefix, name);

	return true;
}


bool
CSerializableFile::write_sgserializable_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	const char* sgserializable_name, EPrimitveType generic)
{
	if (!is_task_warn('w')) return false;

	if (!write_sgserializable_begin_wrapped(type, sgserializable_name,
											generic))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_sgserializable_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	char* sgserializable_name, EPrimitveType* generic)
{
	if (!is_task_warn('r')) return false;

	if (!read_sgserializable_begin_wrapped(type, sgserializable_name,
										   generic))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_sgserializable_end(
	const TSGDataType* type, const char* name, const char* prefix,
	const char* sgserializable_name, EPrimitveType generic)
{
	if (!is_task_warn('w')) return false;

	if (!write_sgserializable_end_wrapped(type, sgserializable_name,
										  generic))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_sgserializable_end(
	const TSGDataType* type, const char* name, const char* prefix,
	const char* sgserializable_name, EPrimitveType generic)
{
	if (!is_task_warn('r')) return false;

	if (!read_sgserializable_end_wrapped(type, sgserializable_name,
										 generic))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_type_begin(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (!is_task_warn('w')) return false;

	if (!write_type_begin_wrapped(type, name, prefix))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_type_begin(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (!is_task_warn('r')) return false;

	if (!read_type_begin_wrapped(type, name, prefix))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_type_end(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (!is_task_warn('w')) return false;

	if (!write_type_end_wrapped(type, name, prefix))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_type_end(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (!is_task_warn('r')) return false;

	if (!read_type_end_wrapped(type, name, prefix))
		return false_warn(prefix, name);

	return true;
}
