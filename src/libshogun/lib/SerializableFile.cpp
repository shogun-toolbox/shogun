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

CSerializableFile::CSerializableFile(void) :CSGObject()
{
	init(NULL, 0, "(file)");
}

CSerializableFile::CSerializableFile(FILE* f, char rw) :CSGObject()
{
	init(f, rw, "(file)");
}

CSerializableFile::CSerializableFile(char* fname, char rw) :CSGObject()
{
	char mode[3] = {rw, 'b', '\0'};

	init(NULL, rw, fname);

	if (filename == NULL || *filename == '\0') {
		SG_WARNING("Filename not given for opening file!\n");
		close(); return;
	}

	if (rw != 'r' && rw != 'w') {
		SG_WARNING("Unknown mode '%c'!\n", mode[0]);
		close(); return;
	}

	file = fopen(filename, mode);
	if (file == NULL) {
		SG_WARNING("Error opening file '%s'\n", filename);
		close(); return;
	}
}

CSerializableFile::~CSerializableFile(void)
{
	close();
	if (filename != NULL) { free(filename); filename = NULL; }
}

void
CSerializableFile::init(FILE* file_, char task_, const char* filename_)
{
	file = file_; task = task_; filename = strdup(filename_);
}

void
CSerializableFile::close()
{
	if (file != NULL) { fclose(file); file = NULL; }
	task = 0;
}

bool
CSerializableFile::is_opened(void)
{
	return file != NULL;
}

bool
CSerializableFile::is_task_warn(char rw)
{
	if (rw == 'w' && task != 'w') {
		SG_WARNING("`%s' not opened for writing!\n", filename);
		return false;
	}
	if (rw == 'r' && task != 'r') {
		SG_WARNING("`%s' not opened for reading!\n", filename);
		return false;
	}

	return true;
}

bool
CSerializableFile::false_warn(const char* prefix, const char* name)
{
	if (task == 'w')
		SG_WARNING("Could not write `%s%s' from `%s'!", prefix,
				   name, filename);
	if (task == 'r')
		SG_WARNING("Could not read `%s%s' from `%s'!", prefix,
				   name, filename);
	if (task != 'w' && task != 'r')
		SG_WARNING("Could not read/write `%s%s' from `%s'!",
				   prefix, name, filename);

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
	bool is_null)
{
	if (!is_task_warn('w')) return false;

	if (!write_sgserializable_begin_wrapped(type, is_null))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_sgserializable_begin(
	const TSGDataType* type, const char* name, const char* prefix,
	bool* is_null)
{
	if (!is_task_warn('r')) return false;

	if (!read_sgserializable_begin_wrapped(type, is_null))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::write_sgserializable_end(
	const TSGDataType* type, const char* name, const char* prefix,
	bool is_null)
{
	if (!is_task_warn('w')) return false;

	if (!write_sgserializable_end_wrapped(type, is_null))
		return false_warn(prefix, name);

	return true;
}

bool
CSerializableFile::read_sgserializable_end(
	const TSGDataType* type, const char* name, const char* prefix,
	bool is_null)
{
	if (!is_task_warn('r')) return false;

	if (!read_sgserializable_end_wrapped(type, is_null))
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
