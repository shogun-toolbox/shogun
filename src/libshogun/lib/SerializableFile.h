/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef __SERIALIZABLE_FILE_H__
#define __SERIALIZABLE_FILE_H__

#include <stdio.h>
#include "base/SGObject.h"
#include "lib/DataType.h"

namespace shogun
{
class CSerializableFile :public CSGObject
{
	void init(FILE* file_, char task_, const char* filename_);
	bool is_task_warn(char rw);
	bool false_warn(const char* prefix, const char* name);

protected:
	/** file object */
	FILE* file;
	/** task */
	char task;
	/** name of the handled file */
	char* filename;

	virtual bool write_scalar_wrapped(const TSGDataType* type,
									  const void* param) = 0;
	virtual bool read_scalar_wrapped(const TSGDataType* type,
									 void* param) = 0;

	virtual bool write_cont_begin_wrapped(const TSGDataType* type,
										  index_t len_real_y,
										  index_t len_real_x) = 0;
	virtual bool read_cont_begin_wrapped(const TSGDataType* type,
										 index_t* len_read_y,
										 index_t* len_read_x) = 0;

	virtual bool write_cont_end_wrapped(const TSGDataType* type) = 0;
	virtual bool read_cont_end_wrapped(const TSGDataType* type) = 0;

	virtual bool write_item_begin_wrapped(const TSGDataType* type,
										  index_t y, index_t x) = 0;
	virtual bool read_item_begin_wrapped(const TSGDataType* type,
										 index_t y, index_t x) = 0;

	virtual bool write_item_end_wrapped(const TSGDataType* type,
										index_t y, index_t x) = 0;
	virtual bool read_item_end_wrapped(const TSGDataType* type,
									   index_t y, index_t x) = 0;

	virtual bool write_sgserializable_begin_wrapped(
		const TSGDataType* type) = 0;
	virtual bool read_sgserializable_begin_wrapped(
		const TSGDataType* type) = 0;

	virtual bool write_sgserializable_end_wrapped(
		const TSGDataType* type) = 0;
	virtual bool read_sgserializable_end_wrapped(
		const TSGDataType* type) = 0;

	virtual bool write_type_begin_wrapped(const TSGDataType* type,
										  const char* name,
										  const char* prefix) = 0;
	virtual bool read_type_begin_wrapped(const TSGDataType* type,
										 const char* name,
										 const char* prefix) = 0;

	virtual bool write_type_end_wrapped(const TSGDataType* type,
										const char* name,
										const char* prefix) = 0;
	virtual bool read_type_end_wrapped(const TSGDataType* type,
									   const char* name,
									   const char* prefix) = 0;

public:
	/** default constructor */
	explicit CSerializableFile(void);

	/** constructor
	 *
	 * @param f already opened file
	 */
	explicit CSerializableFile(FILE* f, char rw);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	explicit CSerializableFile(char* fname, char rw='r');

	/** default destructor */
	virtual ~CSerializableFile(void);

	/** @return object name */
	inline virtual const char* get_name() const
		{ return "SerializableFile"; }

	virtual void close(void);
	virtual bool is_opened(void);

	virtual bool write_scalar(
		const TSGDataType* type, const char* name, const char* prefix,
		const void* param);
	virtual bool read_scalar(
		const TSGDataType* type, const char* name, const char* prefix,
		void* param);

	virtual bool write_cont_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t len_real_y, index_t len_real_x);
	virtual bool read_cont_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t* len_read_y, index_t* len_read_x);

	virtual bool write_cont_end(
		const TSGDataType* type, const char* name, const char* prefix);
	virtual bool read_cont_end(
		const TSGDataType* type, const char* name, const char* prefix);

	virtual bool write_item_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y, index_t x);
	virtual bool read_item_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y, index_t x);

	virtual bool write_item_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y, index_t x);
	virtual bool read_item_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y, index_t x);

	virtual bool write_sgserializable_begin(
		const TSGDataType* type, const char* name, const char* prefix);
	virtual bool read_sgserializable_begin(
		const TSGDataType* type, const char* name, const char* prefix);

	virtual bool write_sgserializable_end(
		const TSGDataType* type, const char* name, const char* prefix);
	virtual bool read_sgserializable_end(
		const TSGDataType* type, const char* name, const char* prefix);

	virtual bool write_type_begin(
		const TSGDataType* type, const char* name, const char* prefix);
	virtual bool read_type_begin(
		const TSGDataType* type, const char* name, const char* prefix);

	virtual bool write_type_end(
		const TSGDataType* type, const char* name, const char* prefix);
	virtual bool read_type_end(
		const TSGDataType* type, const char* name, const char* prefix);
};
}
#endif // __SERIALIZABLE_FILE_H__
