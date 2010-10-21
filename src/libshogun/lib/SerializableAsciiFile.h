/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __SERIALIZABLE_ASCII_FILE_H__
#define __SERIALIZABLE_ASCII_FILE_H__

#include "lib/common.h"
#include "lib/SerializableFile.h"

namespace shogun
{
class CSerializableAsciiFile :public CSerializableFile
{
	::std::vector<long> stack_fpos;

	void init(void);
	bool ignore(void);

protected:
	virtual bool write_scalar_wrapped(const TSGDataType* type,
									  const void* param);
	virtual bool read_scalar_wrapped(const TSGDataType* type,
									 void* param);

	virtual bool write_cont_begin_wrapped(const TSGDataType* type,
										  index_t len_real_y,
										  index_t len_real_x);
	virtual bool read_cont_begin_wrapped(const TSGDataType* type,
										 index_t* len_read_y,
										 index_t* len_read_x);

	virtual bool write_cont_end_wrapped(const TSGDataType* type,
										index_t len_real_y,
										index_t len_real_x);
	virtual bool read_cont_end_wrapped(const TSGDataType* type,
									   index_t len_read_y,
									   index_t len_read_x);

	virtual bool write_item_begin_wrapped(const TSGDataType* type,
										  index_t y, index_t x);
	virtual bool read_item_begin_wrapped(const TSGDataType* type,
										 index_t y, index_t x);

	virtual bool write_item_end_wrapped(const TSGDataType* type,
										index_t y, index_t x);
	virtual bool read_item_end_wrapped(const TSGDataType* type,
									   index_t y, index_t x);

	virtual bool write_sgserializable_begin_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitveType generic);
	virtual bool read_sgserializable_begin_wrapped(
		const TSGDataType* type, char* sgserializable_name,
		EPrimitveType* generic);

	virtual bool write_sgserializable_end_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitveType generic);
	virtual bool read_sgserializable_end_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitveType generic);

	virtual bool write_type_begin_wrapped(const TSGDataType* type,
										  const char* name,
										  const char* prefix);
	virtual bool read_type_begin_wrapped(const TSGDataType* type,
										 const char* name,
										 const char* prefix);

	virtual bool write_type_end_wrapped(const TSGDataType* type,
										const char* name,
										const char* prefix);
	virtual bool read_type_end_wrapped(const TSGDataType* type,
									   const char* name,
									   const char* prefix);

public:
	/** default constructor */
	explicit CSerializableAsciiFile(void);

	/** constructor
	 *
	 * @param f already opened file
	 */
	explicit CSerializableAsciiFile(FILE* fstream, char rw);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	explicit CSerializableAsciiFile(char* fname, char rw='r');

	/** default destructor */
	virtual ~CSerializableAsciiFile();

	/** @return object name */
	inline virtual const char* get_name() const {
		return "SerializableAsciiFile";
	}
};
}

#endif /* __SERIALIZABLE_ASCII_FILE_H__  */
