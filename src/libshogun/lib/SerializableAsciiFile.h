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
	bool write_scalar(EPrimitveType type, const void* param);
	bool write_vector(const TSGDataType* type, const void* param,
					  uint64_t size);

protected:
	virtual bool write_type_wrapped(
		const TSGDataType* type, const void* param, const char* name,
		const char* prefix);
	virtual bool read_type_wrapped(
		const TSGDataType* type, void* param, const char* name,
		const char* prefix);

public:
	/** default constructor */
	explicit CSerializableAsciiFile(void);

	/** constructor
	 *
	 * @param f already opened file
	 */
	explicit CSerializableAsciiFile(FILE* f, char rw);

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
