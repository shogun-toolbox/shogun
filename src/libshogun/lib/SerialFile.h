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

#ifndef __SERIALFILE_H__
#define __SERIALFILE_H__

#include <stdio.h>
#include "base/SGObject.h"
#include "lib/DataType.h"

namespace shogun
{
/** @brief A File access base class.
 *
 * A file is assumed to be a seekable raw data stream.
 *
 * \sa CAsciiFile
 * \sa CBinaryFile
 * \sa CHDF5File
 *
 */
class CSerialFile :public CSGObject
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

	virtual bool write_type_wrapped(
		const TSGDataType* type, const void* param, const char* name,
		const char* prefix) = 0;
	virtual bool read_type_wrapped(
		const TSGDataType* type, void* param, const char* name,
		const char* prefix) = 0;

public:
	/** default constructor */
	explicit CSerialFile(void);

	/** constructor
	 *
	 * @param f already opened file
	 */
	explicit CSerialFile(FILE* f, char rw);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	explicit CSerialFile(char* fname, char rw='r');

	/** default destructor */
	virtual ~CSerialFile(void);

	/** @return object name */
	inline virtual const char* get_name() const { return "SerialFile"; }

	virtual void close(void);
	virtual bool is_opened(void);

	virtual bool write_type(const TSGDataType* type, const void* param,
							const char* name, const char* prefix);
	virtual bool read_type(const TSGDataType* type, void* param,
						   const char* name, const char* prefix);
};
}
#endif // __SERIALFILE_H__
