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

#ifndef __FILE_H__
#define __FILE_H__

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
class CFile :public CSGObject
{
protected:
	bool is_task_warn(char rw);
	bool false_warn(const char* prefix, const char* name);

	virtual void close(void);

	/** file object */
	FILE* file;
	/** task */
	char task;
	/** name of the handled file */
	char* filename;

public:
	/** default constructor */
	explicit CFile(void);

	/** constructor
	 *
	 * @param f already opened file
	 */
	explicit CFile(FILE* f, char rw);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	explicit CFile(char* fname, char rw='r');

	/** default destructor */
	virtual ~CFile(void);

	/** @return object name */
	inline virtual const char* get_name() const { return "File"; }

	virtual bool write_type(const TSGDataType* type, const void* param,
							const char* name, const char* prefix) = 0;
	virtual bool read_type(const TSGDataType* type, void* param,
						   const char* name, const char* prefix) = 0;
};
}
#endif // __FILE_H__
