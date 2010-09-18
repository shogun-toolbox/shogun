/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __BINARY_FILE_H__
#define __BINARY_FILE_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/io.h>
#include <shogun/lib/SimpleFile.h>

namespace shogun
{
/** @brief A Binary file access class.
 *
 * A file consists of a SG00 fourcc header then an alternation of a type header and
 * data. The current implementation is capable of storing only a single
 * header/data type. Multiple headers are currently not implemented.
 */
class CBinaryFile: public CFile
{
public:
	/** constructor
	 *
	 * @param f already opened file
	 */
	CBinaryFile(FILE* f, char rw);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	CBinaryFile(char* fname, char rw='r');

	/** default destructor */
	virtual ~CBinaryFile();

	/** @return object name */
	inline virtual const char* get_name() const { return "BinaryFile"; }
};
}
#endif //__BINARY_FILE_H__
