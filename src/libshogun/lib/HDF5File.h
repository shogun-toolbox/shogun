/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __HDF5_FILE_H__
#define __HDF5_FILE_H__

#include "lib/common.h"
#include "lib/File.h"

#ifdef HAVE_HDF5

namespace shogun
{
class CHDF5File :public CFile
{
public:
		/** default constructor */
	explicit CHDF5File(void);

	/** constructor
	 *
	 * @param f already opened file
	 */
	explicit CHDF5File(FILE* f, char rw);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	explicit CHDF5File(char* fname, char rw='r');

	/** default destructor */
	virtual ~CHDF5File();

	/** @return object name */
	inline virtual const char* get_name() const { return "AsciiFile"; }

	virtual bool write_type(const TSGDataType* type, const void* param,
							const char* name, const char* prefix);
	virtual bool read_type(const TSGDataType* type, void* param,
						   const char* name, const char* prefix);
};
}

#endif //  HAVE_HDF5
#endif //__HDF5_FILE_H__
