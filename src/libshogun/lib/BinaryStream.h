/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __BINARYSTREAM_H__
#define __BINARYSTREAM_H__

#include "lib/io.h"
#include "base/SGObject.h"

#include <stdio.h>
#include <string.h>

/** @brief memory mapped emulation via binary streams (files)
*
* Implements memory mapped file emulation (\sa CMemoryMappedFile) via standard
* file operations like fseek, fread etc
*/
template <class T> class CBinaryStream : public CSGObject
{
	public:
		/** constructor
		 *
		 * open a file for read mode
		 *
		 * @param fname name of file, zero terminated string
		 * @param flag determines read or read write mode (currently only 'r' is supported)
		 */
		CBinaryStream(const char* fname, const char* flag="r")
		: CSGObject()
		{
			fd = fopen(fname, flag);
			if (!fd)
				SG_ERROR("Error opening file\n");

			struct stat sb;
			if (stat(fname, &sb) == -1)
				SG_ERROR("Error determining file size\n");

			length = sb.st_size;
			SG_DEBUG("Opened file '%s' of size %ld byte\n", fname, length);
		}

		/** destructor */
		virtual ~CBinaryStream()
		{
			fclose(fd);
		}

		/** get the number of objects of type T cointained in the file 
		 *
		 * @return length of file
		 */
		uint64_t get_length()
		{
			return length/sizeof(T);
		}

		/** get the size of the file in bytes
		 *
		 * @return size of file in bytes
		 */
		uint64_t get_size()
		{
			return length;
		}

		/** get next line from file
		 *
		 * The returned line may be modfied in case the file was opened
		 * read/write. It is otherwise read-only.
		 *
		 * @param len length of line (returned via reference)
		 * @param offs offset to be passed for reading next line, should be 0
		 * 			initially (returned via reference)
		 *
		 * @return line (NOT ZERO TERMINATED)
		 */
		char* get_line(uint64_t& len, uint64_t& offs)
		{
			return NULL;
		}

		/** count the number of lines in a file
		 *
		 * @return number of lines
		 */
		int32_t get_num_lines()
		{
			return 0;
		}

		/** operator overload for file read only access
		 *
		 * @param index index
		 * @return element at index
		 */
		inline T operator[](int32_t index) const
		{

			if (fseek(fd, ((long) sizeof(T))*((long) index), SEEK_SET) != 0)
				SG_ERROR("Error seeking to %ld\n", sizeof(T)*((int64_t) index));

			T ptr;

			if ( fread(&ptr, sizeof(T), 1, fd) != 1)
				SG_ERROR("Error calling fread\n");

			return T;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "BinaryStream"; }

	protected:
		/** file descriptor */
		FILE* fd;
		/** size of file */
		uint64_t length;
		/** mode */
		char rw;
};
#endif // BINARY_STREAM
