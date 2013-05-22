/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#ifndef __LINE_READER_H__
#define __LINE_READER_H__

#include <shogun/base/SGObject.h>

namespace shogun
{
/** @brief Tools for reading lines from ascii files
 */
class CLineReader : public CSGObject
{
public:
	/**
	 * getdelim() implementation.
	 *
	 * Reads upto delimiter from stream into a dynamically
	 * expanding buffer, lineptr, and returns the number of
	 * characters read.
	 * See specification of standard getdelim() for details.
	 *
	 * @param lineptr Buffer to store the string.
	 * @param n Size of buffer.
	 * @param delimiter Delimiter upto (and including) which to read.
	 * @param stream FILE pointer to read from.
	 *
	 * @return Number of bytes read.
	 */
	static int32_t getdelim(char **lineptr, int32_t *n, char delimiter, FILE* stream);

	/**
	 * getline() implementation.
	 *
	 * Reads upto and including the first \n from the file.
	 * @param lineptr Buffer
	 * @param n Size of buffer
	 * @param stream FILE pointer to read from
	 *
	 * @return Number of bytes read
	 */
	static int32_t getline(char **lineptr, int32_t *n, FILE *stream);

	/** @return object name */
	virtual const char* get_name() const { return "CLineReader"; }

public:
	/* minimal size of reading buffer */
	static const int32_t chunk_size = 1048576; 
};

}

#endif /* __LINE_READER_H__ */
