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
#include <shogun/lib/SGVector.h>
#include <shogun/lib/CircularBuffer.h>

namespace shogun
{
/** @brief Class for buffered reading lines from a ascii file
 */
class CLineReader : public CSGObject
{
public:
	/** default constructor */
	CLineReader();

	/** create object associated with the stream to read
	 *
	 * @param stream readable stream
	 */
	CLineReader(FILE* stream);

	/** create object associated with the stream to read
	 * and specify maximum length of a string that can be read
	 *
	 * @param stream readable stream
	 * @param buffer_size size of internal buffer
	 */
	CLineReader(FILE* stream, int32_t max_string_length);

	/** deconstructor */
	~CLineReader();
	
	/** check for next line in the stream
	 * this method can read data from the stream and
	 * there is no warranty that after reading the caret will 
	 * set at the beginning of a new line
	 *
	 * @return true if there is next line, false - otherwise
	 */
	bool has_next_line();

	/** get read line from the buffer into SGVector
	 *
	 * @return SGVector that contains line
	 */
	SGVector<char> get_next_line();

	/** @return object name */
	virtual const char* get_name() const { return "CLineReader"; }

private:
	/** read one line into buffer
	 *
	 * @return length of line
 	 */
	int32_t read_line(char delimiter);

	/** copy chars into SGVector from source
	 *
	 * @param line destination string
	 * @param line_len length of line in source
	 * @param source source array of chars
	 */
	SGVector<char> copy_line(int32_t line_len);

private:
	/** internal buffer for searching */
	CCircularBuffer* m_buffer;

	/** tokenizer */
	CDelimiterTokenizer* m_tokenizer;

	/** readable stream */
	FILE* m_stream;	

	/** maximum length of a line that can be read */
	int32_t m_max_line_length;

	/** length of next line in the buffer */
	int32_t m_next_line_length;	
};

}

#endif /* __LINE_READER_H__ */
