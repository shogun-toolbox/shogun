/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evgeniy Andreev, Heiko Strathmann, Yuyu Zhang, Thoralf Klein,
 *          Fernando Iglesias, Bjoern Esser
 */

#ifndef __CIRCULARBUFFER_H_
#define __CIRCULARBUFFER_H_

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{
class Tokenizer;

/** @brief Implementation of circular buffer
 * This buffer has logical structure such as queue (FIFO).
 * But this queue is cyclic: tape, ends of which are connected,
 * just instead tape there is block of physical memory.
 * So, if you push big block of data it can be situated
 * both at the end and the begin of buffer's memory.
 *
 * w: http://en.wikipedia.org/wiki/Circular_buffer
 */
class CircularBuffer : public SGObject
{
public:
	/** default constructor */
	CircularBuffer();

	/** constructor
	 *
	 * @param buffer_size size of buffer
	 */
	CircularBuffer(int32_t buffer_size);

	/** destructor */
	~CircularBuffer();

	/** set tokenizer
	 *
	 * @param tokenizer tokenizer
	 */
	void set_tokenizer(std::shared_ptr<Tokenizer> tokenizer);

	/** push data into buffer from memory block
	 *
	 * @param source source data
	 * @return number of bytes written
	 */
	int32_t push(SGVector<char> source);

	/** push data into buffer from file
	 *
	 * @param source source file (stream)
	 * @param source_size size of data to read
	 */
	int32_t push(FILE* source, int32_t source_size);

	/** remove characters from buffer
	 *
	 * @param num_chars number of characters that should be removed
	 * @return characters removed
	 */
	SGVector<char> pop(int32_t num_chars);

	/** returns true or false based on whether
	 * there exists another token in the text
	 *
	 * @return if another token exists
	 */
	bool has_next();

	/** method that returns the indices, start and end, of
	 *  the next token in buffer
	 *
	 * @param start token's starting index
	 * @return token's ending index (inclusive)
	 */
	index_t next_token_idx(index_t &start);

	/** remove characters from buffer
	 * similar with pop(), but but does not return a string.
	 *
	 * @param num_chars number of characters that should be skipped
	 */
	void skip_characters(int32_t num_chars);

	/** @return number of free bytes in buffer */
	int32_t available() const
	{
		return m_bytes_available;
	}

	/** @return number of bytes contained in buffer */
	int32_t num_bytes_contained() const
	{
		return m_bytes_count;
	}

	/** clear buffer */
	void clear();

	/** @return object name */
	virtual const char* get_name() const { return "CircularBuffer"; }

private:
	/** class initialization */
	void init();

	/** append memory block to buffer */
	int32_t append_chunk(const char* source, int32_t source_size,
					bool from_buffer_begin);

	/** append data from file to buffer */
	int32_t append_chunk(FILE* source, int32_t source_size,
					bool from_buffer_begin);

	/** detach memory block from buffer */
	void detach_chunk(char** dest, int32_t* dest_size, int32_t dest_offset, int32_t num_bytes,
					bool from_buffer_begin);

	/** returns true or false based on whether
	 * there exists another token in the text
	 */
	bool has_next_locally(char* begin, char* end);

	/** method that returns the indices, start and end, of
	 *  the next token in part of buffer
	 */
	index_t next_token_idx_locally(index_t &start, char* begin, char* end);

	/** move pointer to another position */
	void move_pointer(char** pointer, char* new_position);

private:
	/** internal memory */
	SGVector<char> m_buffer;

	/** pointer to end of buffer's memory */
	char* m_buffer_end;

	/** begin of buffer */
	char* m_begin_pos;

	/** end of buffer */
	char* m_end_pos;

	/** tokenizer */
	std::shared_ptr<Tokenizer> m_tokenizer;

	/** position at which the search starts */
	index_t m_last_idx;

	/** number of free bytes */
	int32_t m_bytes_available;

	/** number of bytes contained */
	int32_t m_bytes_count;
};

}

#endif /* _CIRCULARBUFFER_H_ */
