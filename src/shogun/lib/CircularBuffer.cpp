/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evgeniy Andreev, Thoralf Klein, Weijie Lin, Bjoern Esser, 
 *          Viktor Gal
 */

#include <shogun/lib/CircularBuffer.h>
#include <shogun/lib/Tokenizer.h>
#include <shogun/io/SGIO.h>

#include <cstdio>
#include <cstring>
#include <utility>

using namespace shogun;

CircularBuffer::CircularBuffer()
{
	init();
}

CircularBuffer::CircularBuffer(int32_t buffer_size)
{
	init();

	m_buffer=SGVector<char>(buffer_size);
	m_buffer_end=m_buffer.vector+m_buffer.vlen;

	m_begin_pos=m_buffer.vector;
	m_end_pos=m_begin_pos;

	m_bytes_available=m_buffer.vlen;
}

CircularBuffer::~CircularBuffer()
{
	
}

void CircularBuffer::set_tokenizer(std::shared_ptr<Tokenizer> tokenizer)
{
	
	
	m_tokenizer=std::move(tokenizer);
}

int32_t CircularBuffer::push(SGVector<char> source)
{
	if (source.vector==NULL || source.vlen==0)
	{
		error("CircularBuffer::push(SGVector<char>):: Invalid parameters! Source shouldn't be NULL or zero sized");
		return -1;
	}

	int32_t bytes_to_write;
	if (source.vlen>m_bytes_available)
		bytes_to_write=m_bytes_available;
	else
		bytes_to_write=source.vlen;

	if (bytes_to_write==0)
		return 0;

	// determine which part of the memory block is free to read
	if (m_end_pos>=m_begin_pos)
	{
		auto bytes_to_memory_end=m_buffer.vlen-std::distance(m_buffer.vector, m_end_pos);
		if (bytes_to_memory_end<bytes_to_write)
		{
			// we need write as at end of memory block and at begin
			// because logical structure of buffer is ring
			int32_t first_chunk_size=bytes_to_memory_end;
			int32_t second_chunk_size=bytes_to_write-first_chunk_size;

			bytes_to_write=append_chunk(source.vector, first_chunk_size, false);
			bytes_to_write+=append_chunk(source.vector+first_chunk_size, second_chunk_size, true);
		}
		else
		{
			bytes_to_write=append_chunk(source.vector, bytes_to_write, false);
		}
	}
	else
	{
		bytes_to_write=append_chunk(source.vector, bytes_to_write, false);
	}

	return bytes_to_write;
}

int32_t CircularBuffer::push(FILE* source, int32_t source_size)
{
	if (source==NULL || source_size==0)
	{
		error("CircularBuffer::push(FILE*, int32_t):: Invalid parameters! Source shouldn't be NULL or zero sized");
		return -1;
	}

	int32_t bytes_to_write;
	if (source_size>m_bytes_available)
		bytes_to_write=m_bytes_available;
	else
		bytes_to_write=source_size;

	if (bytes_to_write==0)
		return 0;

	// determine which part of the memory block is free to read
	if (m_end_pos>=m_begin_pos)
	{
		int32_t bytes_to_memory_end=m_buffer.vlen-std::distance(m_buffer.vector, m_end_pos);
		if (bytes_to_memory_end<bytes_to_write)
		{
			// we need write as at end of memory block and at begin
			// because logical structure of buffer is ring
			int32_t first_chunk_size=bytes_to_memory_end;
			int32_t second_chunk_size=bytes_to_write-first_chunk_size;

			bytes_to_write=append_chunk(source, first_chunk_size, false);
			bytes_to_write+=append_chunk(source, second_chunk_size, true);
		}
		else
		{
			bytes_to_write=append_chunk(source, bytes_to_write, false);
		}
	}
	else
	{
		bytes_to_write=append_chunk(source, bytes_to_write, false);
	}

	return bytes_to_write;
}

SGVector<char> CircularBuffer::pop(int32_t num_bytes)
{
	SGVector<char> result;

	int32_t bytes_to_read;
	if (num_bytes>m_bytes_count)
		bytes_to_read=m_bytes_count;
	else
		bytes_to_read=num_bytes;

	if (bytes_to_read==0)
		return 0;

	// determine which part of the memory block will be read
	if (m_begin_pos>=m_end_pos)
	{
		int32_t bytes_to_memory_end=m_buffer.vlen-(m_begin_pos-m_buffer.vector);
		if (bytes_to_memory_end<bytes_to_read)
		{
			// read continious block from end of memory and from begin
			int32_t first_chunk_size=bytes_to_memory_end;
			int32_t second_chunk_size=bytes_to_read-first_chunk_size;

			detach_chunk(&result.vector, &result.vlen, 0, first_chunk_size, false);
			detach_chunk(&result.vector, &result.vlen, first_chunk_size, second_chunk_size, true);
		}
		else
		{
			detach_chunk(&result.vector, &result.vlen, 0, bytes_to_read, false);
		}
	}
	else
	{
		detach_chunk(&result.vector, &result.vlen, 0, bytes_to_read, false);
	}

	return result;
}

bool CircularBuffer::has_next()
{
	if (m_tokenizer==NULL)
	{
		error("CircularBuffer::has_next():: Tokenizer is not initialized");
		return false;
	}

	if (m_bytes_count==0)
		return false;

	auto head_length=std::distance(m_begin_pos, m_buffer_end);

	// determine position of finder pointer in memory block
	if (m_last_idx<head_length)
	{
		if (m_end_pos>=m_begin_pos && m_bytes_available!=0)
		{
			return has_next_locally(m_begin_pos+m_last_idx, m_end_pos);
		}
		else
		{
			bool temp=false;
			temp=has_next_locally(m_begin_pos+m_last_idx, m_buffer_end);
			return (temp > 0)
				? temp
				: has_next_locally(m_buffer.vector+m_last_idx-head_length, m_end_pos);
		}
	}
	else
	{
		return has_next_locally(m_buffer.vector+m_last_idx-head_length, m_end_pos);
	}

	return false;
}

index_t CircularBuffer::next_token_idx(index_t &start)
{
	index_t end;

	if (m_tokenizer==NULL)
	{
		error("CircularBuffer::next_token_idx(index_t&):: Tokenizer is not initialized");
		return 0;
	}

	if (m_bytes_count==0)
		return m_bytes_count;

	auto tail_length=std::distance(m_buffer.vector, m_end_pos);
	auto head_length=std::distance(m_begin_pos, m_buffer_end);

	// determine position of finder pointer in memory block
	if (m_last_idx<head_length)
	{
		if (m_end_pos>=m_begin_pos && m_bytes_available!=0)
		{
			end=next_token_idx_locally(start, m_begin_pos+m_last_idx, m_end_pos);
			if (end<=m_bytes_count)
				return end;
		}
		else
		{
			index_t temp_start;

			// in this case we should find first at end of memory block
			end=next_token_idx_locally(start, m_begin_pos+m_last_idx, m_buffer_end);

			if (end<head_length)
				return end;

			// and then at begin
			end=next_token_idx_locally(temp_start, m_buffer.vector+m_last_idx-head_length, m_end_pos);

			if (start>=head_length)
				start=temp_start;

			return end;
		}
	}
	else
	{
		end=next_token_idx_locally(start, m_buffer.vector+m_last_idx-head_length, m_end_pos);
		if (end-head_length<=tail_length)
			return end;
	}

	start=0;
	return start;
}

void CircularBuffer::skip_characters(int32_t num_chars)
{
	auto head_length = std::distance(m_begin_pos, m_buffer_end);
	if (head_length >= num_chars)
		move_pointer(&m_begin_pos, m_begin_pos+num_chars);
	else
		move_pointer(&m_begin_pos, m_buffer.vector+num_chars-head_length);

	m_last_idx-=num_chars;
	if (m_last_idx<0)
		m_last_idx=0;

	m_bytes_available+=num_chars;
	m_bytes_count-=num_chars;
}

void CircularBuffer::clear()
{
	m_begin_pos=m_buffer.vector;
	m_end_pos=m_begin_pos;

	m_last_idx=0;
	m_bytes_available=m_buffer.vlen;
	m_bytes_count=0;
}

void CircularBuffer::init()
{
	m_buffer=SGVector<char>();
	m_buffer_end=NULL;
	m_tokenizer=NULL;

	m_begin_pos=NULL;
	m_end_pos=NULL;

	m_last_idx=0;
	m_bytes_available=0;
	m_bytes_count=0;
}

int32_t CircularBuffer::append_chunk(const char* source, int32_t source_size,
					bool from_buffer_begin)
{
	if (source==NULL || source_size==0)
	{
		error("CircularBuffer::append_chunk(const char*, int32_t, bool):: Invalid parameters!\
				Source shouldn't be NULL or zero sized");
		return -1;
	}

	if (from_buffer_begin)
		m_end_pos=m_buffer.vector;

	sg_memcpy(m_end_pos, source, source_size);
	move_pointer(&m_end_pos, m_end_pos+source_size);

	m_bytes_available-=source_size;
	m_bytes_count+=source_size;

	return source_size;
}

int32_t CircularBuffer::append_chunk(FILE* source, int32_t source_size,
					bool from_buffer_begin)
{
	int32_t actually_read=fread(m_end_pos, sizeof(char), source_size, source);

	if (from_buffer_begin && actually_read==source_size)
		m_end_pos=m_buffer.vector;
	move_pointer(&m_end_pos, m_end_pos+actually_read);

	m_bytes_available-=actually_read;
	m_bytes_count+=actually_read;

	return actually_read;
}

void CircularBuffer::detach_chunk(char** dest, int32_t* dest_size, int32_t dest_offset, int32_t num_bytes,
					bool from_buffer_begin)
{
	if (dest==NULL || dest_size==NULL)
	{
		error("CircularBuffer::detach_chunk(...):: Invalid parameters! Pointers are NULL");
		return;
	}

	if (*dest==NULL)
	{
		*dest=SG_MALLOC(char, num_bytes+dest_offset);
		*dest_size=num_bytes+dest_offset;
	}

	if (*dest_size<num_bytes+dest_offset)
	{
		*dest=SG_REALLOC(char, *dest, *dest_size, num_bytes+dest_offset);
		*dest_size=num_bytes+dest_offset;
	}

	if (from_buffer_begin)
		m_begin_pos=m_buffer.vector;

	sg_memcpy(*dest+dest_offset, m_begin_pos, num_bytes);
	move_pointer(&m_begin_pos, m_begin_pos+num_bytes);

	m_last_idx-=num_bytes;
	if (m_last_idx<0)
		m_last_idx=0;

	m_bytes_available+=num_bytes;
	m_bytes_count-=num_bytes;
}

bool CircularBuffer::has_next_locally(char* part_begin, char* part_end)
{
	auto num_bytes_to_search=std::distance(part_begin, part_end);

	SGVector<char> buffer_part(part_begin, num_bytes_to_search, false);
	m_tokenizer->set_text(buffer_part);

	return m_tokenizer->has_next();
}

index_t CircularBuffer::next_token_idx_locally(index_t &start, char* part_begin, char* part_end)
{
	index_t end=0;
	auto num_bytes_to_search=std::distance(part_begin, part_end);
	if (num_bytes_to_search<=0)
	{
		start=0;
		return m_last_idx;
	}

	SGVector<char> buffer_part(part_begin, num_bytes_to_search, false);
	m_tokenizer->set_text(buffer_part);

	end=m_tokenizer->next_token_idx(start);

	start+=m_last_idx;
	m_last_idx+=end;

	if (end==num_bytes_to_search)
		return m_last_idx;
	else
		return m_last_idx++;
}

void CircularBuffer::move_pointer(char** pointer, char* new_position)
{
	*pointer = (new_position >= m_buffer_end)
		? m_buffer.vector
		: new_position;
}
