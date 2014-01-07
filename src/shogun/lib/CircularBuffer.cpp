/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#include <lib/CircularBuffer.h>

#include <cstdio>
#include <cstring>

using namespace shogun;

CCircularBuffer::CCircularBuffer()
{
	init();
}

CCircularBuffer::CCircularBuffer(int32_t buffer_size)
{
	init();

	m_buffer=SGVector<char>(buffer_size);
	m_buffer_end=m_buffer.vector+m_buffer.vlen;

	m_begin_pos=m_buffer.vector;
	m_end_pos=m_begin_pos;

	m_bytes_available=m_buffer.vlen;
}

CCircularBuffer::~CCircularBuffer()
{
	SG_UNREF(m_tokenizer);
}

void CCircularBuffer::set_tokenizer(CTokenizer* tokenizer)
{
	SG_REF(tokenizer);
	SG_UNREF(m_tokenizer);
	m_tokenizer=tokenizer;
}

int32_t CCircularBuffer::push(SGVector<char> source)
{
	if (source.vector==NULL || source.vlen==0)
	{
		SG_ERROR("CCircularBuffer::push(SGVector<char>):: Invalid parameters! Source shouldn't be NULL or zero sized\n");
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
		int32_t bytes_to_memory_end=m_buffer.vlen-(m_end_pos-m_buffer.vector);
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

int32_t CCircularBuffer::push(FILE* source, int32_t source_size)
{
	if (source==NULL || source_size==0)
	{
		SG_ERROR("CCircularBuffer::push(FILE*, int32_t):: Invalid parameters! Source shouldn't be NULL or zero sized\n");
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
		int32_t bytes_to_memory_end=m_buffer.vlen-(m_end_pos-m_buffer.vector);
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

SGVector<char> CCircularBuffer::pop(int32_t num_bytes)
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

bool CCircularBuffer::has_next()
{
	if (m_tokenizer==NULL)
	{
		SG_ERROR("CCircularBuffer::has_next():: Tokenizer is not initialized\n");
		return false;
	}

	if (m_bytes_count==0)
		return false;

	int32_t head_length=m_buffer_end-m_begin_pos;

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

			if (temp)
				return temp;

			return has_next_locally(m_buffer.vector+m_last_idx-head_length, m_end_pos);
		}
	}
	else
	{
		return has_next_locally(m_buffer.vector+m_last_idx-head_length, m_end_pos);
	}

	return false;
}

index_t CCircularBuffer::next_token_idx(index_t &start)
{
	index_t end;

	if (m_tokenizer==NULL)
	{
		SG_ERROR("CCircularBuffer::next_token_idx(index_t&):: Tokenizer is not initialized\n");
		return 0;
	}

	if (m_bytes_count==0)
		return m_bytes_count;

	int32_t tail_length=m_end_pos-m_buffer.vector;
	int32_t head_length=m_buffer_end-m_begin_pos;

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

void CCircularBuffer::skip_characters(int32_t num_chars)
{
	move_pointer(&m_begin_pos, m_begin_pos+num_chars);

	m_last_idx-=num_chars;
	if (m_last_idx<0)
		m_last_idx=0;

	m_bytes_available+=num_chars;
	m_bytes_count-=num_chars;
}

void CCircularBuffer::clear()
{
	m_begin_pos=m_buffer.vector;
	m_end_pos=m_begin_pos;

	m_last_idx=0;
	m_bytes_available=m_buffer.vlen;
	m_bytes_count=0;
}

void CCircularBuffer::init()
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

int32_t CCircularBuffer::append_chunk(const char* source, int32_t source_size,
					bool from_buffer_begin)
{
	if (source==NULL || source_size==0)
	{
		SG_ERROR("CCircularBuffer::append_chunk(const char*, int32_t, bool):: Invalid parameters!\
				Source shouldn't be NULL or zero sized\n");
		return -1;
	}

	if (from_buffer_begin)
		m_end_pos=m_buffer.vector;

	memcpy(m_end_pos, source, source_size);
	move_pointer(&m_end_pos, m_end_pos+source_size);

	m_bytes_available-=source_size;
	m_bytes_count+=source_size;

	return source_size;
}

int32_t CCircularBuffer::append_chunk(FILE* source, int32_t source_size,
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

void CCircularBuffer::detach_chunk(char** dest, int32_t* dest_size, int32_t dest_offset, int32_t num_bytes,
					bool from_buffer_begin)
{
	if (dest==NULL || dest_size==NULL)
	{
		SG_ERROR("CCircularBuffer::detach_chunk(...):: Invalid parameters! Pointers are NULL\n");
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

	memcpy(*dest+dest_offset, m_begin_pos, num_bytes);
	move_pointer(&m_begin_pos, m_begin_pos+num_bytes);

	m_last_idx-=num_bytes;
	if (m_last_idx<0)
		m_last_idx=0;

	m_bytes_available+=num_bytes;
	m_bytes_count-=num_bytes;
}

bool CCircularBuffer::has_next_locally(char* part_begin, char* part_end)
{
	int32_t num_bytes_to_search=part_end-part_begin;

	SGVector<char> buffer_part(part_begin, num_bytes_to_search, false);
	m_tokenizer->set_text(buffer_part);

	return m_tokenizer->has_next();
}

index_t CCircularBuffer::next_token_idx_locally(index_t &start, char* part_begin, char* part_end)
{
	index_t end=0;
	int32_t num_bytes_to_search=part_end-part_begin;
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

void CCircularBuffer::move_pointer(char** pointer, char* new_position)
{
	*pointer=new_position;
	if (*pointer>=m_buffer.vector+m_buffer.vlen)
		*pointer=m_buffer.vector;
}
