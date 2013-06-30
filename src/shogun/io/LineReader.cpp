/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#include <cstdio>_
#include <shogun/io/LineReader.h>

using namespace shogun;

CLineReader::CLineReader()
	: m_stream(NULL), m_max_line_length(0), m_next_line_length(-1)
{
	m_buffer=new CCircularBuffer(0);
}

CLineReader::CLineReader(FILE* stream)
	: m_stream(stream), m_max_line_length(10*1024*1024), m_next_line_length(-1)
{
	m_buffer=new CCircularBuffer(m_max_line_length);
	m_tokenizer=new CDelimiterTokenizer();
	m_tokenizer->delimiters['\n']=1;
	m_buffer->set_tokenizer(m_tokenizer);
}

CLineReader::CLineReader(FILE* stream, int32_t max_line_length)
	: m_stream(stream), m_max_line_length(max_line_length), m_next_line_length(-1)
{
	m_buffer=new CCircularBuffer(m_max_line_length);
	m_tokenizer=new CDelimiterTokenizer();
	m_tokenizer->delimiters['\n']=1;
	m_buffer->set_tokenizer(m_tokenizer);
}

CLineReader::~CLineReader()
{
	SG_UNREF(m_tokenizer);
	SG_UNREF(m_buffer);
}

bool CLineReader::has_next_line()
{
	if (m_stream==NULL || m_max_line_length==0)
	{
		SG_ERROR("Class is not initialized");
		return false;
	}

	if (ferror(m_stream))
	{
		SG_ERROR("Error reading file");
		return false;
	}

	if (feof(m_stream) && m_buffer->num_bytes_contained()==0)
		return false; // nothing to read

	return true;	
}

SGVector<char> CLineReader::get_next_line()
{
	SGVector<char> line;
	
	m_next_line_length=read_line('\n');
	if (m_next_line_length==-1)
		line=SGVector<char>();
	else
		line=copy_line(m_next_line_length);

	return line;
}

int32_t CLineReader::read_line(char delimiter)
{
	int32_t line_end=0;
	int32_t bytes_to_skip=0;
	int32_t bytes_to_read=0;

	while (1)
	{
		line_end+=m_buffer->next_token_idx(bytes_to_skip)-bytes_to_skip;

		if (m_buffer->num_bytes_contained()!=0 && line_end<m_buffer->num_bytes_contained())
		{
			return line_end;
			//m_buffer->skip_characters(bytes_to_skip);
			//return line_end-bytes_to_skip;
		}
		else if (m_buffer->available()==0)
			return -1; // we need some limit in case file does not contain delimiter

		// if there is no delimiter in buffer
		// try get more data from stream
		// and write it into buffer
		if (m_buffer->available() < m_max_line_length)
			bytes_to_read=m_buffer->available();
		else
			bytes_to_read=m_max_line_length;

		m_buffer->push(m_stream, bytes_to_read);		
		if (ferror(m_stream))
		{
			SG_ERROR("Error reading file");
			return -1;
		}
	}	
}

SGVector<char> CLineReader::copy_line(int32_t line_len)
{
	SGVector<char> line;

	if (line_len==0)
		line=SGVector<char>();
	else
		line=m_buffer->pop(line_len);

	m_buffer->skip_characters(1);

	return line;
}
