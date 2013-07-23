/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#include <cstdio>
#include <shogun/io/LineReader.h>

using namespace shogun;

CLineReader::CLineReader()
	: m_stream(NULL), m_max_line_length(0), m_next_line_length(-1)
{
	m_buffer=new CCircularBuffer(0);
	m_tokenizer=NULL;
}

CLineReader::CLineReader(FILE* stream, char delimiter)
	: m_stream(stream), m_max_line_length(10*1024*1024), m_next_line_length(-1)
{
	m_buffer=new CCircularBuffer(m_max_line_length);
	m_tokenizer=new CDelimiterTokenizer();
	m_tokenizer->delimiters[delimiter]=1;
	m_buffer->set_tokenizer(m_tokenizer);
}

CLineReader::CLineReader(int32_t max_line_length, FILE* stream, char delimiter)
	: m_stream(stream), m_max_line_length(max_line_length), m_next_line_length(-1)
{
	m_buffer=new CCircularBuffer(m_max_line_length);
	m_tokenizer=new CDelimiterTokenizer();
	m_tokenizer->delimiters[delimiter]=1;
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

	if (feof(m_stream) && m_buffer->num_bytes_contained()<=0)
		return false; // nothing to read

	return true;	
}

SGVector<char> CLineReader::get_next_line()
{
	SGVector<char> line;	

	m_next_line_length=read_line();
	if (m_next_line_length==-1)
		line=SGVector<char>();
	else
		line=copy_line(m_next_line_length);

	return line;
}

void CLineReader::set_delimiter(char delimiter)
{
	m_tokenizer->delimiters[delimiter]=1;
}

void CLineReader::clear_delimiters()
{
	m_tokenizer->clear_delimiters();
}

int32_t CLineReader::read_line()
{
	int32_t line_end=0;
	int32_t bytes_to_skip=0;
	int32_t bytes_to_read=0;

	while (1)
	{
		line_end+=m_buffer->next_token_idx(bytes_to_skip)-bytes_to_skip;

		if (m_buffer->num_bytes_contained()!=0 && line_end<m_buffer->num_bytes_contained())
			return line_end;
		else if (m_buffer->available()==0)
			return -1; // we need some limit in case file does not contain delimiter

		// if there is no delimiter in buffer
		// try get more data from stream
		// and write it into buffer
		if (m_buffer->available() < m_max_line_length)
			bytes_to_read=m_buffer->available();
		else
			bytes_to_read=m_max_line_length;

		if (feof(m_stream))
			return line_end;
		else
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
