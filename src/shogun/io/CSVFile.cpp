/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#include <shogun/io/CSVFile.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

using namespace shogun;

CCSVFile::CCSVFile()
{
	init();
}

CCSVFile::CCSVFile(FILE* f, const char* name, char delimiter, char quote) :
	CFile(f, name)
{
	init(delimiter, quote);
}

CCSVFile::CCSVFile(const char* fname, char rw, const char* name, char delimiter, char quote) :
	CFile(fname, rw, name)
{
	init(delimiter, quote);
}

CCSVFile::~CCSVFile()
{
	SG_UNREF(m_tokenizer);
	SG_UNREF(m_line_tokenizer);
	SG_UNREF(m_parser);
	SG_UNREF(m_line_reader);
}

void CCSVFile::set_fortran_order()
{
	m_fortran_order=true;
}

void CCSVFile::set_c_order()
{
	m_fortran_order=false;
}

void CCSVFile::skip_lines(int32_t num_lines)
{
	for (int32_t i=0; i<num_lines; i++)
		m_line_reader->skip_line();
}

void CCSVFile::init()
{
	m_tokenizer=new CDelimiterTokenizer();
	m_line_tokenizer=new CDelimiterTokenizer();
	m_parser=new CParser();
	m_line_reader=new CLineReader();
}

void CCSVFile::init(char delimiter, char quote)
{
	m_delimiter=delimiter;

	m_tokenizer=new CDelimiterTokenizer(true);
	m_tokenizer->delimiters[delimiter]=1;
	m_tokenizer->delimiters[' ']=1;

	m_line_tokenizer=new CDelimiterTokenizer(true);
	m_line_tokenizer->delimiters['\n']=1;

	m_parser=new CParser();
	m_parser->set_tokenizer(m_tokenizer);

	m_line_reader=new CLineReader(file, m_line_tokenizer);	
}

#define GET_VECTOR(fname, read_func, sg_type) \
void CCSVFile::fname(sg_type*& vector, int32_t& len) \
{ \
	if (!m_line_reader->has_next()) \
		return; \
	\
	SGVector<char> line=m_line_reader->read_line(); \
	m_tokenizer->set_text(line); \
	\
	len=0; \
	while (m_tokenizer->has_next()) \
	{ \
		index_t temp_start=0; \
		m_tokenizer->next_token_idx(temp_start); \
	\
		len++; \
	} \
	\
	m_parser->set_text(line); \
	vector=SG_MALLOC(sg_type, len); \
	for (int32_t i=0; i<len; i++) \
	{ \
		vector[i]=m_parser->read_func(); \
	} \
	\
}

GET_VECTOR(get_vector, read_char, int8_t)
GET_VECTOR(get_vector, read_byte, uint8_t)
GET_VECTOR(get_vector, read_char, char)
GET_VECTOR(get_vector, read_int, int32_t)
GET_VECTOR(get_vector, read_uint, uint32_t)
GET_VECTOR(get_vector, read_short_real, float32_t)
GET_VECTOR(get_vector, read_real, float64_t)
GET_VECTOR(get_vector, read_long_real, floatmax_t)
GET_VECTOR(get_vector, read_short, int16_t)
GET_VECTOR(get_vector, read_word, uint16_t)
GET_VECTOR(get_vector, read_long, int64_t)
GET_VECTOR(get_vector, read_ulong, uint64_t)
#undef GET_VECTOR

#define GET_MATRIX(fname, read_func, sg_type) \
void CCSVFile::fname(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec) \
{ \
	\
	int32_t nlines=0; \
	int32_t ntokens=-1; \
	\
	int32_t last_idx=0; \
	SGVector<sg_type> line_memory(true); \
	SGVector<sg_type> temp(true); \
	while(m_line_reader->has_next()) \
	{ \
		read_func(temp.vector, temp.vlen); \
		if (ntokens<0) \
			ntokens=temp.vlen; \
		\
		if (ntokens!=temp.vlen) \
			return; \
		\
		line_memory.resize_vector(last_idx+temp.vlen); \
		for (int32_t i=0; i<temp.vlen; i++) \
		{ \
			line_memory[i+last_idx]=temp[i]; \
		} \
		last_idx+=temp.vlen; \
		\
		nlines++; \
	} \
	\
	if (m_fortran_order) \
	{ \
		num_feat=nlines; \
		num_vec=ntokens; \
		SGVector<sg_type>::convert_to_matrix(matrix, num_vec, num_feat, line_memory.vector, line_memory.vlen, true); \
	} \
	else \
	{ \
		num_feat=ntokens; \
		num_vec=nlines; \
		SGVector<sg_type>::convert_to_matrix(matrix, num_vec, num_feat, line_memory.vector, line_memory.vlen, false); \
	} \
} \

GET_MATRIX(get_matrix, get_vector, int8_t)
GET_MATRIX(get_matrix, get_vector, uint8_t)
GET_MATRIX(get_matrix, get_vector, char)
GET_MATRIX(get_matrix, get_vector, int32_t)
GET_MATRIX(get_matrix, get_vector, uint32_t)
GET_MATRIX(get_matrix, get_vector, int64_t)
GET_MATRIX(get_matrix, get_vector, uint64_t)
GET_MATRIX(get_matrix, get_vector, float32_t)
GET_MATRIX(get_matrix, get_vector, float64_t)
GET_MATRIX(get_matrix, get_vector, floatmax_t)
GET_MATRIX(get_matrix, get_vector, int16_t)
GET_MATRIX(get_matrix, get_vector, uint16_t)
#undef GET_MATRIX

#define SET_VECTOR(fname, format, sg_type) \
void CCSVFile::fname(const sg_type* vector, int32_t len) \
{ \
	for (int32_t i=0; i<len; i++) \
	{ \
		fprintf(file, #format "%c", vector[i], m_delimiter); \
	} \
	fprintf(file, "\n"); \
}

SET_VECTOR(set_vector, %hhd, int8_t)
SET_VECTOR(set_vector, %hhu, uint8_t)
SET_VECTOR(set_vector, %hhd, char)
SET_VECTOR(set_vector, %d, int32_t)
SET_VECTOR(set_vector, %u, uint32_t)
SET_VECTOR(set_vector, %ld, int64_t)
SET_VECTOR(set_vector, %lu, uint64_t)
SET_VECTOR(set_vector, %e, float32_t)
SET_VECTOR(set_vector, %e, float64_t)
SET_VECTOR(set_vector, %Le, floatmax_t)
SET_VECTOR(set_vector, %hd, int16_t)
SET_VECTOR(set_vector, %hu, uint16_t)
#undef SET_VECTOR

#define SET_MATRIX(fname, format, sg_type) \
void CCSVFile::fname(const sg_type* matrix, int32_t num_feat, int32_t num_vec) \
{ \
	if (m_fortran_order) \
	{ \
		for (int32_t i=0; i<num_feat; i++) \
		{ \
			for (int32_t j=0; j<num_vec; j++) \
				fprintf(file, #format "%c", matrix[j+i*num_vec], m_delimiter); \
			fprintf(file, "\n"); \
		} \
	} \
	else \
	{ \
		for (int32_t i=0; i<num_vec; i++) \
		{ \
			for (int32_t j=0; j<num_feat; j++) \
				fprintf(file, #format "%c", matrix[i+j*num_vec], m_delimiter); \
			fprintf(file, "\n"); \
		} \
	} \
}

SET_MATRIX(set_matrix, %hhd, int8_t)
SET_MATRIX(set_matrix, %hhu, uint8_t)
SET_MATRIX(set_matrix, %hhd, char)
SET_MATRIX(set_matrix, %d, int32_t)
SET_MATRIX(set_matrix, %u, uint32_t)
SET_MATRIX(set_matrix, %ld, int64_t)
SET_MATRIX(set_matrix, %lu, uint64_t)
SET_MATRIX(set_matrix, %e, float32_t)
SET_MATRIX(set_matrix, %e, float64_t)
SET_MATRIX(set_matrix, %Le, floatmax_t)
SET_MATRIX(set_matrix, %hd, int16_t)
SET_MATRIX(set_matrix, %hu, uint16_t)
#undef SET_MATRIX
