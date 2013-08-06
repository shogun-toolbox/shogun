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

CCSVFile::CCSVFile(FILE* f, const char* name) :
	CFile(f, name)
{
	init();
	init_with_defaults();
}

CCSVFile::CCSVFile(const char* fname, char rw, const char* name) :
	CFile(fname, rw, name)
{
	init();
	init_with_defaults();
}

CCSVFile::~CCSVFile()
{
	SG_UNREF(m_tokenizer);
	SG_UNREF(m_line_tokenizer);
	SG_UNREF(m_parser);
	SG_UNREF(m_line_reader);
}

void CCSVFile::set_order(csv_data_order order)
{
	m_order=order;
}

void CCSVFile::set_delimiter(char delimiter)
{
	m_tokenizer->delimiters[m_delimiter]=0;
	
	m_delimiter=delimiter;
	m_tokenizer->delimiters[m_delimiter]=1;
}

void CCSVFile::skip_lines(int32_t num_lines)
{
	for (int32_t i=0; i<num_lines; i++)
		m_line_reader->skip_line();
}

void CCSVFile::init()
{
	m_order=C_ORDER;
	m_delimiter=0;

	m_tokenizer=NULL;
	m_line_tokenizer=NULL;
	m_parser=NULL;
	m_line_reader=NULL;
}

void CCSVFile::init_with_defaults()
{
	m_order=C_ORDER;
	m_delimiter=',';

	m_tokenizer=new CDelimiterTokenizer(true);
	m_tokenizer->delimiters[m_delimiter]=1;
	m_tokenizer->delimiters[' ']=1;
	SG_REF(m_tokenizer);

	m_line_tokenizer=new CDelimiterTokenizer(true);
	m_line_tokenizer->delimiters['\n']=1;
	SG_REF(m_line_tokenizer);

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
	int32_t num_feat=0; \
	int32_t num_vec=0; \
	get_matrix(vector, num_feat, num_vec); \
	\
	if (num_feat==1) \
	{ \
		len=num_vec; \
		return; \
	} \
	\
	if (num_vec==1) \
	{ \
		len=num_feat; \
		return; \
	} \
	\
	len=0; \
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
	int32_t nlines=0; \
	int32_t ntokens=-1; \
	int32_t len=0; \
	\
	int32_t last_idx=0; \
	SGVector<sg_type> line_memory(true); \
	\
	while(m_line_reader->has_next()) \
	{ \
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
		\
		if (ntokens<0) \
			ntokens=len; \
		\
		if (ntokens!=len) \
			return; \
		\
		line_memory.resize_vector(last_idx+len); \
		for (int32_t i=0; i<len; i++) \
		{ \
			line_memory[i+last_idx]=m_parser->read_func(); \
		} \
		last_idx+=len; \
		\
		nlines++; \
	} \
	\
	if (m_order==FORTRAN_ORDER) \
	{ \
		num_feat=nlines; \
		num_vec=ntokens; \
		SGVector<sg_type>::convert_to_matrix(matrix, num_vec, num_feat, line_memory.vector, line_memory.vlen, false); \
	} \
	else \
	{ \
		num_feat=ntokens; \
		num_vec=nlines; \
		SGVector<sg_type>::convert_to_matrix(matrix, num_vec, num_feat, line_memory.vector, line_memory.vlen, true); \
	} \
} \

GET_MATRIX(get_matrix, read_char, int8_t)
GET_MATRIX(get_matrix, read_byte, uint8_t)
GET_MATRIX(get_matrix, read_char, char)
GET_MATRIX(get_matrix, read_int, int32_t)
GET_MATRIX(get_matrix, read_uint, uint32_t)
GET_MATRIX(get_matrix, read_short_real, float32_t)
GET_MATRIX(get_matrix, read_real, float64_t)
GET_MATRIX(get_matrix, read_long_real, floatmax_t)
GET_MATRIX(get_matrix, read_short, int16_t)
GET_MATRIX(get_matrix, read_word, uint16_t)
GET_MATRIX(get_matrix, read_long, int64_t)
GET_MATRIX(get_matrix, read_ulong, uint64_t)
#undef GET_MATRIX

#define SET_VECTOR(fname, format, sg_type) \
void CCSVFile::fname(const sg_type* vector, int32_t len) \
{ \
	if (m_order==FORTRAN_ORDER) \
	{ \
		for (int32_t i=0; i<len; i++) \
		{ \
			fprintf(file, "%" format "\n", vector[i]); \
		} \
	} \
	else \
	{ \
		for (int32_t i=0; i<len; i++) \
		{ \
			fprintf(file, "%" format "%c", vector[i], m_delimiter); \
		} \
		fprintf(file, "\n"); \
	} \
}

SET_VECTOR(set_vector, SCNi8, int8_t)
SET_VECTOR(set_vector, SCNu8, uint8_t)
SET_VECTOR(set_vector, SCNu8, char)
SET_VECTOR(set_vector, SCNi32, int32_t)
SET_VECTOR(set_vector, SCNu32, uint32_t)
SET_VECTOR(set_vector, SCNi64, int64_t)
SET_VECTOR(set_vector, SCNu64, uint64_t)
SET_VECTOR(set_vector, "g", float32_t)
SET_VECTOR(set_vector, "lg", float64_t)
SET_VECTOR(set_vector, "Lg", floatmax_t)
SET_VECTOR(set_vector, SCNi16, int16_t)
SET_VECTOR(set_vector, SCNu16, uint16_t)
#undef SET_VECTOR

#define SET_MATRIX(fname, format, sg_type) \
void CCSVFile::fname(const sg_type* matrix, int32_t num_feat, int32_t num_vec) \
{ \
	if (m_order==FORTRAN_ORDER) \
	{ \
		for (int32_t i=0; i<num_feat; i++) \
		{ \
			for (int32_t j=0; j<num_vec; j++) \
				fprintf(file, "%" format "%c", matrix[j+i*num_vec], m_delimiter); \
			fprintf(file, "\n"); \
		} \
	} \
	else \
	{ \
		for (int32_t i=0; i<num_vec; i++) \
		{ \
			for (int32_t j=0; j<num_feat; j++) \
				fprintf(file, "%" format "%c", matrix[i+j*num_vec], m_delimiter); \
			fprintf(file, "\n"); \
		} \
	} \
}

SET_MATRIX(set_matrix, SCNi8, int8_t)
SET_MATRIX(set_matrix, SCNu8, uint8_t)
SET_MATRIX(set_matrix, SCNu8, char)
SET_MATRIX(set_matrix, SCNi32, int32_t)
SET_MATRIX(set_matrix, SCNu32, uint32_t)
SET_MATRIX(set_matrix, SCNi64, int64_t)
SET_MATRIX(set_matrix, SCNu64, uint64_t)
SET_MATRIX(set_matrix, "g", float32_t)
SET_MATRIX(set_matrix, "lg", float64_t)
SET_MATRIX(set_matrix, "Lg", floatmax_t)
SET_MATRIX(set_matrix, SCNi16, int16_t)
SET_MATRIX(set_matrix, SCNu16, uint16_t)
#undef SET_MATRIX
