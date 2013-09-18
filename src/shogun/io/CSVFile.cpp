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

CCSVFile::CCSVFile(int fd, const char* mode, const char* name) :
	CFile(fd, mode, name)
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

void CCSVFile::set_transpose(bool value)
{
	is_data_transposed=value;
}

void CCSVFile::set_delimiter(char delimiter)
{
	m_tokenizer->delimiters[m_delimiter]=0;

	m_delimiter=delimiter;
	m_tokenizer->delimiters[m_delimiter]=1;

	m_tokenizer->delimiters[' ']=1;
}

void CCSVFile::set_lines_to_skip(int32_t num_lines)
{
	m_num_to_skip=num_lines;
}

int32_t CCSVFile::get_stats(int32_t& num_tokens)
{
	int32_t num_lines=0;
	num_tokens=-1;

	while (m_line_reader->has_next())
	{
		if (num_tokens==-1)
		{
			SGVector<char> line=m_line_reader->read_line();
			m_tokenizer->set_text(line);

			num_tokens=0;
			while (m_tokenizer->has_next())
			{
				index_t temp_start=0;
				m_tokenizer->next_token_idx(temp_start);
				num_tokens++;
			}
		}
		else
			m_line_reader->skip_line();
		num_lines++;
	}
	m_line_reader->reset();

	return num_lines;
}

void CCSVFile::init()
{
	is_data_transposed=false;
	m_delimiter=0;
	m_num_to_skip=0;

	m_tokenizer=NULL;
	m_line_tokenizer=NULL;
	m_parser=NULL;
	m_line_reader=NULL;
}

void CCSVFile::init_with_defaults()
{
	is_data_transposed=false;
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

void CCSVFile::skip_lines(int32_t num_lines)
{
	for (int32_t i=0; i<num_lines; i++)
		m_line_reader->skip_line();
}

#define GET_VECTOR(read_func, sg_type) \
void CCSVFile::get_vector(sg_type*& vector, int32_t& len) \
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

GET_VECTOR(read_char, int8_t)
GET_VECTOR(read_byte, uint8_t)
GET_VECTOR(read_char, char)
GET_VECTOR(read_int, int32_t)
GET_VECTOR(read_uint, uint32_t)
GET_VECTOR(read_short_real, float32_t)
GET_VECTOR(read_real, float64_t)
GET_VECTOR(read_long_real, floatmax_t)
GET_VECTOR(read_short, int16_t)
GET_VECTOR(read_word, uint16_t)
GET_VECTOR(read_long, int64_t)
GET_VECTOR(read_ulong, uint64_t)
#undef GET_VECTOR

#define GET_MATRIX(read_func, sg_type) \
void CCSVFile::get_matrix(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec) \
{ \
	int32_t num_lines=0; \
	int32_t num_tokens=-1; \
	int32_t current_line_idx=0; \
	SGVector<char> line; \
	\
	skip_lines(m_num_to_skip); \
	num_lines=get_stats(num_tokens); \
	\
	SG_SET_LOCALE_C; \
	\
	matrix=SG_MALLOC(sg_type, num_lines*num_tokens); \
	skip_lines(m_num_to_skip); \
	while (m_line_reader->has_next()) \
	{ \
		line=m_line_reader->read_line(); \
		m_parser->set_text(line); \
		\
		for (int32_t i=0; i<num_tokens; i++) \
		{ \
			if (!m_parser->has_next()) \
				return; \
			\
			if (!is_data_transposed) \
				matrix[i+current_line_idx*num_tokens]=m_parser->read_func(); \
			else \
				matrix[current_line_idx+i*num_tokens]=m_parser->read_func(); \
		} \
		current_line_idx++; \
	} \
	\
	SG_RESET_LOCALE; \
	\
	if (!is_data_transposed) \
	{ \
		num_feat=num_tokens; \
		num_vec=num_lines; \
	} \
	else \
	{ \
		num_feat=num_lines; \
		num_vec=num_tokens; \
	} \
}

GET_MATRIX(read_char, int8_t)
GET_MATRIX(read_byte, uint8_t)
GET_MATRIX(read_char, char)
GET_MATRIX(read_int, int32_t)
GET_MATRIX(read_uint, uint32_t)
GET_MATRIX(read_short_real, float32_t)
GET_MATRIX(read_real, float64_t)
GET_MATRIX(read_long_real, floatmax_t)
GET_MATRIX(read_short, int16_t)
GET_MATRIX(read_word, uint16_t)
GET_MATRIX(read_long, int64_t)
GET_MATRIX(read_ulong, uint64_t)
#undef GET_MATRIX

#define GET_NDARRAY(read_func, sg_type) \
void CCSVFile::get_ndarray(sg_type*& array, int32_t*& dims, int32_t& num_dims) \
{ \
	SG_NOTIMPLEMENTED \
}

GET_NDARRAY(read_byte, uint8_t)
GET_NDARRAY(read_char, char)
GET_NDARRAY(read_int, int32_t)
GET_NDARRAY(read_short_real, float32_t)
GET_NDARRAY(read_real, float64_t)
GET_NDARRAY(read_short, int16_t)
GET_NDARRAY(read_word, uint16_t)
#undef GET_NDARRAY

#define GET_SPARSE_MATRIX(read_func, sg_type) \
void CCSVFile::get_sparse_matrix( \
			SGSparseVector<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec) \
{ \
	SG_NOTIMPLEMENTED \
}

GET_SPARSE_MATRIX(read_char, bool)
GET_SPARSE_MATRIX(read_char, int8_t)
GET_SPARSE_MATRIX(read_byte, uint8_t)
GET_SPARSE_MATRIX(read_char, char)
GET_SPARSE_MATRIX(read_int, int32_t)
GET_SPARSE_MATRIX(read_uint, uint32_t)
GET_SPARSE_MATRIX(read_short_real, float32_t)
GET_SPARSE_MATRIX(read_real, float64_t)
GET_SPARSE_MATRIX(read_long_real, floatmax_t)
GET_SPARSE_MATRIX(read_short, int16_t)
GET_SPARSE_MATRIX(read_word, uint16_t)
GET_SPARSE_MATRIX(read_long, int64_t)
GET_SPARSE_MATRIX(read_ulong, uint64_t)
#undef GET_SPARSE_MATRIX

#define SET_VECTOR(format, sg_type) \
void CCSVFile::set_vector(const sg_type* vector, int32_t len) \
{ \
	SG_SET_LOCALE_C; \
	\
	if (!is_data_transposed) \
	{ \
		for (int32_t i=0; i<len; i++) \
			fprintf(file, "%" format "\n", vector[i]); \
	} \
	else \
	{ \
		int32_t i; \
		for (i=0; i<len-1; i++) \
			fprintf(file, "%" format "%c", vector[i], m_delimiter); \
		fprintf(file, "%" format "\n", vector[i]); \
	} \
	\
	SG_RESET_LOCALE; \
}

SET_VECTOR(SCNi8, int8_t)
SET_VECTOR(SCNu8, uint8_t)
SET_VECTOR(SCNu8, char)
SET_VECTOR(SCNi32, int32_t)
SET_VECTOR(SCNu32, uint32_t)
SET_VECTOR(SCNi64, int64_t)
SET_VECTOR(SCNu64, uint64_t)
SET_VECTOR("g", float32_t)
SET_VECTOR("lg", float64_t)
SET_VECTOR("Lg", floatmax_t)
SET_VECTOR(SCNi16, int16_t)
SET_VECTOR(SCNu16, uint16_t)
#undef SET_VECTOR

#define SET_MATRIX(format, sg_type) \
void CCSVFile::set_matrix(const sg_type* matrix, int32_t num_feat, int32_t num_vec) \
{ \
	SG_SET_LOCALE_C; \
	\
	if (!is_data_transposed) \
	{ \
		for (int32_t i=0; i<num_vec; i++) \
		{ \
			int32_t j; \
			for (j=0; j<num_feat-1; j++) \
				fprintf(file, "%" format "%c", matrix[j+i*num_feat], m_delimiter); \
			fprintf(file, "%" format "\n", matrix[j+i*num_feat]); \
		} \
	} \
	else \
	{ \
		for (int32_t i=0; i<num_feat; i++) \
		{ \
			int32_t j; \
			for (j=0; j<num_vec-1; j++) \
				fprintf(file, "%" format "%c", matrix[i+j*num_vec], m_delimiter); \
			fprintf(file, "%" format "\n", matrix[i+j*num_vec]); \
		} \
	} \
	\
	SG_RESET_LOCALE; \
}

SET_MATRIX(SCNi8, int8_t)
SET_MATRIX(SCNu8, uint8_t)
SET_MATRIX(SCNu8, char)
SET_MATRIX(SCNi32, int32_t)
SET_MATRIX(SCNu32, uint32_t)
SET_MATRIX(SCNi64, int64_t)
SET_MATRIX(SCNu64, uint64_t)
SET_MATRIX("g", float32_t)
SET_MATRIX("lg", float64_t)
SET_MATRIX("Lg", floatmax_t)
SET_MATRIX(SCNi16, int16_t)
SET_MATRIX(SCNu16, uint16_t)
#undef SET_MATRIX

#define SET_SPARSE_MATRIX(format, sg_type) \
void CCSVFile::set_sparse_matrix( \
			const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec) \
{ \
	SG_NOTIMPLEMENTED \
}

SET_SPARSE_MATRIX(SCNi8, bool)
SET_SPARSE_MATRIX(SCNi8, int8_t)
SET_SPARSE_MATRIX(SCNu8, uint8_t)
SET_SPARSE_MATRIX(SCNu8, char)
SET_SPARSE_MATRIX(SCNi32, int32_t)
SET_SPARSE_MATRIX(SCNu32, uint32_t)
SET_SPARSE_MATRIX(SCNi64, int64_t)
SET_SPARSE_MATRIX(SCNu64, uint64_t)
SET_SPARSE_MATRIX("g", float32_t)
SET_SPARSE_MATRIX("lg", float64_t)
SET_SPARSE_MATRIX("Lg", floatmax_t)
SET_SPARSE_MATRIX(SCNi16, int16_t)
SET_SPARSE_MATRIX(SCNu16, uint16_t)
#undef SET_SPARSE_MATRIX

void CCSVFile::get_string_list(
			SGString<char>*& strings, int32_t& num_str,
			int32_t& max_string_len)
{
	SGVector<char> line;
	int32_t current_line_idx=0;
	int32_t num_tokens=0;

	max_string_len=0;
	num_str=get_stats(num_tokens);
	strings=SG_MALLOC(SGString<char>, num_str);

	skip_lines(m_num_to_skip);
	while (m_line_reader->has_next())
	{
		line=m_line_reader->read_line();
		strings[current_line_idx].slen=line.vlen;
		strings[current_line_idx].string=SG_MALLOC(char, line.vlen);
		for (int32_t i=0; i<line.vlen; i++)
			strings[current_line_idx].string[i]=line[i];
	
		if (line.vlen>max_string_len)
			max_string_len=line.vlen;

		current_line_idx++;
	}

	num_str=current_line_idx;
}

#define GET_STRING_LIST(sg_type) \
void CCSVFile::get_string_list( \
			SGString<sg_type>*& strings, int32_t& num_str, \
			int32_t& max_string_len) \
{ \
	SG_NOTIMPLEMENTED \
}

GET_STRING_LIST(int8_t)
GET_STRING_LIST(uint8_t)
GET_STRING_LIST(int32_t)
GET_STRING_LIST(uint32_t)
GET_STRING_LIST(int64_t)
GET_STRING_LIST(uint64_t)
GET_STRING_LIST(float32_t)
GET_STRING_LIST(float64_t)
GET_STRING_LIST(floatmax_t)
GET_STRING_LIST(int16_t)
GET_STRING_LIST(uint16_t)
#undef GET_STRING_LIST

void CCSVFile::set_string_list(
			const SGString<char>* strings, int32_t num_str)
{
	for (int32_t i=0; i<num_str; i++)
	{
		for (int32_t j=0; j<strings[i].slen; j++)
			fprintf(file, "%c", strings[i].string[j]);
		fprintf(file, "\n");
	}
}

#define SET_STRING_LIST(sg_type) \
void CCSVFile::set_string_list( \
			const SGString<sg_type>* strings, int32_t num_str) \
{ \
	SG_NOTIMPLEMENTED \
}

SET_STRING_LIST(int8_t)
SET_STRING_LIST(uint8_t)
SET_STRING_LIST(int32_t)
SET_STRING_LIST(uint32_t)
SET_STRING_LIST(int64_t)
SET_STRING_LIST(uint64_t)
SET_STRING_LIST(float32_t)
SET_STRING_LIST(float64_t)
SET_STRING_LIST(floatmax_t)
SET_STRING_LIST(int16_t)
SET_STRING_LIST(uint16_t)
#undef SET_STRING_LIST

void CCSVFile::tokenize(char delim, substring s, v_array<substring>& ret)
{
	ret.erase();
	char *last = s.start;
	for (; s.start != s.end; s.start++)
	{
		if (*s.start == delim)
		{
			if (s.start != last)
			{
				substring temp = {last,s.start};
				ret.push(temp);
			}
			last = s.start+1;
		}
	}
	if (s.start != last)
	{
		substring final = {last, s.start};
		ret.push(final);
	}
}
