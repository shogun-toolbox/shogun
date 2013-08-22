/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#include <shogun/io/LibSVMFile.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/base/DynArray.h>

using namespace shogun;

CLibSVMFile::CLibSVMFile()
{
	init();
}

CLibSVMFile::CLibSVMFile(FILE* f, const char* name) :
	CFile(f, name)
{
	init();
	init_with_defaults();
}

CLibSVMFile::CLibSVMFile(const char* fname, char rw, const char* name) :
	CFile(fname, rw, name)
{
	init();
	init_with_defaults();
}

CLibSVMFile::~CLibSVMFile()
{
	SG_UNREF(m_whitespace_tokenizer);
	SG_UNREF(m_delimiter_tokenizer);
	SG_UNREF(m_line_tokenizer);
	SG_UNREF(m_parser);
	SG_UNREF(m_line_reader);
}

void CLibSVMFile::init()
{
	m_delimiter=0;

	m_whitespace_tokenizer=NULL;
	m_delimiter_tokenizer=NULL;
	m_line_tokenizer=NULL;
	m_parser=NULL;
	m_line_reader=NULL;
}

void CLibSVMFile::init_with_defaults()
{
	m_delimiter=':';

	m_whitespace_tokenizer=new CDelimiterTokenizer(true);
	m_whitespace_tokenizer->delimiters[' ']=1;
	SG_REF(m_whitespace_tokenizer);

	m_delimiter_tokenizer=new CDelimiterTokenizer(true);
	m_delimiter_tokenizer->delimiters[m_delimiter]=1;
	SG_REF(m_delimiter_tokenizer);

	m_line_tokenizer=new CDelimiterTokenizer(true);
	m_line_tokenizer->delimiters['\n']=1;
	SG_REF(m_line_tokenizer);

	m_parser=new CParser();
	m_line_reader=new CLineReader(file, m_line_tokenizer);
}

#define GET_SPARSE_MATRIX(read_func, sg_type) \
void CLibSVMFile::get_sparse_matrix(SGSparseVector<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec) \
{ \
	float64_t* labels=NULL; \
	get_sparse_matrix(matrix, num_feat, num_vec, labels, false); \
}

GET_SPARSE_MATRIX(read_bool, bool)
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

#define GET_LABELED_SPARSE_MATRIX(read_func, sg_type) \
void CLibSVMFile::get_sparse_matrix(SGSparseVector<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec, \
					float64_t*& labels, bool load_labels) \
{ \
	num_feat=0; \
	\
	SG_INFO("counting line numbers in file %s\n", filename) \
	num_vec=get_num_lines(); \
	\
	int32_t current_line_ind=0; \
	SGVector<char> line; \
	\
	int32_t num_entries=0; \
	DynArray<SGVector<char> > entries; \
	\
	matrix=SG_MALLOC(SGSparseVector<sg_type>, num_vec); \
	if (load_labels) \
		labels=SG_MALLOC(float64_t, num_vec); \
	\
	while (m_line_reader->has_next()) \
	{ \
		num_entries=0; \
		entries.reset(SGVector<char>(false)); \
		line=m_line_reader->read_line(); \
		\
		m_parser->set_tokenizer(m_whitespace_tokenizer); \
		m_parser->set_text(line); \
		\
		if (load_labels && m_parser->has_next()) \
			labels[current_line_ind]=m_parser->read_real(); \
		\
		while (m_parser->has_next()) \
		{ \
			entries.push_back(m_parser->read_string()); \
			num_entries++; \
		} \
		\
		matrix[current_line_ind]=SGSparseVector<sg_type>(num_entries); \
		for (int32_t i=0; i<num_entries; i++) \
		{ \
			m_parser->set_tokenizer(m_delimiter_tokenizer); \
			m_parser->set_text(entries[i]); \
			\
			int32_t feat_index=0; \
			if (m_parser->has_next()) \
				feat_index=m_parser->read_int(); \
			\
			sg_type entry=0; \
			if (m_parser->has_next()) \
				entry=m_parser->read_func(); \
			\
			if (feat_index>num_feat) \
				num_feat=feat_index; \
			\
			matrix[current_line_ind].features[i].feat_index=feat_index; \
			matrix[current_line_ind].features[i].entry=entry; \
		} \
		\
		current_line_ind++; \
		SG_PROGRESS(current_line_ind, 0, num_vec, 1, "LOADING:\t") \
	} \
	\
	SG_INFO("file successfully read\n") \
}

GET_LABELED_SPARSE_MATRIX(read_bool, bool)
GET_LABELED_SPARSE_MATRIX(read_char, int8_t)
GET_LABELED_SPARSE_MATRIX(read_byte, uint8_t)
GET_LABELED_SPARSE_MATRIX(read_char, char)
GET_LABELED_SPARSE_MATRIX(read_int, int32_t)
GET_LABELED_SPARSE_MATRIX(read_uint, uint32_t)
GET_LABELED_SPARSE_MATRIX(read_short_real, float32_t)
GET_LABELED_SPARSE_MATRIX(read_real, float64_t)
GET_LABELED_SPARSE_MATRIX(read_long_real, floatmax_t)
GET_LABELED_SPARSE_MATRIX(read_short, int16_t)
GET_LABELED_SPARSE_MATRIX(read_word, uint16_t)
GET_LABELED_SPARSE_MATRIX(read_long, int64_t)
GET_LABELED_SPARSE_MATRIX(read_ulong, uint64_t)
#undef GET_LABELED_SPARSE_MATRIX

#define SET_SPARSE_MATRIX(format, sg_type) \
void CLibSVMFile::set_sparse_matrix( \
			const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec) \
{ \
	set_sparse_matrix(matrix, num_feat, num_vec, NULL); \
}

SET_SPARSE_MATRIX(SCNi32, bool)
SET_SPARSE_MATRIX(SCNi8, int8_t)
SET_SPARSE_MATRIX(SCNu8, uint8_t)
SET_SPARSE_MATRIX(SCNu8, char)
SET_SPARSE_MATRIX(SCNi32, int32_t)
SET_SPARSE_MATRIX(SCNu32, uint32_t)
SET_SPARSE_MATRIX(SCNi64, int64_t)
SET_SPARSE_MATRIX(SCNu64, uint64_t)
SET_SPARSE_MATRIX(".16g", float32_t)
SET_SPARSE_MATRIX(".16lg", float64_t)
SET_SPARSE_MATRIX(".16Lg", floatmax_t)
SET_SPARSE_MATRIX(SCNi16, int16_t)
SET_SPARSE_MATRIX(SCNu16, uint16_t)
#undef SET_SPARSE_MATRIX

#define SET_LABELED_SPARSE_MATRIX(format, sg_type) \
void CLibSVMFile::set_sparse_matrix( \
			const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec, \
			const float64_t* labels) \
{ \
	for (int32_t i=0; i<num_vec; i++) \
	{ \
		if (labels!=NULL) \
			fprintf(file, "%lg ", labels[i]); \
		\
		for (int32_t j=0; j<matrix[i].num_feat_entries; j++) \
		{ \
			fprintf(file, "%d%c%" format " ", \
				matrix[i].features[j].feat_index, \
				m_delimiter, \
				matrix[i].features[j].entry); \
		} \
		fprintf(file, "\n"); \
	} \
}

SET_LABELED_SPARSE_MATRIX(SCNi32, bool)
SET_LABELED_SPARSE_MATRIX(SCNi8, int8_t)
SET_LABELED_SPARSE_MATRIX(SCNu8, uint8_t)
SET_LABELED_SPARSE_MATRIX(SCNu8, char)
SET_LABELED_SPARSE_MATRIX(SCNi32, int32_t)
SET_LABELED_SPARSE_MATRIX(SCNu32, uint32_t)
SET_LABELED_SPARSE_MATRIX(SCNi64, int64_t)
SET_LABELED_SPARSE_MATRIX(SCNu64, uint64_t)
SET_LABELED_SPARSE_MATRIX(".16g", float32_t)
SET_LABELED_SPARSE_MATRIX(".16lg", float64_t)
SET_LABELED_SPARSE_MATRIX(".16Lg", floatmax_t)
SET_LABELED_SPARSE_MATRIX(SCNi16, int16_t)
SET_LABELED_SPARSE_MATRIX(SCNu16, uint16_t)
#undef SET_LABELED_SPARSE_MATRIX

int32_t CLibSVMFile::get_num_lines()
{
	int32_t num_lines=0;
	while (m_line_reader->has_next()) 
	{
		m_line_reader->skip_line();	
		num_lines++;
	}
	m_line_reader->reset();

	return num_lines;
}
