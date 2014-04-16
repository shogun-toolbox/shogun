/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#include <shogun/io/LibSVMFile.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/base/DynArray.h>
#include <shogun/io/LineReader.h>
#include <shogun/io/Parser.h>
#include <shogun/lib/DelimiterTokenizer.h>

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
	SG_UNREF(m_delimiter_feat_tokenizer);
	SG_UNREF(m_delimiter_label_tokenizer);
	SG_UNREF(m_line_tokenizer);
	SG_UNREF(m_parser);
	SG_UNREF(m_line_reader);
}

void CLibSVMFile::init()
{
	m_delimiter_feat=0;

	m_whitespace_tokenizer=NULL;
	m_delimiter_feat_tokenizer=NULL;
	m_delimiter_label_tokenizer=NULL;
	m_line_tokenizer=NULL;
	m_parser=NULL;
	m_line_reader=NULL;
}

void CLibSVMFile::init_with_defaults()
{
	m_delimiter_feat=':';
	m_delimiter_label=',';

	m_whitespace_tokenizer=new CDelimiterTokenizer(true);
	m_whitespace_tokenizer->delimiters[' ']=1;
	SG_REF(m_whitespace_tokenizer);

	m_delimiter_feat_tokenizer=new CDelimiterTokenizer(true);
	m_delimiter_feat_tokenizer->delimiters[m_delimiter_feat]=1;
	SG_REF(m_delimiter_feat_tokenizer);
    
	m_delimiter_label_tokenizer=new CDelimiterTokenizer(true);
	m_delimiter_label_tokenizer->delimiters[m_delimiter_label]=1;
	SG_REF(m_delimiter_label_tokenizer);
	
	m_line_tokenizer=new CDelimiterTokenizer(true);
	m_line_tokenizer->delimiters['\n']=1;
	SG_REF(m_line_tokenizer);

	m_parser=new CParser();
	m_line_reader=new CLineReader(file, m_line_tokenizer);
}

#define GET_SPARSE_MATRIX(read_func, sg_type) \
void CLibSVMFile::get_sparse_matrix(SGSparseVector<sg_type>*& mat_feat, int32_t& num_feat, int32_t& num_vec) \
{ \
	SGVector<float64_t>* multilabel; \
	int32_t num_classes; \
	get_sparse_matrix(mat_feat, num_feat, num_vec, multilabel, num_classes, false); \
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
void CLibSVMFile::get_sparse_matrix(SGSparseVector<sg_type>*& mat_feat, int32_t& num_feat, int32_t& num_vec, \
					float64_t*& labels,  bool load_labels) \
{ \
	SGVector<float64_t>* multilabel; \
	int32_t num_classes; \
	get_sparse_matrix(mat_feat, num_feat, num_vec, multilabel, num_classes, load_labels); \
	\
	for (int32_t i=0; i<num_vec; i++) \
	{ \
		REQUIRE(multilabel[i].size()==1, \
			"%s a multilabel file. You are trying to read it with a single-label reader.", filename); \
	} \
	labels=SG_MALLOC(float64_t, num_vec); \
	\
	for (int32_t i=0; i<num_vec; i++) \
		labels[i]=multilabel[i][0]; \
	SG_FREE(multilabel); \
} \

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

#define GET_MULTI_LABELED_SPARSE_MATRIX(read_func, sg_type) \
void CLibSVMFile::get_sparse_matrix(SGSparseVector<sg_type>*& mat_feat, int32_t& num_feat, int32_t& num_vec, \
					SGVector<float64_t>*& multilabel, int32_t& num_classes, bool load_labels) \
{ \
	num_feat=0; \
	\
	SG_INFO("counting line numbers in file %s\n", filename) \
	num_vec=get_num_lines(); \
	\
	int32_t current_line_ind=0; \
	SGVector<char> line; \
	\
	int32_t num_feat_entries=0; \
	DynArray<SGVector<char> > entries_feat; \
	DynArray<float64_t > entries_label; \
	DynArray<float64_t> classes; \
	\
	mat_feat=SG_MALLOC(SGSparseVector<sg_type>, num_vec); \
	multilabel=SG_MALLOC(SGVector<float64_t>, num_vec); \
	\
	num_classes=0; \
	SG_SET_LOCALE_C; \
	\
	while (m_line_reader->has_next()) \
	{ \
		num_feat_entries=0; \
		entries_feat.reset(SGVector<char>(false)); \
		line=m_line_reader->read_line(); \
		\
		m_parser->set_tokenizer(m_whitespace_tokenizer); \
		m_parser->set_text(line); \
		\
		SGVector<char> entry_label; \
		if (load_labels && m_parser->has_next()) \
		{ \
			entry_label=m_parser->read_string(); \
			if (is_feat_entry(entry_label)) \
			{ \
				entries_feat.push_back(entry_label); \
				num_feat_entries++; \
				entry_label=SGVector<char>(0); \
			} \
		} \
		\
		while (m_parser->has_next()) \
		{ \
			entries_feat.push_back(m_parser->read_string()); \
			num_feat_entries++; \
		} \
		\
		mat_feat[current_line_ind]=SGSparseVector<sg_type>(num_feat_entries); \
		for (int32_t i=0; i<num_feat_entries; i++) \
		{ \
			m_parser->set_tokenizer(m_delimiter_feat_tokenizer); \
			m_parser->set_text(entries_feat[i]); \
			\
			int32_t feat_index=0; \
			\
			if (m_parser->has_next()) \
				feat_index=m_parser->read_int(); \
			\
			sg_type entry=0; \
			\
			if (m_parser->has_next()) \
				entry=m_parser->read_func(); \
			\
			if (feat_index>num_feat) \
				num_feat=feat_index; \
			\
			mat_feat[current_line_ind].features[i].feat_index=feat_index-1; \
			mat_feat[current_line_ind].features[i].entry=entry; \
		} \
		\
		if (load_labels) \
		{ \
			m_parser->set_tokenizer(m_delimiter_label_tokenizer); \
			m_parser->set_text(entry_label); \
			\
			int32_t num_label_entries=0; \
			entries_label.reset(0); \
			\
			while (m_parser->has_next()) \
			{ \
				num_label_entries++; \
				float64_t label_val=m_parser->read_real(); \
				\
				if (classes.find_element(label_val)==-1) \
					classes.push_back(label_val); \
				\
				entries_label.push_back(label_val); \
			} \
			multilabel[current_line_ind]=SGVector<float64_t>(num_label_entries); \
			\
			for (int32_t j=0; j < num_label_entries; j++) \
				multilabel[current_line_ind][j]=entries_label[j]; \
			\
		} \
		\
		current_line_ind++; \
		SG_PROGRESS(current_line_ind, 0, num_vec, 1, "LOADING:\t") \
	} \
	num_classes=classes.get_num_elements(); \
	\
	SG_RESET_LOCALE; \
	\
	SG_INFO("file successfully read\n") \
}

GET_MULTI_LABELED_SPARSE_MATRIX(read_bool, bool)
GET_MULTI_LABELED_SPARSE_MATRIX(read_char, int8_t)
GET_MULTI_LABELED_SPARSE_MATRIX(read_byte, uint8_t)
GET_MULTI_LABELED_SPARSE_MATRIX(read_char, char)
GET_MULTI_LABELED_SPARSE_MATRIX(read_int, int32_t)
GET_MULTI_LABELED_SPARSE_MATRIX(read_uint, uint32_t)
GET_MULTI_LABELED_SPARSE_MATRIX(read_short_real, float32_t)
GET_MULTI_LABELED_SPARSE_MATRIX(read_real, float64_t)
GET_MULTI_LABELED_SPARSE_MATRIX(read_long_real, floatmax_t)
GET_MULTI_LABELED_SPARSE_MATRIX(read_short, int16_t)
GET_MULTI_LABELED_SPARSE_MATRIX(read_word, uint16_t)
GET_MULTI_LABELED_SPARSE_MATRIX(read_long, int64_t)
GET_MULTI_LABELED_SPARSE_MATRIX(read_ulong, uint64_t)
#undef GET_MULTI_LABELED_SPARSE_MATRIX

#define SET_SPARSE_MATRIX(format, sg_type) \
void CLibSVMFile::set_sparse_matrix( \
			const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec) \
{ \
	SGVector <float64_t>* labels = NULL; \
	set_sparse_matrix(matrix, num_feat, num_vec, labels); \
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
	SGVector<float64_t>* multilabel=SG_MALLOC(SGVector<float64_t>, num_vec); \
	\
	for (int32_t i=0; i<num_vec; i++) \
	{ \
		multilabel[i]=SGVector<float64_t>(1); \
		multilabel[i][0]=labels[i]; \
	} \
	\
	set_sparse_matrix(matrix, num_feat, num_vec, multilabel); \
	SG_FREE(multilabel); \
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

#define SET_MULTI_LABELED_SPARSE_MATRIX(format, sg_type) \
void CLibSVMFile::set_sparse_matrix( \
			const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec, \
			const SGVector<float64_t>* multilabel) \
{ \
	SG_SET_LOCALE_C; \
	\
	for (int32_t i=0; i<num_vec; i++) \
	{ \
		if (multilabel!=NULL) \
		{ \
			if (multilabel[i].size()==0) \
				fprintf(file, " "); \
			\
			for (int32_t j=0; j <multilabel[i].size(); j++) \
			{ \
				fprintf(file, "%lg", multilabel[i][j]); \
				\
				if (j==multilabel[i].size()-1) \
					fprintf(file, " "); \
				else \
					fprintf(file, ","); \
			} \
		} \
		\
		for (int32_t j=0; j<matrix[i].num_feat_entries; j++) \
		{ \
			fprintf(file, "%d%c%" format " ", \
			matrix[i].features[j].feat_index+1, \
			m_delimiter_feat, \
			matrix[i].features[j].entry); \
		} \
		fprintf(file, "\n"); \
	} \
	\
	SG_RESET_LOCALE; \
}

SET_MULTI_LABELED_SPARSE_MATRIX(SCNi32, bool)
SET_MULTI_LABELED_SPARSE_MATRIX(SCNi8, int8_t)
SET_MULTI_LABELED_SPARSE_MATRIX(SCNu8, uint8_t)
SET_MULTI_LABELED_SPARSE_MATRIX(SCNu8, char)
SET_MULTI_LABELED_SPARSE_MATRIX(SCNi32, int32_t)
SET_MULTI_LABELED_SPARSE_MATRIX(SCNu32, uint32_t)
SET_MULTI_LABELED_SPARSE_MATRIX(SCNi64, int64_t)
SET_MULTI_LABELED_SPARSE_MATRIX(SCNu64, uint64_t)
SET_MULTI_LABELED_SPARSE_MATRIX(".16g", float32_t)
SET_MULTI_LABELED_SPARSE_MATRIX(".16lg", float64_t)
SET_MULTI_LABELED_SPARSE_MATRIX(".16Lg", floatmax_t)
SET_MULTI_LABELED_SPARSE_MATRIX(SCNi16, int16_t)
SET_MULTI_LABELED_SPARSE_MATRIX(SCNu16, uint16_t)
#undef SET_MULTI_LABELED_SPARSE_MATRIX

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

bool CLibSVMFile::is_feat_entry(const SGVector<char> entry)
{
	CParser* parser = new CParser();
	parser->set_tokenizer(m_delimiter_feat_tokenizer);
	parser->set_text(entry);
	bool isfeat = false;
	
	if (parser->has_next())
	{
		parser->read_real();
                
		if (parser->has_next())
			isfeat = true;

	}

	SG_UNREF(parser);

	return isfeat;
}
