/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/StreamingAsciiFile.h"
#include "lib/Mathematics.h"

#include <ctype.h>

using namespace shogun;

CStreamingAsciiFile::CStreamingAsciiFile(void)
	: CStreamingFile()
{
	SG_UNSTABLE("CStreamingAsciiFile::CStreamingAsciiFile(void)", "\n");
}

CStreamingAsciiFile::CStreamingAsciiFile(FILE* f, const char* name)
	: CStreamingFile(f, name)
{
}

CStreamingAsciiFile::CStreamingAsciiFile(char* fname, char rw, const char* name)
	: CStreamingFile(fname, rw, name)
{
}

CStreamingAsciiFile::~CStreamingAsciiFile()
{
}

inline bool CStreamingAsciiFile::str_to_bool(char *str)
{
	return (atoi(str)!=0);
}

/* Methods for reading dense vectors from an ascii file */

#define GET_VECTOR(fname, conv, sg_type)			\
void CStreamingAsciiFile::fname(sg_type*& vector, int32_t& num_feat)	\
{									\
	size_t buffer_size=1024;					\
	char* buffer=new char[buffer_size];				\
	ssize_t bytes_read;						\
									\
	bytes_read=CAsciiFile::getline(&buffer, &buffer_size, file);	\
									\
									\
	if (bytes_read<=0)						\
	{								\
		vector=NULL;						\
		num_feat=-1;						\
		return;							\
	}								\
									\
									\
	SG_DEBUG("line read from file:\n%s\n", buffer);			\
									\
	/* determine num_feat, populate dynamic array */		\
	int32_t nf=0;							\
	num_feat=0;							\
									\
	char* ptr_item=NULL;						\
	char* ptr_data=buffer;						\
	DynArray<char*>* items=new DynArray<char*>();			\
									\
	while (*ptr_data)						\
	{								\
		if ((*ptr_data=='\n') ||				\
		    (ptr_data - buffer >= bytes_read - 1))		\
		{							\
			if (ptr_item)					\
				nf++;					\
									\
			append_item(items, ptr_data, ptr_item);		\
			num_feat=nf;					\
									\
			nf=0;						\
			ptr_item=NULL;					\
			break;						\
		}							\
		else if (!isblank(*ptr_data) && !ptr_item)		\
		{							\
			ptr_item=ptr_data;				\
		}							\
		else if (isblank(*ptr_data) && ptr_item)		\
		{							\
			append_item(items, ptr_data, ptr_item);		\
			ptr_item=NULL;					\
			nf++;						\
		}							\
									\
		ptr_data++;						\
	}								\
									\
	SG_DEBUG("num_feat %d\n", num_feat);				\
	delete buffer;							\
									\
	/* now copy data into vector */					\
	vector=new sg_type[num_feat];					\
	printf("alloced %d sg_type in address:%p.\n", num_feat, vector); \
	for (int32_t i=0; i<num_feat; i++)				\
	{								\
		char* item=items->get_element(i);			\
		vector[i]=conv(item);					\
		delete[] item;						\
	}								\
	delete items;							\
}

GET_VECTOR(get_byte_vector, atoi, uint8_t)
GET_VECTOR(get_char_vector, atoi, char)
GET_VECTOR(get_int_vector, atoi, int32_t)
GET_VECTOR(get_shortreal_vector, atof, float32_t)
GET_VECTOR(get_real_vector, atof, float64_t)
GET_VECTOR(get_short_vector, atoi, int16_t)
GET_VECTOR(get_word_vector, atoi, uint16_t)
GET_VECTOR(get_int8_vector, atoi, int8_t)
GET_VECTOR(get_uint_vector, atoi, uint32_t)
GET_VECTOR(get_long_vector, atoi, int64_t)
GET_VECTOR(get_ulong_vector, atoi, uint64_t)
GET_VECTOR(get_longreal_vector, atoi, floatmax_t)
#undef GET_VECTOR

/* Methods for reading a dense vector and a label from an ascii file */

#define GET_VECTOR_AND_LABEL(fname, conv, sg_type)			\
	void CStreamingAsciiFile::fname(sg_type*& vector, int32_t& num_feat, float64_t& label) \
	{								\
		size_t buffer_size=1024;				\
		char* buffer=new char[buffer_size];			\
		ssize_t bytes_read;					\
									\
		bytes_read=CAsciiFile::getline(&buffer, &buffer_size, file); \
									\
									\
		if (bytes_read<=0)					\
		{							\
			vector=NULL;					\
			num_feat=-1;					\
			return;						\
		}							\
									\
									\
		SG_DEBUG("line read from file:\n%s\n", buffer);		\
									\
		/* determine num_feat, populate dynamic array */	\
		int32_t nf=0;						\
		num_feat=0;						\
									\
		char* ptr_item=NULL;					\
		char* ptr_data=buffer;					\
		DynArray<char*>* items=new DynArray<char*>();		\
									\
		while (*ptr_data)					\
		{							\
			if ((*ptr_data=='\n') ||			\
			    (ptr_data - buffer >= bytes_read - 1))	\
			{						\
				if (ptr_item)				\
					nf++;				\
									\
				append_item(items, ptr_data, ptr_item);	\
				num_feat=nf;				\
									\
				nf=0;					\
				ptr_item=NULL;				\
				break;					\
			}						\
			else if (!isblank(*ptr_data) && !ptr_item)	\
			{						\
				ptr_item=ptr_data;			\
			}						\
			else if (isblank(*ptr_data) && ptr_item)	\
			{						\
				append_item(items, ptr_data, ptr_item);	\
				ptr_item=NULL;				\
				nf++;					\
			}						\
									\
			ptr_data++;					\
		}							\
									\
		SG_DEBUG("num_feat %d\n", num_feat);			\
		delete buffer;						\
		/* The first element is the label */			\
		label=atof(items->get_element(0));			\
		/* now copy rest of the data into vector */		\
		vector=new sg_type[num_feat-1];				\
		for (int32_t i=1; i<num_feat; i++)			\
		{							\
			char* item=items->get_element(i);		\
			vector[i-1]=conv(item);				\
			delete[] item;					\
		}							\
		delete items;						\
		num_feat--;						\
	}

GET_VECTOR_AND_LABEL(get_bool_vector_and_label, str_to_bool, bool)
GET_VECTOR_AND_LABEL(get_byte_vector_and_label, atoi, uint8_t)
GET_VECTOR_AND_LABEL(get_char_vector_and_label, atoi, char)
GET_VECTOR_AND_LABEL(get_int_vector_and_label, atoi, int32_t)
GET_VECTOR_AND_LABEL(get_shortreal_vector_and_label, atof, float32_t)
GET_VECTOR_AND_LABEL(get_real_vector_and_label, atof, float64_t)
GET_VECTOR_AND_LABEL(get_short_vector_and_label, atoi, int16_t)
GET_VECTOR_AND_LABEL(get_word_vector_and_label, atoi, uint16_t)
GET_VECTOR_AND_LABEL(get_int8_vector_and_label, atoi, int8_t)
GET_VECTOR_AND_LABEL(get_uint_vector_and_label, atoi, uint32_t)
GET_VECTOR_AND_LABEL(get_long_vector_and_label, atoi, int64_t)
GET_VECTOR_AND_LABEL(get_ulong_vector_and_label, atoi, uint64_t)
GET_VECTOR_AND_LABEL(get_longreal_vector_and_label, atoi, floatmax_t)
#undef GET_VECTOR_AND_LABEL

/* Methods for reading a string vector from an ascii file (see StringFeatures) */

#define GET_STRING(fname, conv, sg_type)				\
void CStreamingAsciiFile::fname(sg_type*& vector, int32_t& len)		\
{									\
	size_t buffer_size=1024;					\
	char* buffer=new char[buffer_size];				\
	ssize_t bytes_read;						\
									\
	bytes_read=CAsciiFile::getline(&buffer, &buffer_size, file);	\
									\
	if (bytes_read<=1)						\
	{								\
		vector=NULL;						\
		len=-1;							\
		return;							\
	}								\
									\
	SG_DEBUG("Line read from the file:\n%s\n", buffer);		\
	/* Remove terminating \n */					\
	if (buffer[bytes_read-1]=='\n')					\
	{								\
		len=bytes_read-1;					\
		buffer[bytes_read-1]='\0';				\
	}								\
	else								\
		len=bytes_read;						\
	vector=(sg_type *) buffer;					\
}									

GET_STRING(get_bool_string, str_to_bool, bool)
GET_STRING(get_byte_string, atoi, uint8_t)
GET_STRING(get_char_string, atoi, char)
GET_STRING(get_int_string, atoi, int32_t)
GET_STRING(get_shortreal_string, atof, float32_t)
GET_STRING(get_real_string, atof, float64_t)
GET_STRING(get_short_string, atoi, int16_t)
GET_STRING(get_word_string, atoi, uint16_t)
GET_STRING(get_int8_string, atoi, int8_t)
GET_STRING(get_uint_string, atoi, uint32_t)
GET_STRING(get_long_string, atoi, int64_t)
GET_STRING(get_ulong_string, atoi, uint64_t)
GET_STRING(get_longreal_string, atoi, floatmax_t)
#undef GET_STRING

/* Methods for reading a string vector and a label from an ascii file */

#define GET_STRING_AND_LABEL(fname, conv, sg_type)			\
void CStreamingAsciiFile::fname(sg_type*& vector, int32_t& len, float64_t& label) \
{									\
	size_t buffer_size=1024;					\
	char* buffer=new char[buffer_size];				\
	ssize_t bytes_read;						\
									\
	bytes_read=CAsciiFile::getline(&buffer, &buffer_size, file);	\
									\
	if (bytes_read<=1)						\
	{								\
		vector=NULL;						\
		len=-1;							\
		return;							\
	}								\
									\
	int32_t str_start_pos=-1;					\
									\
	for (int32_t i=0; i<bytes_read; i++)				\
	{								\
		if (buffer[i] == ' ')					\
		{							\
			buffer[i]='\0';					\
			label=atoi(buffer);				\
			buffer[i]=' ';					\
			str_start_pos=i+1;				\
			break;						\
		}							\
	}								\
	/* If no label found, set vector=NULL and length=-1 */		\
	if (str_start_pos == -1)					\
	{								\
		vector=NULL;						\
		len=-1;							\
		return;							\
	}								\
	/* Remove terminating \n */					\
	if (buffer[bytes_read-1]=='\n')					\
	{								\
		buffer[bytes_read-1]='\0';				\
		len=bytes_read-str_start_pos-1;				\
	}								\
	else								\
		len=bytes_read-str_start_pos;				\
									\
	vector=(sg_type*) &buffer[str_start_pos];			\
}

GET_STRING_AND_LABEL(get_bool_string_and_label, str_to_bool, bool)
GET_STRING_AND_LABEL(get_byte_string_and_label, atoi, uint8_t)
GET_STRING_AND_LABEL(get_char_string_and_label, atoi, char)
GET_STRING_AND_LABEL(get_int_string_and_label, atoi, int32_t)
GET_STRING_AND_LABEL(get_shortreal_string_and_label, atof, float32_t)
GET_STRING_AND_LABEL(get_real_string_and_label, atof, float64_t)
GET_STRING_AND_LABEL(get_short_string_and_label, atoi, int16_t)
GET_STRING_AND_LABEL(get_word_string_and_label, atoi, uint16_t)
GET_STRING_AND_LABEL(get_int8_string_and_label, atoi, int8_t)
GET_STRING_AND_LABEL(get_uint_string_and_label, atoi, uint32_t)
GET_STRING_AND_LABEL(get_long_string_and_label, atoi, int64_t)
GET_STRING_AND_LABEL(get_ulong_string_and_label, atoi, uint64_t)
GET_STRING_AND_LABEL(get_longreal_string_and_label, atoi, floatmax_t)
#undef GET_STRING_AND_LABEL

/* Methods for reading a sparse vector from an ascii file */

#define GET_SPARSE_VECTOR(fname, conv, sg_type)				\
void CStreamingAsciiFile::fname(SGSparseVectorEntry<sg_type>*& vector, int32_t& len) \
{									\
	size_t buffer_size=1024;					\
	char* buffer=new char[buffer_size];				\
	ssize_t bytes_read;						\
									\
	bytes_read=CAsciiFile::getline(&buffer, &buffer_size, file);	\
									\
	if (bytes_read<=1)						\
	{								\
		vector=NULL;						\
		len=-1;							\
		return;							\
	}								\
									\
	SG_DEBUG("Line read from the file:\n%s\n", buffer);		\
									\
	/* Remove terminating \n */					\
	int32_t num_chars;						\
	if (buffer[bytes_read-1]=='\n')					\
	  {								\
	    num_chars=bytes_read-1;					\
	    buffer[num_chars]='\0';					\
	  }								\
	else								\
	  num_chars=bytes_read;						\
									\
	int32_t num_dims=0;						\
	for (int32_t i=0; i<num_chars; i++)				\
	{								\
		if (buffer[i]==':')					\
		{							\
			num_dims++;					\
		}							\
	}								\
									\
	int32_t index_start_pos=-1;					\
	int32_t feature_start_pos;					\
	int32_t current_feat=0;						\
	vector=new SGSparseVectorEntry<sg_type>[num_dims];		\
	for (int32_t i=0; i<num_chars; i++)				\
	{								\
		if (buffer[i]==':')					\
		{							\
			buffer[i]='\0';					\
			vector[current_feat].feat_index=(int32_t) atoi(buffer+index_start_pos)-1; \
			/* Unset index_start_pos */			\
			index_start_pos=-1;				\
									\
			feature_start_pos=i+1;				\
			while ((buffer[i]!=' ') && (i<num_chars))	\
			{						\
				i++;					\
			}						\
									\
			buffer[i]='\0';					\
			vector[current_feat].entry=(sg_type) conv(buffer+feature_start_pos); \
									\
			current_feat++;					\
		}							\
		else if (buffer[i]==' ')				\
		  i++;							\
		else							\
		  {							\
		    /* Set index_start_pos if not set already */	\
		    /* if already set, it means the index is  */	\
		    /* more than one digit long.              */	\
		    if (index_start_pos == -1)				\
			index_start_pos=i;				\
		  }							\
	}								\
									\
	len=current_feat;						\
}

GET_SPARSE_VECTOR(get_bool_sparse_vector, str_to_bool, bool)
GET_SPARSE_VECTOR(get_byte_sparse_vector, atoi, uint8_t)
GET_SPARSE_VECTOR(get_char_sparse_vector, atoi, char)
GET_SPARSE_VECTOR(get_int_sparse_vector, atoi, int32_t)
GET_SPARSE_VECTOR(get_shortreal_sparse_vector, atof, float32_t)
GET_SPARSE_VECTOR(get_real_sparse_vector, atof, float64_t)
GET_SPARSE_VECTOR(get_short_sparse_vector, atoi, int16_t)
GET_SPARSE_VECTOR(get_word_sparse_vector, atoi, uint16_t)
GET_SPARSE_VECTOR(get_int8_sparse_vector, atoi, int8_t)
GET_SPARSE_VECTOR(get_uint_sparse_vector, atoi, uint32_t)
GET_SPARSE_VECTOR(get_long_sparse_vector, atoi, int64_t)
GET_SPARSE_VECTOR(get_ulong_sparse_vector, atoi, uint64_t)
GET_SPARSE_VECTOR(get_longreal_sparse_vector, atoi, floatmax_t)
#undef GET_SPARSE_VECTOR

/* Methods for reading a sparse vector and a label from an ascii file */

#define GET_SPARSE_VECTOR_AND_LABEL(fname, conv, sg_type)			\
void CStreamingAsciiFile::fname(SGSparseVectorEntry<sg_type>*& vector, int32_t& len, float64_t& label) \
{									\
	size_t buffer_size=1024;					\
	char* buffer=new char[buffer_size];				\
	ssize_t bytes_read;						\
									\
	bytes_read=CAsciiFile::getline(&buffer, &buffer_size, file);	\
									\
	if (bytes_read<=1)						\
	{								\
		vector=NULL;						\
		len=-1;							\
		return;							\
	}								\
									\
	SG_DEBUG("Line read from the file:\n%s\n", buffer);		\
									\
	/* Remove terminating \n */					\
	int32_t num_chars;						\
	if (buffer[bytes_read-1]=='\n')					\
	{								\
		num_chars=bytes_read-1;					\
		buffer[num_chars]='\0';					\
	}								\
	else								\
		num_chars=bytes_read;					\
									\
	int32_t num_dims=0;						\
	for (int32_t i=0; i<num_chars; i++)				\
	{								\
		if (buffer[i]==':')					\
		{							\
			num_dims++;					\
		}							\
	}								\
									\
	int32_t index_start_pos=-1;					\
	int32_t feature_start_pos;					\
	int32_t current_feat=0;						\
	int32_t label_pos=-1;						\
	vector=new SGSparseVectorEntry<sg_type>[num_dims];		\
									\
	for (int32_t i=1; i<num_chars; i++)				\
	{								\
		if (buffer[i]==':')					\
		{							\
			break;						\
		}							\
		if ( (buffer[i]==' ') && (buffer[i-1]!=' ') )		\
		{							\
			buffer[i]='\0';					\
			label_pos=i;					\
			label=atof(buffer);				\
			break;						\
		}							\
	}								\
									\
	if (label_pos==-1)						\
		SG_ERROR("No label found!\n");				\
									\
	buffer+=label_pos+1;						\
	num_chars-=label_pos+1;						\
	for (int32_t i=0; i<num_chars; i++)				\
	{								\
		if (buffer[i]==':')					\
		{							\
			buffer[i]='\0';					\
			vector[current_feat].feat_index=(int32_t) atoi(buffer+index_start_pos)-1; \
			/* Unset index_start_pos */			\
			index_start_pos=-1;				\
									\
			feature_start_pos=i+1;				\
			while ((buffer[i]!=' ') && (i<num_chars))	\
			{						\
				i++;					\
			}						\
									\
			buffer[i]='\0';					\
			vector[current_feat].entry=(sg_type) conv(buffer+feature_start_pos); \
									\
			current_feat++;					\
		}							\
		else if (buffer[i]==' ')				\
			i++;						\
		else							\
		{							\
			/* Set index_start_pos if not set already */	\
			/* if already set, it means the index is  */	\
			/* more than one digit long.              */	\
			if (index_start_pos == -1)			\
				index_start_pos=i;			\
		}							\
	}								\
									\
	len=current_feat;						\
}

GET_SPARSE_VECTOR_AND_LABEL(get_bool_sparse_vector_and_label, str_to_bool, bool)
GET_SPARSE_VECTOR_AND_LABEL(get_byte_sparse_vector_and_label, atoi, uint8_t)
GET_SPARSE_VECTOR_AND_LABEL(get_char_sparse_vector_and_label, atoi, char)
GET_SPARSE_VECTOR_AND_LABEL(get_int_sparse_vector_and_label, atoi, int32_t)
GET_SPARSE_VECTOR_AND_LABEL(get_shortreal_sparse_vector_and_label, atof, float32_t)
GET_SPARSE_VECTOR_AND_LABEL(get_real_sparse_vector_and_label, atof, float64_t)
GET_SPARSE_VECTOR_AND_LABEL(get_short_sparse_vector_and_label, atoi, int16_t)
GET_SPARSE_VECTOR_AND_LABEL(get_word_sparse_vector_and_label, atoi, uint16_t)
GET_SPARSE_VECTOR_AND_LABEL(get_int8_sparse_vector_and_label, atoi, int8_t)
GET_SPARSE_VECTOR_AND_LABEL(get_uint_sparse_vector_and_label, atoi, uint32_t)
GET_SPARSE_VECTOR_AND_LABEL(get_long_sparse_vector_and_label, atoi, int64_t)
GET_SPARSE_VECTOR_AND_LABEL(get_ulong_sparse_vector_and_label, atoi, uint64_t)
GET_SPARSE_VECTOR_AND_LABEL(get_longreal_sparse_vector_and_label, atoi, floatmax_t)
#undef GET_SPARSE_VECTOR_AND_LABEL

template <class T>
void CStreamingAsciiFile::append_item(
	DynArray<T>* items, char* ptr_data, char* ptr_item)
{
	size_t len=(ptr_data-ptr_item)/sizeof(char);
	char* item=new char[len+1];
	memset(item, 0, sizeof(char)*(len+1));
	item=strncpy(item, ptr_item, len);

	SG_DEBUG("current %c, len %d, item %s\n", *ptr_data, len, item);
	items->append_element(item);
}
