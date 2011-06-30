/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/SparseAsciiStream.h"

using namespace shogun;

CSparseAsciiStream::CSparseAsciiStream()
	: CSparseStream()
{
}

CSparseAsciiStream::CSparseAsciiStream(CAsciiFile* f)
	: CSparseStream()
{
	ascii_file=f;
}

CSparseAsciiStream::~CSparseAsciiStream()
{
}

inline bool CSparseAsciiStream::str_to_bool(char *str)
{
	return (atoi(str)!=0);
}

/* Methods for reading a sparse vector from an ascii file */
#define GET_VECTOR(fname, conv, sg_type)				\
void CSparseAsciiStream::fname(SGSparseVectorEntry<sg_type>*& vector, int32_t& len) \
{									\
	size_t buffer_size=1024;					\
	char* buffer=new char[buffer_size];				\
	ssize_t bytes_read;						\
									\
	bytes_read=ascii_file->get_line(&buffer, &buffer_size);		\
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

GET_VECTOR(get_bool_vector, str_to_bool, bool)
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

/* Methods for reading a sparse vector and a label from an ascii file */
#define GET_VECTOR_AND_LABEL(fname, conv, sg_type)			\
void CSparseAsciiStream::fname(SGSparseVectorEntry<sg_type>*& vector, int32_t& len, float64_t& label) \
{									\
	size_t buffer_size=1024;					\
	char* buffer=new char[buffer_size];				\
	ssize_t bytes_read;						\
									\
	bytes_read=ascii_file->get_line(&buffer, &buffer_size);		\
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
