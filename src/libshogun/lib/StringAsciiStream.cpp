/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/StringAsciiStream.h"

using namespace shogun;

CStringAsciiStream::CStringAsciiStream()
	: CStringStream()
{
}

CStringAsciiStream::CStringAsciiStream(CAsciiFile* f)
	: CStringStream()
{
	ascii_file=f;
}

CStringAsciiStream::~CStringAsciiStream()
{
}

inline bool CStringAsciiStream::str_to_bool(char *str)
{
	return (atoi(str)!=0);
}

/* Methods for reading dense vectors from an ascii file */
#define GET_VECTOR(fname, conv, sg_type)				\
void CStringAsciiStream::fname(sg_type*& vector, int32_t& len)		\
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

/* Methods for reading a string vector and a label from an ascii file */

#define GET_VECTOR_AND_LABEL(fname, conv, sg_type)			\
void CStringAsciiStream::fname(sg_type*& vector, int32_t& len, float64_t& label) \
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
