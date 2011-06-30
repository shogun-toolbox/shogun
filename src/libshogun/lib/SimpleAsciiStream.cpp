/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/SimpleAsciiStream.h"

using namespace shogun;

CSimpleAsciiStream::CSimpleAsciiStream()
	: CSimpleStream()
{
}

CSimpleAsciiStream::CSimpleAsciiStream(CAsciiFile* f)
	: CSimpleStream()
{
	ascii_file=f;
}

CSimpleAsciiStream::~CSimpleAsciiStream()
{
}

inline bool CSimpleAsciiStream::str_to_bool(char *str)
{
	return (atoi(str)!=0);
}

/* Methods for reading dense vectors from an ascii file */
#define GET_VECTOR(fname, conv, sg_type)			\
void CSimpleAsciiStream::fname(sg_type*& vector, int32_t& num_feat)	\
{									\
	size_t buffer_size=1024;					\
	char* buffer=new char[buffer_size];				\
	ssize_t bytes_read;						\
									\
	bytes_read=ascii_file->get_line(&buffer, &buffer_size);		\
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
			ascii_file->append_item				\
				(items, ptr_data, ptr_item);		\
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
			ascii_file->append_item				\
				(items, ptr_data, ptr_item);		\
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

GET_VECTOR(get_bool_vector, atoi, bool)
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
void CSimpleAsciiStream::fname(sg_type*& vector, int32_t& num_feat, float64_t& label) \
{									\
	size_t buffer_size=1024;					\
	char* buffer=new char[buffer_size];				\
	ssize_t bytes_read;						\
									\
	bytes_read=ascii_file->get_line(&buffer, &buffer_size);		\
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
			ascii_file->append_item				\
				(items, ptr_data, ptr_item);		\
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
			ascii_file->append_item				\
				(items, ptr_data, ptr_item);		\
			ptr_item=NULL;					\
			nf++;						\
		}							\
									\
		ptr_data++;						\
	}								\
									\
	SG_DEBUG("num_feat %d\n", num_feat);				\
	delete buffer;							\
	/* The first element is the label */				\
	label=atof(items->get_element(0));				\
	/* now copy rest of the data into vector */			\
	vector=new sg_type[num_feat-1];					\
	for (int32_t i=1; i<num_feat; i++)				\
	{								\
		char* item=items->get_element(i);			\
		vector[i-1]=conv(item);					\
		delete[] item;						\
	}								\
	delete items;							\
	num_feat--;							\
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
