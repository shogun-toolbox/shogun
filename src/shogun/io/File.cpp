/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <shogun/io/File.h>

#include <shogun/features/StringFeatures.h>
#include <shogun/features/SparseFeatures.h>

using namespace shogun;

CFile::CFile() : CSGObject()
{
	file=NULL;
	filename=NULL;
	variable_name=NULL;
}

CFile::CFile(FILE* f, const char* name) : CSGObject()
{
	file=f;
	filename=NULL;
	variable_name=NULL;

	if (name)
		set_variable_name(name);
}

CFile::CFile(char* fname, char rw, const char* name) : CSGObject()
{
	variable_name=NULL;
	task=rw;
	filename=strdup(fname);
	char mode[2];
	mode[0]=rw;
	mode[1]='\0';

	if (rw=='r' || rw == 'w')
	{
		if (filename)
		{
			if (!(file=fopen((const char*) filename, (const char*) mode)))
				SG_ERROR("Error opening file '%s'\n", filename);
		}
	}
	else
		SG_ERROR("unknown mode '%c'\n", mode[0]);

	if (name)
		set_variable_name(name);
}

void CFile::get_vector(bool*& vector, int32_t& len)
{
	int32_t* int_vector;
	get_vector(int_vector, len);

	ASSERT(len>0);
	vector= SG_MALLOC(bool, len);

	for (int32_t i=0; i<len; i++)
		vector[i]= (int_vector[i]!=0);

	SG_FREE(int_vector);
}

void CFile::set_vector(const bool* vector, int32_t len)
{
	int32_t* int_vector = SG_MALLOC(int32_t, len);
	for (int32_t i=0;i<len;i++)
	{
		if (vector[i])
			int_vector[i]=1;
		else
			int_vector[i]=0;
	}
	set_vector(int_vector,len);
	SG_FREE(int_vector);
}

void CFile::get_matrix(bool*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	uint8_t * byte_matrix;
	get_matrix(byte_matrix,num_feat,num_vec);

	ASSERT(num_feat > 0 && num_vec > 0)
	matrix = SG_MALLOC(bool, num_feat*num_vec);

	for(int32_t i = 0;i < num_vec;i++)
	{
		for(int32_t j = 0;j < num_feat;j++)
			matrix[i*num_feat+j] = byte_matrix[i*num_feat+j] != 0 ? 1 : 0;
	}

	SG_FREE(byte_matrix);
}

void CFile::set_matrix(const bool* matrix, int32_t num_feat, int32_t num_vec)
{
	uint8_t * byte_matrix = SG_MALLOC(uint8_t, num_feat*num_vec);
	for(int32_t i = 0;i < num_vec;i++)
	{
		for(int32_t j = 0;j < num_feat;j++)
			byte_matrix[i*num_feat+j] = matrix[i*num_feat+j] != 0 ? 1 : 0;
	}

	set_matrix(byte_matrix,num_feat,num_vec);

	SG_FREE(byte_matrix);
}

void CFile::get_string_list(
		SGString<bool>*& strings, int32_t& num_str,
		int32_t& max_string_len)
{
	SGString<int8_t>* strs;
	get_int8_string_list(strs, num_str, max_string_len);

	ASSERT(num_str>0 && max_string_len>0);
	strings=SG_MALLOC(SGString<bool>, num_str);

	for(int32_t i = 0;i < num_str;i++)
	{
		strings[i].length = strs[i].length;
                strings[i].string = SG_MALLOC(bool, strs[i].length);
		for(int32_t j = 0;j < strs[i].length;j++)
		strings[i].string[j] = strs[i].string[j] != 0 ? 1 : 0;
	}

	for(int32_t i = 0;i < num_str;i++)
		SG_FREE(strs[i].string);
	SG_FREE(strs);
}

void CFile::set_string_list(const SGString<bool>* strings, int32_t num_str)
{
	SGString<int8_t> * strs = SG_MALLOC(SGString<int8_t>, num_str);

	for(int32_t i = 0;i < num_str;i++)
	{
		strs[i].length = strings[i].length;
		strs[i].string = SG_MALLOC(int8_t, strings[i].length);
		for(int32_t j = 0;j < strings[i].length;j++)
		strs[i].string[j] = strings[i].string[j] != 0 ? 1 : 0;
	}

	set_int8_string_list(strs,num_str);

	for(int32_t i = 0;i < num_str;i++)
		SG_FREE(strs[i].string);
	SG_FREE(strs);
}

CFile::~CFile()
{
	close();
}

void CFile::set_variable_name(const char* name)
{
	SG_FREE(variable_name);
	variable_name=strdup(name);
}

char* CFile::get_variable_name()
{
	return strdup(variable_name);
}
