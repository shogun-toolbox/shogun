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

#include "lib/File.h"

#include "features/StringFeatures.h"
#include "features/SparseFeatures.h"

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

void CFile::get_bool_vector(bool*& vector, int32_t& len)
{
	int32_t* int_vector;
	get_int_vector(int_vector, len);

	ASSERT(len>0);
	vector= new bool[len];

	for (int32_t i=0; i<len; i++)
		vector[i]= (int_vector[i]!=0);

	delete[] int_vector;
}

void CFile::set_bool_vector(const bool* vector, int32_t len)
{
	int32_t* int_vector = new int32_t[len];
	for (int32_t i=0;i<len;i++)
	{
		if (vector[i])
			int_vector[i]=1;
		else
			int_vector[i]=0;
	}
	set_int_vector(int_vector,len);
	delete[] int_vector;
}

void CFile::get_bool_matrix(bool*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	SG_NOTIMPLEMENTED;
}

void CFile::set_bool_matrix(const bool* matrix, int32_t num_feat, int32_t num_vec)
{
	SG_NOTIMPLEMENTED;
}

void CFile::get_bool_string_list(
		TString<bool>*& strings, int32_t& num_str,
		int32_t& max_string_len)
{
	TString<int32_t>* strs;
	get_int_string_list(strs, num_str, max_string_len);

	ASSERT(num_str>0 && max_string_len>0);
	strings=new TString<bool>[num_str];

	SG_NOTIMPLEMENTED;
	//FIXME
}

void CFile::set_bool_string_list(const TString<bool>* strings, int32_t num_str)
{
	SG_NOTIMPLEMENTED;
	//FIXME
}

CFile::~CFile()
{
	close();
}

void CFile::set_variable_name(const char* name)
{
	free(variable_name);
	variable_name=strdup(name);
}

char* CFile::get_variable_name()
{
	return strdup(variable_name);
}
