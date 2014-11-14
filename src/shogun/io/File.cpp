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
#include <string.h>

#include <shogun/io/File.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/SGObject.h>

#include <shogun/lib/memory.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGString.h>

using namespace shogun;

CFile::CFile() : CSGObject()
{
	file=NULL;
	filename=NULL;
	variable_name=NULL;
	task='\0';
}

CFile::CFile(FILE* f, const char* name) : CSGObject()
{
	file=f;
	filename=NULL;
	variable_name=NULL;
	task='\0';

	if (name)
		set_variable_name(name);
}

CFile::CFile(int fd, const char* mode, const char* name) : CSGObject()
{
	file=fdopen(fd, mode);
	filename=NULL;
	variable_name=NULL;
	task=mode[0];

	if (name)
		set_variable_name(name);
}

CFile::CFile(const char* fname, char rw, const char* name) : CSGObject()
{
	variable_name=NULL;
	task=rw;
	filename=get_strdup(fname);
	char mode[2];
	mode[0]=rw;
	mode[1]='\0';

	if (rw=='r' || rw == 'w')
	{
		if (filename)
		{
			if (!(file=fopen((const char*) filename, (const char*) mode)))
				SG_ERROR("Error opening file '%s'\n", filename)
		}
	}
	else
		SG_ERROR("unknown mode '%c'\n", mode[0])

	if (name)
		set_variable_name(name);
}

CFile::~CFile()
{
	close();
}

#define VECTOR_GETSET(type)                                 \
void CFile::set_vector(const type* vector, int32_t len)     \
{                                                           \
    int32_t* int_vector = SG_MALLOC(int32_t, len);          \
    for (int32_t i=0;i<len;i++)                             \
    {                                                       \
        if (vector[i])                                      \
            int_vector[i]=1;                                \
        else                                                \
            int_vector[i]=0;                                \
    }                                                       \
    set_vector(int_vector,len);                             \
    SG_FREE(int_vector);                                    \
}                                                           \
                                                            \
void CFile::get_vector(type*& vector, int32_t& len)         \
{                                                           \
    int32_t* int_vector;                                    \
    get_vector(int_vector, len);                            \
                                                            \
    ASSERT(len>0)                                           \
    vector= SG_MALLOC(type, len);                           \
                                                            \
    for (int32_t i=0; i<len; i++)                           \
        vector[i]= (int_vector[i]!=0);                      \
                                                            \
    SG_FREE(int_vector);                                    \
}
VECTOR_GETSET(bool)
VECTOR_GETSET(int8_t)
VECTOR_GETSET(uint8_t)
VECTOR_GETSET(char)
VECTOR_GETSET(int32_t)
VECTOR_GETSET(uint32_t)
VECTOR_GETSET(float32_t)
VECTOR_GETSET(float64_t)
VECTOR_GETSET(floatmax_t)
VECTOR_GETSET(int16_t)
VECTOR_GETSET(uint16_t)
VECTOR_GETSET(int64_t)
VECTOR_GETSET(uint64_t)

#undef VECTOR_GETSET

#define MATRIX_GETSET(type)                                                 \
void CFile::set_matrix(const type* matrix, int32_t num_feat, int32_t num_vec)   \
{                                                                           \
    uint8_t * byte_matrix = SG_MALLOC(uint8_t, num_feat*num_vec);           \
    for(int32_t i = 0;i < num_vec;i++)                                      \
    {                                                                       \
        for(int32_t j = 0;j < num_feat;j++)                                 \
            byte_matrix[i*num_feat+j] = matrix[i*num_feat+j] != 0 ? 1 : 0;  \
    }                                                                       \
                                                                            \
    set_matrix(byte_matrix,num_feat,num_vec);                               \
                                                                            \
    SG_FREE(byte_matrix);                                                   \
}                                                                           \
                                                                            \
void CFile::get_matrix(type*& matrix, int32_t& num_feat, int32_t& num_vec)  \
{                                                                           \
    uint8_t * byte_matrix;                                                  \
    get_matrix(byte_matrix,num_feat,num_vec);                               \
                                                                            \
    ASSERT(num_feat > 0 && num_vec > 0)                                     \
    matrix = SG_MALLOC(type, num_feat*num_vec);                             \
                                                                            \
    for(int32_t i = 0;i < num_vec;i++)                                      \
    {                                                                       \
        for(int32_t j = 0;j < num_feat;j++)                                 \
            matrix[i*num_feat+j] = byte_matrix[i*num_feat+j] != 0 ? 1 : 0;  \
    }                                                                       \
                                                                            \
    SG_FREE(byte_matrix);                                                   \
}
MATRIX_GETSET(bool)
MATRIX_GETSET(int8_t)
MATRIX_GETSET(uint8_t)
MATRIX_GETSET(char)
MATRIX_GETSET(int32_t)
MATRIX_GETSET(uint32_t)
MATRIX_GETSET(float32_t)
MATRIX_GETSET(float64_t)
MATRIX_GETSET(floatmax_t)
MATRIX_GETSET(int16_t)
MATRIX_GETSET(uint16_t)
MATRIX_GETSET(int64_t)
MATRIX_GETSET(uint64_t)

#undef MATRIX_GETSET

#define NDARRAY_GETTER(type)                                                \
void CFile::get_ndarray(type*& array, int32_t*& dims, int32_t& num_dims)    \
{                                                                           \
    SG_NOTIMPLEMENTED;                                                      \
}
NDARRAY_GETTER(uint8_t)
NDARRAY_GETTER(char)
NDARRAY_GETTER(int32_t)
NDARRAY_GETTER(float32_t)
NDARRAY_GETTER(float64_t)
NDARRAY_GETTER(int16_t)
NDARRAY_GETTER(uint16_t)

#undef NDARRAY_GETTER

void CFile::set_variable_name(const char* name)
{
	SG_FREE(variable_name);
	variable_name=get_strdup(name);
}

char* CFile::get_variable_name()
{
	return get_strdup(variable_name);
}

#define STRING_LIST_GETSET(type)                                    \
void CFile::get_string_list(SGString<type>*& strings, int32_t& num_str, int32_t& max_string_len)  \
{                                                                   \
    SGString<int8_t>* strs;                                         \
    get_string_list(strs, num_str, max_string_len);                 \
                                                                    \
    ASSERT(num_str>0 && max_string_len>0)                           \
    strings=SG_MALLOC(SGString<type>, num_str);                     \
                                                                    \
    for(int32_t i = 0;i < num_str;i++)                              \
    {                                                               \
        strings[i].slen = strs[i].slen;                             \
                strings[i].string = SG_MALLOC(type, strs[i].slen);  \
        for(int32_t j = 0;j < strs[i].slen;j++)                     \
        strings[i].string[j] = strs[i].string[j] != 0 ? 1 : 0;      \
    }                                                               \
                                                                    \
    for(int32_t i = 0;i < num_str;i++)                              \
        SG_FREE(strs[i].string);                                    \
                                                                    \
    SG_FREE(strs);                                                  \
}                                                                   \
                                                                    \
void CFile::set_string_list(const SGString<type>* strings, int32_t num_str) \
{                                                                   \
    SGString<int8_t> * strs = SG_MALLOC(SGString<int8_t>, num_str); \
                                                                    \
    for(int32_t i = 0;i < num_str;i++)                              \
    {                                                               \
        strs[i].slen = strings[i].slen;                             \
        strs[i].string = SG_MALLOC(int8_t, strings[i].slen);        \
        for(int32_t j = 0;j < strings[i].slen;j++)                  \
        strs[i].string[j] = strings[i].string[j] != 0 ? 1 : 0;      \
    }                                                               \
                                                                    \
    set_string_list(strs,num_str);                                  \
                                                                    \
    for(int32_t i = 0;i < num_str;i++)                              \
        SG_FREE(strs[i].string);                                    \
                                                                    \
    SG_FREE(strs);                                                  \
}
STRING_LIST_GETSET(bool)
STRING_LIST_GETSET(int8_t)
STRING_LIST_GETSET(uint8_t)
STRING_LIST_GETSET(char)
STRING_LIST_GETSET(int32_t)
STRING_LIST_GETSET(uint32_t)
STRING_LIST_GETSET(float32_t)
STRING_LIST_GETSET(float64_t)
STRING_LIST_GETSET(floatmax_t)
STRING_LIST_GETSET(int16_t)
STRING_LIST_GETSET(uint16_t)
STRING_LIST_GETSET(int64_t)
STRING_LIST_GETSET(uint64_t)

#undef STRING_LIST_GETSET

#define SPARSE_VECTOR_GETSET(type)                                      \
void CFile::set_sparse_vector(const SGSparseVectorEntry<type>* entries, int32_t num_feat) \
{                                                                       \
    SGSparseVector<type> v((SGSparseVectorEntry<type>*) entries, num_feat, false);  \
    set_sparse_matrix(&v, 0, 1);                                        \
}                                                                       \
                                                                        \
void CFile::get_sparse_vector(GSparseVectorEntry<type>*& entries, int32_t& num_feat)     \
{                                                                       \
    SGSparseVector<type>* v;                                            \
    int32_t dummy;                                                      \
    int32_t nvec;                                                       \
    get_sparse_matrix(v, dummy, nvec);                                  \
    ASSERT(nvec==1)                                                     \
    entries=v->features;                                                \
    num_feat=v->num_feat_entries;                                       \
}
SPARSE_VECTOR_GETSET(bool)
SPARSE_VECTOR_GETSET(int8_t)
SPARSE_VECTOR_GETSET(uint8_t)
SPARSE_VECTOR_GETSET(char)
SPARSE_VECTOR_GETSET(int32_t)
SPARSE_VECTOR_GETSET(uint32_t)
SPARSE_VECTOR_GETSET(float32_t)
SPARSE_VECTOR_GETSET(float64_t)
SPARSE_VECTOR_GETSET(floatmax_t)
SPARSE_VECTOR_GETSET(int16_t)
SPARSE_VECTOR_GETSET(uint16_t)
SPARSE_VECTOR_GETSET(int64_t)
SPARSE_VECTOR_GETSET(uint64_t)

#undef SPARSE_VECTOR_GETSET

#define SPARSE_MATRIX_GETSET(type)                                      \
void CFile::set_sparse_matrix(const SGSparseVector<type>* matrix, int32_t num_feat, int32_t num_vec)    \
{                                                                       \
    SG_NOTIMPLEMENTED;                                                  \
}                                                                       \
                                                                        \
void CFile::get_sparse_matrix(SGSparseVector<type>*& matrix, int32_t& num_feat, int32_t& num_vec)     \
{                                                                       \
    SG_NOTIMPLEMENTED;                                                  \
}
SPARSE_MATRIX_GETSET(bool)
SPARSE_MATRIX_GETSET(int8_t)
SPARSE_MATRIX_GETSET(uint8_t)
SPARSE_MATRIX_GETSET(char)
SPARSE_MATRIX_GETSET(int32_t)
SPARSE_MATRIX_GETSET(uint32_t)
SPARSE_MATRIX_GETSET(float32_t)
SPARSE_MATRIX_GETSET(float64_t)
SPARSE_MATRIX_GETSET(floatmax_t)
SPARSE_MATRIX_GETSET(int16_t)
SPARSE_MATRIX_GETSET(uint16_t)
SPARSE_MATRIX_GETSET(int64_t)
SPARSE_MATRIX_GETSET(uint64_t)

#undef SPARSE_MATRIX_GETSET

char* CFile::read_whole_file(char* fname, size_t& len)
{
    FILE* tmpf=fopen(fname, "r");
    ASSERT(tmpf)
    fseek(tmpf,0,SEEK_END);
    len=ftell(tmpf);
    ASSERT(len>0)
    rewind(tmpf);
    char* result = SG_MALLOC(char, len);
    size_t total=fread(result,1,len,tmpf);
    ASSERT(total==len)
    fclose(tmpf);
    return result;
}
