/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein, Heiko Strathmann, 
 *          Chiyuan Zhang, Viktor Gal
 */

#include <stdio.h>
#include <string.h>

#include <shogun/io/File.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/SGObject.h>

#include <shogun/lib/memory.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

File::File() : SGObject()
{
	file=NULL;
	filename=NULL;
	variable_name=NULL;
	task='\0';
}

File::File(FILE* f, const char* name) : SGObject()
{
	file=f;
	filename=NULL;
	variable_name=NULL;
	task='\0';

	if (name)
		set_variable_name(name);
}

#ifdef HAVE_FDOPEN
File::File(int fd, const char* mode, const char* name) : SGObject()
{
	file=fdopen(fd, mode);
	filename=NULL;
	variable_name=NULL;
	task=mode[0];

	if (name)
		set_variable_name(name);
}
#endif

File::File(const char* fname, char rw, const char* name) : SGObject()
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
				error("Error opening file '{}'", filename);
		}
	}
	else
		error("unknown mode '{}'", mode[0]);

	if (name)
		set_variable_name(name);
}

void File::get_vector(bool*& vector, int32_t& len)
{
	int32_t* int_vector;
	get_vector(int_vector, len);

	ASSERT(len>0)
	vector= SG_MALLOC(bool, len);

	for (int32_t i=0; i<len; i++)
		vector[i]= (int_vector[i]!=0);

	SG_FREE(int_vector);
}

void File::set_vector(const bool* vector, int32_t len)
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

void File::get_matrix(bool*& matrix, int32_t& num_feat, int32_t& num_vec)
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

void File::set_matrix(const bool* matrix, int32_t num_feat, int32_t num_vec)
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

void File::get_string_list(
		SGVector<bool>*& strings, int32_t& num_str,
		int32_t& max_string_len)
{
	SGVector<int8_t>* strs;
	get_string_list(strs, num_str, max_string_len);

	ASSERT(num_str>0 && max_string_len>0)
	strings=SG_MALLOC(SGVector<bool>, num_str);

	for(int32_t i = 0;i < num_str;i++)
	{
		strings[i] = SGVector<bool>(strs[i].vlen);
		for(int32_t j = 0;j < strs[i].vlen;j++)
		strings[i].vector[j] = strs[i].vector[j] != 0 ? 1 : 0;
	}

	SG_FREE(strs);
}

void File::set_string_list(const SGVector<bool>* strings, int32_t num_str)
{
	SGVector<int8_t> * strs = SG_MALLOC(SGVector<int8_t>, num_str);

	for(int32_t i = 0;i < num_str;i++)
	{
		strs[i] = SGVector<int8_t>(strings[i].vlen);
		for(int32_t j = 0;j < strings[i].vlen;j++)
		strs[i].vector[j] = strings[i].vector[j] != 0 ? 1 : 0;
	}

	set_string_list(strs,num_str);

	SG_FREE(strs);
}

File::~File()
{
	close();
}

void File::set_variable_name(const char* name)
{
	SG_FREE(variable_name);
	variable_name=get_strdup(name);
}

char* File::get_variable_name()
{
	return get_strdup(variable_name);
}

#define SPARSE_VECTOR_GETTER(type)										\
void File::set_sparse_vector(											\
			const SGSparseVectorEntry<type>* entries, int32_t num_feat)	\
{																		\
	SGSparseVector<type> v((SGSparseVectorEntry<type>*) entries, num_feat, false);	\
	set_sparse_matrix(&v, 0, 1);										\
}																		\
																		\
void File::get_sparse_vector(											\
			SGSparseVectorEntry<type>*& entries, int32_t& num_feat)		\
{																		\
	SGSparseVector<type>* v;											\
	int32_t dummy;														\
	int32_t nvec;														\
	get_sparse_matrix(v, dummy, nvec);									\
	ASSERT(nvec==1)													\
	entries=v->features;												\
	num_feat=v->num_feat_entries;										\
}
SPARSE_VECTOR_GETTER(bool)
SPARSE_VECTOR_GETTER(int8_t)
SPARSE_VECTOR_GETTER(uint8_t)
SPARSE_VECTOR_GETTER(char)
SPARSE_VECTOR_GETTER(int32_t)
SPARSE_VECTOR_GETTER(uint32_t)
SPARSE_VECTOR_GETTER(float32_t)
SPARSE_VECTOR_GETTER(float64_t)
SPARSE_VECTOR_GETTER(floatmax_t)
SPARSE_VECTOR_GETTER(int16_t)
SPARSE_VECTOR_GETTER(uint16_t)
SPARSE_VECTOR_GETTER(int64_t)
SPARSE_VECTOR_GETTER(uint64_t)

#undef SPARSE_VECTOR_GETTER


char* File::read_whole_file(char* fname, size_t& len)
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
