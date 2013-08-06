/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Parts of this code are copyright (c) 2009 Yahoo! Inc.
 * All rights reserved.  The copyrights embodied in the content of
 * this file are licensed under the BSD (revised) open source license.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/features/SparseFeatures.h>
#include <shogun/io/File.h>
#include <shogun/io/AsciiFile.h>
#include <shogun/mathematics/Math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

using namespace shogun;

CAsciiFile::CAsciiFile()
{
	SG_UNSTABLE("CAsciiFile::CAsciiFile()", "\n")
}

CAsciiFile::CAsciiFile(FILE* f, const char* name) : CFile(f, name)
{
}

CAsciiFile::CAsciiFile(const char* fname, char rw, const char* name) : CFile(fname, rw, name)
{
}

CAsciiFile::~CAsciiFile()
{
}

#define GET_VECTOR(fname, mfname, sg_type) \
void CAsciiFile::fname(sg_type*& vec, int32_t& len) \
{													\
	vec=NULL;										\
	len=0;											\
	int32_t num_feat=0;								\
	int32_t num_vec=0;								\
	mfname(vec, num_feat, num_vec);					\
	if ((num_feat==1) || (num_vec==1))				\
	{												\
		if (num_feat==1)							\
			len=num_vec;							\
		else										\
			len=num_feat;							\
	}												\
	else											\
	{												\
		SG_FREE(vec);								\
		vec=NULL;									\
		len=0;										\
		SG_ERROR("Could not read vector from"		\
				" file %s (shape %dx%d found but "	\
				"vector expected).\n", filename,	\
				num_vec, num_feat);					\
	}												\
}

GET_VECTOR(get_vector, get_matrix, int8_t)
GET_VECTOR(get_vector, get_matrix, uint8_t)
GET_VECTOR(get_vector, get_matrix, char)
GET_VECTOR(get_vector, get_matrix, int32_t)
GET_VECTOR(get_vector, get_matrix, uint32_t)
GET_VECTOR(get_vector, get_matrix, float32_t)
GET_VECTOR(get_vector, get_matrix, float64_t)
GET_VECTOR(get_vector, get_matrix, floatmax_t)
GET_VECTOR(get_vector, get_matrix, int16_t)
GET_VECTOR(get_vector, get_matrix, uint16_t)
GET_VECTOR(get_vector, get_matrix, int64_t)
GET_VECTOR(get_vector, get_matrix, uint64_t)
#undef GET_VECTOR

#define GET_MATRIX(fname, conv, sg_type)										\
void CAsciiFile::fname(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{																				\
	struct stat stats;															\
	if (stat(filename, &stats)!=0)												\
		SG_ERROR("Could not get file statistics.\n")							\
																				\
	char* data=SG_CALLOC(char, stats.st_size+1);								\
	size_t nread=fread(data, sizeof(char), stats.st_size, file);				\
	if (nread<=0)																\
		SG_ERROR("Could not read data from %s.\n", filename)					\
																				\
	SG_DEBUG("data read from file:\n%s\n", data)								\
																				\
	/* determine num_feat and num_vec, populate dynamic array */ 				\
	int32_t nf=0;																\
	num_feat=0;																	\
	num_vec=0;																	\
	char* ptr_item=NULL;														\
	char* ptr_data=data;														\
	DynArray<char*>* items=new DynArray<char*>();						\
																				\
	while (*ptr_data)															\
	{																			\
		if (*ptr_data=='\n')													\
		{																		\
			if (ptr_item)														\
				nf++;															\
																				\
			if (num_feat!=0 && nf!=num_feat)									\
				SG_ERROR("Number of features mismatches (%d != %d) in vector"	\
						" %d in file %s.\n", num_feat, nf, num_vec, filename);	\
																				\
			append_item(items, ptr_data, ptr_item);								\
			num_feat=nf;														\
			num_vec++;															\
			nf=0;																\
			ptr_item=NULL;														\
		}																		\
		else if (!isblank(*ptr_data) && !ptr_item)								\
		{																		\
			ptr_item=ptr_data;													\
		}																		\
		else if (isblank(*ptr_data) && ptr_item)								\
		{																		\
			append_item(items, ptr_data, ptr_item);								\
			ptr_item=NULL;														\
			nf++;																\
		}																		\
																				\
		ptr_data++;																\
	}																			\
																				\
	SG_DEBUG("num feat: %d, num_vec %d\n", num_feat, num_vec)					\
	SG_FREE(data);																\
																				\
	/* now copy data into matrix */ 											\
	matrix=SG_MALLOC(sg_type, num_vec*num_feat);										\
	for (int32_t i=0; i<num_vec; i++)											\
	{																			\
		for (int32_t j=0; j<num_feat; j++)										\
		{																		\
			char* item=items->get_element(i*num_feat+j);						\
			matrix[i*num_feat+j]=conv(item);									\
			SG_FREE(item);														\
		}																		\
	}																			\
	delete items;																\
}

GET_MATRIX(get_matrix, atoi, uint8_t)
GET_MATRIX(get_matrix, atoi, int8_t)
GET_MATRIX(get_matrix, atoi, char)
GET_MATRIX(get_matrix, atoi, int32_t)
GET_MATRIX(get_matrix, atoi, uint32_t)
GET_MATRIX(get_matrix, atoll, int64_t)
GET_MATRIX(get_matrix, atoll, uint64_t)
GET_MATRIX(get_matrix, atof, float32_t)
GET_MATRIX(get_matrix, atof, float64_t)
GET_MATRIX(get_matrix, atof, floatmax_t)
GET_MATRIX(get_matrix, atoi, int16_t)
GET_MATRIX(get_matrix, atoi, uint16_t)
#undef GET_MATRIX

#define GET_NDARRAY(fname, conv, sg_type)							\
void CAsciiFile::fname(sg_type*& array, int32_t *& dims, int32_t & num_dims)			\
{												\
	struct stat stats;									\
	if (stat(filename, &stats)!=0)								\
		SG_ERROR("Could not get file statistics.\n")					\
												\
	char* data=SG_MALLOC(char, stats.st_size+1);							\
	memset(data, 0, sizeof(char)*(stats.st_size+1));					\
	size_t nread=fread(data, sizeof(char), stats.st_size, file);				\
	if (nread<=0)										\
		SG_ERROR("Could not read data from %s.\n", filename)				\
												\
	SG_DEBUG("data read from file:\n%s\n", data)						\
												\
	/* determine size of array */ 								\
	int32_t length=0;									\
	int32_t counter=0;                                                              	\
	size_t total=0;                                     					\
        num_dims = -1;                          						\
	char* ptr_item=NULL;									\
	char* ptr_data=data;									\
	DynArray<char*>* items=new DynArray<char*>();						\
                                                                                                \
        /* read line with sizes of array*/                          				\
        while(*ptr_data != '\n')                                            			\
        {                                                                                       \
            if(isblank(*ptr_data) && ptr_item)                          			\
            {                                                                                   \
                append_item(items, ptr_data, ptr_item);     					\
                num_dims++;                                                             	\
                ptr_item = NULL;                                                        	\
            }                                                                                   \
            else if(!isblank(*ptr_data) && !ptr_item)               				\
                ptr_item = ptr_data;                                                        	\
                                                                                                \
            ptr_data++;                                                                     	\
        }                                                                                       \
        ptr_item = NULL;                                                                        \
        ptr_data++;                                                                             \
        											\
	/* read array data*/                                                                    \
	while(*ptr_data)									\
	{											\
		if (*ptr_data=='\n')								\
		{										\
			if (ptr_item)								\
				counter++;							\
												\
			if (length!=0 && counter!=length)					\
				SG_ERROR("Invalid number of data (%d != %d) in line"		\
				" %d in file %s.\n", length, counter, total, filename);		\
												\
			append_item(items, ptr_data, ptr_item);					\
			length=counter;								\
			total++;								\
			counter=0;								\
			ptr_item=NULL;								\
		}										\
		else if (!isblank(*ptr_data) && !ptr_item)					\
		{										\
			ptr_item=ptr_data;							\
		}										\
		else if (isblank(*ptr_data) && ptr_item)					\
		{										\
			append_item(items, ptr_data, ptr_item);					\
			ptr_item=NULL;								\
			counter++;								\
		}										\
												\
		ptr_data++;									\
	}											\
												\
	SG_DEBUG("num of data in line: %d, num of lines %d\n", counter, total)			\
	SG_FREE(data);										\
												\
	/* determining sizes of dimensions*/                                                	\
        char * item;                                                                            \
        item=items->get_element(0);                                                             \
        if(atoi(item) != num_dims)                                                              \
            SG_ERROR("Invalid number of dimensions!\n")                            		\
        SG_FREE(item);                                                                          \
        dims = SG_MALLOC(int32_t, num_dims);                                                           \
        for(int32_t i =0;i < num_dims;i++)                                              	\
        {                                                                                       \
            item = items->get_element(i+1);                                 			\
            dims[i] = atoi(item);                                                           	\
            SG_FREE(item);                                                                      \
        }                                                                                       \
        if (dims[num_dims-1] != length)                                                 	\
            SG_ERROR("Invalid number of lines in file!\n")                 			\
                                                                                    		\
        /* converting array data */								\
        total *= length;									\
	array=SG_MALLOC(sg_type, total);        							\
	for (size_t i=0; i<total; i++)								\
	{											\
			item=items->get_element(i+(num_dims+1));				\
			array[i]=conv(item);							\
			SG_FREE(item);								\
	}											\
	delete items;										\
}

GET_NDARRAY(get_ndarray, atoi, uint8_t)
GET_NDARRAY(get_ndarray, atoi, int8_t)
GET_NDARRAY(get_ndarray, atoi, char)
GET_NDARRAY(get_ndarray, atoi, int32_t)
GET_NDARRAY(get_ndarray, atoi, uint32_t)
GET_NDARRAY(get_ndarray, atoll, int64_t)
GET_NDARRAY(get_ndarray, atoll, uint64_t)
GET_NDARRAY(get_ndarray, atof, float32_t)
GET_NDARRAY(get_ndarray, atof, float64_t)
GET_NDARRAY(get_ndarray, atof, floatmax_t)
GET_NDARRAY(get_ndarray, atoi, int16_t)
GET_NDARRAY(get_ndarray, atoi, uint16_t)
#undef GET_NDARRAY

#define GET_SPARSEMATRIX(fname, conv, sg_type)										\
void CAsciiFile::fname(SGSparseVector<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{	\
	size_t blocksize=1024*1024;	\
	size_t required_blocksize=blocksize;	\
	uint8_t* dummy=SG_MALLOC(uint8_t, blocksize);	\
	\
	if (file)	\
	{	\
		num_vec=0;	\
		num_feat=0;	\
	\
		SG_INFO("counting line numbers in file %s\n", filename)	\
		size_t sz=blocksize;	\
		size_t block_offs=0;	\
		size_t old_block_offs=0;	\
		fseek(file, 0, SEEK_END);	\
		size_t fsize=ftell(file);	\
		rewind(file);	\
	\
		while (sz == blocksize)	\
		{	\
			sz=fread(dummy, sizeof(uint8_t), blocksize, file);	\
			for (size_t i=0; i<sz; i++)	\
			{	\
				block_offs++;	\
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))	\
				{	\
					num_vec++;	\
					required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs+1);	\
					old_block_offs=block_offs;	\
				}	\
			}	\
			SG_PROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t")	\
		}	\
	\
		SG_INFO("found %d feature vectors\n", num_vec)	\
		SG_FREE(dummy);	\
		blocksize=required_blocksize;	\
		dummy = SG_MALLOC(uint8_t, blocksize+1); /*allow setting of '\0' at EOL*/	\
		matrix=SG_MALLOC(SGSparseVector<sg_type>, num_vec);	\
		for (int i=0; i<num_vec; i++)	\
			new (&matrix[i]) SGSparseVector<sg_type>();	\
		rewind(file);	\
		sz=blocksize;	\
		int32_t lines=0;	\
		while (sz == blocksize)	\
		{	\
			sz=fread(dummy, sizeof(uint8_t), blocksize, file);	\
	\
			size_t old_sz=0;	\
			for (size_t i=0; i<sz; i++)	\
			{	\
				if (i==sz-1 && dummy[i]!='\n' && sz==blocksize)	\
				{	\
					size_t len=i-old_sz+1;	\
					uint8_t* data=&dummy[old_sz];	\
	\
					for (size_t j=0; j<len; j++)	\
						dummy[j]=data[j];	\
	\
					sz=fread(dummy+len, sizeof(uint8_t), blocksize-len, file);	\
					i=0;	\
					old_sz=0;	\
					sz+=len;	\
				}	\
	\
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))	\
				{	\
	\
					size_t len=i-old_sz;	\
					uint8_t* data=&dummy[old_sz];	\
	\
					int32_t dims=0;	\
					for (size_t j=0; j<len; j++)	\
					{	\
						if (data[j]==':')	\
							dims++;	\
					}	\
	\
					if (dims<=0)	\
					{	\
						SG_ERROR("Error in line %d - number of"	\
								" dimensions is %d line is %d characters"	\
								" long\n line_content:'%.*s'\n", lines,	\
								dims, len, len, (const char*) data);	\
					}	\
	\
					SGSparseVectorEntry<sg_type>* feat=SG_MALLOC(SGSparseVectorEntry<sg_type>, dims);	\
	\
					/* skip label part */	\
					size_t j=0;	\
					for (; j<len; j++)	\
					{	\
						if (data[j]==':')	\
						{	\
							j=-1; /* file without label*/	\
							break;	\
						}	\
	\
						if (data[j]==' ')	\
						{	\
							data[j]='\0';	\
	\
							/* skip label part */	\
							break;	\
						}	\
					}	\
	\
					int32_t d=0;	\
					j++;	\
					uint8_t* start=&data[j];	\
					for (; j<len; j++)	\
					{	\
						if (data[j]==':')	\
						{	\
							data[j]='\0';	\
	\
							feat[d].feat_index=(int32_t) atoi((const char*) start)-1;	\
							num_feat=CMath::max(num_feat, feat[d].feat_index+1);	\
	\
							j++;	\
							start=&data[j];	\
							for (; j<len; j++)	\
							{	\
								if (data[j]==' ' || data[j]=='\n')	\
								{	\
									data[j]='\0';	\
									feat[d].entry=(sg_type) conv((const char*) start);	\
									d++;	\
									break;	\
								}	\
							}	\
	\
							if (j==len)	\
							{	\
								data[j]='\0';	\
								feat[dims-1].entry=(sg_type) conv((const char*) start);	\
							}	\
	\
							j++;	\
							start=&data[j];	\
						}	\
					}	\
	\
					matrix[lines].num_feat_entries=dims;	\
					matrix[lines].features=feat;	\
	\
					old_sz=i+1;	\
					lines++;	\
					SG_PROGRESS(lines, 0, num_vec, 1, "LOADING:\t")	\
				}	\
			}	\
		}	\
	\
		SG_INFO("file successfully read\n")	\
	}	\
	\
	SG_FREE(dummy);	\
}

GET_SPARSEMATRIX(get_sparse_matrix, atoi, bool)
GET_SPARSEMATRIX(get_sparse_matrix, atoi, uint8_t)
GET_SPARSEMATRIX(get_sparse_matrix, atoi, int8_t)
GET_SPARSEMATRIX(get_sparse_matrix, atoi, char)
GET_SPARSEMATRIX(get_sparse_matrix, atoi, int32_t)
GET_SPARSEMATRIX(get_sparse_matrix, atoi, uint32_t)
GET_SPARSEMATRIX(get_sparse_matrix, atoll, int64_t)
GET_SPARSEMATRIX(get_sparse_matrix, atoll, uint64_t)
GET_SPARSEMATRIX(get_sparse_matrix, atof, float32_t)
GET_SPARSEMATRIX(get_sparse_matrix, atof, float64_t)
GET_SPARSEMATRIX(get_sparse_matrix, atof, floatmax_t)
GET_SPARSEMATRIX(get_sparse_matrix, atoi, int16_t)
GET_SPARSEMATRIX(get_sparse_matrix, atoi, uint16_t)
#undef GET_SPARSEMATRIX


void CAsciiFile::get_string_list(SGString<uint8_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	size_t blocksize=1024*1024;
	size_t required_blocksize=0;
	uint8_t* dummy=SG_MALLOC(uint8_t, blocksize);
	uint8_t* overflow=NULL;
	int32_t overflow_len=0;

	if (file)
	{
		num_str=0;
		max_string_len=0;

		SG_INFO("counting line numbers in file %s\n", filename)
		size_t sz=blocksize;
		size_t block_offs=0;
		size_t old_block_offs=0;
		fseek(file, 0, SEEK_END);
		size_t fsize=ftell(file);
		rewind(file);

		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(uint8_t), blocksize, file);
			for (size_t i=0; i<sz; i++)
			{
				block_offs++;
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					num_str++;
					required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs);
					old_block_offs=block_offs;
				}
			}
			SG_PROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t")
		}

		SG_INFO("found %d strings\n", num_str)
		SG_DEBUG("block_size=%d\n", required_blocksize)
		SG_FREE(dummy);
		blocksize=required_blocksize;
		dummy=SG_MALLOC(uint8_t, blocksize);
		overflow=SG_MALLOC(uint8_t, blocksize);
		strings=SG_MALLOC(SGString<uint8_t>, num_str);

		rewind(file);
		sz=blocksize;
		int32_t lines=0;
		size_t old_sz=0;
		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(uint8_t), blocksize, file);

			old_sz=0;
			for (size_t i=0; i<sz; i++)
			{
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					int32_t len=i-old_sz;
					max_string_len=CMath::max(max_string_len, len+overflow_len);

					strings[lines].slen=len+overflow_len;
					strings[lines].string=SG_MALLOC(uint8_t, len+overflow_len);

					for (int32_t j=0; j<overflow_len; j++)
						strings[lines].string[j]=overflow[j];
					for (int32_t j=0; j<len; j++)
						strings[lines].string[j+overflow_len]=dummy[old_sz+j];

					// clear overflow
					overflow_len=0;

					//CMath::display_vector(strings[lines].string, len);
					old_sz=i+1;
					lines++;
					SG_PROGRESS(lines, 0, num_str, 1, "LOADING:\t")
				}
			}

			for (size_t i=old_sz; i<sz; i++)
				overflow[i-old_sz]=dummy[i];

			overflow_len=sz-old_sz;
		}
		SG_INFO("file successfully read\n")
		SG_INFO("max_string_length=%d\n", max_string_len)
		SG_INFO("num_strings=%d\n", num_str)
	}

	SG_FREE(dummy);
	SG_FREE(overflow);
}

void CAsciiFile::get_string_list(SGString<int8_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	size_t blocksize=1024*1024;
	size_t required_blocksize=0;
	int8_t* dummy=SG_MALLOC(int8_t, blocksize);
	int8_t* overflow=NULL;
	int32_t overflow_len=0;

	if (file)
	{
		num_str=0;
		max_string_len=0;

		SG_INFO("counting line numbers in file %s\n", filename)
		size_t sz=blocksize;
		size_t block_offs=0;
		size_t old_block_offs=0;
		fseek(file, 0, SEEK_END);
		size_t fsize=ftell(file);
		rewind(file);

		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(int8_t), blocksize, file);
			for (size_t i=0; i<sz; i++)
			{
				block_offs++;
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					num_str++;
					required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs);
					old_block_offs=block_offs;
				}
			}
			SG_PROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t")
		}

		SG_INFO("found %d strings\n", num_str)
		SG_DEBUG("block_size=%d\n", required_blocksize)
		SG_FREE(dummy);
		blocksize=required_blocksize;
		dummy=SG_MALLOC(int8_t, blocksize);
		overflow=SG_MALLOC(int8_t, blocksize);
		strings=SG_MALLOC(SGString<int8_t>, num_str);

		rewind(file);
		sz=blocksize;
		int32_t lines=0;
		size_t old_sz=0;
		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(int8_t), blocksize, file);

			old_sz=0;
			for (size_t i=0; i<sz; i++)
			{
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					int32_t len=i-old_sz;
					max_string_len=CMath::max(max_string_len, len+overflow_len);

					strings[lines].slen=len+overflow_len;
					strings[lines].string=SG_MALLOC(int8_t, len+overflow_len);

					for (int32_t j=0; j<overflow_len; j++)
						strings[lines].string[j]=overflow[j];
					for (int32_t j=0; j<len; j++)
						strings[lines].string[j+overflow_len]=dummy[old_sz+j];

					// clear overflow
					overflow_len=0;

					//CMath::display_vector(strings[lines].string, len);
					old_sz=i+1;
					lines++;
					SG_PROGRESS(lines, 0, num_str, 1, "LOADING:\t")
				}
			}

			for (size_t i=old_sz; i<sz; i++)
				overflow[i-old_sz]=dummy[i];

			overflow_len=sz-old_sz;
		}
		SG_INFO("file successfully read\n")
		SG_INFO("max_string_length=%d\n", max_string_len)
		SG_INFO("num_strings=%d\n", num_str)
	}

	SG_FREE(dummy);
	SG_FREE(overflow);
}

void CAsciiFile::get_string_list(SGString<char>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	size_t blocksize=1024*1024;
	size_t required_blocksize=0;
	char* dummy=SG_MALLOC(char, blocksize);
	char* overflow=NULL;
	int32_t overflow_len=0;

	if (file)
	{
		num_str=0;
		max_string_len=0;

		SG_INFO("counting line numbers in file %s\n", filename)
		size_t sz=blocksize;
		size_t block_offs=0;
		size_t old_block_offs=0;
		fseek(file, 0, SEEK_END);
		size_t fsize=ftell(file);
		rewind(file);

		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(char), blocksize, file);
			for (size_t i=0; i<sz; i++)
			{
				block_offs++;
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					num_str++;
					required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs);
					old_block_offs=block_offs;
				}
			}
			SG_PROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t")
		}

		SG_INFO("found %d strings\n", num_str)
		SG_DEBUG("block_size=%d\n", required_blocksize)
		SG_FREE(dummy);
		blocksize=required_blocksize;
		dummy=SG_MALLOC(char, blocksize);
		overflow=SG_MALLOC(char, blocksize);
		strings=SG_MALLOC(SGString<char>, num_str);

		rewind(file);
		sz=blocksize;
		int32_t lines=0;
		size_t old_sz=0;
		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(char), blocksize, file);

			old_sz=0;
			for (size_t i=0; i<sz; i++)
			{
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					int32_t len=i-old_sz;
					max_string_len=CMath::max(max_string_len, len+overflow_len);

					strings[lines].slen=len+overflow_len;
					strings[lines].string=SG_MALLOC(char, len+overflow_len);

					for (int32_t j=0; j<overflow_len; j++)
						strings[lines].string[j]=overflow[j];
					for (int32_t j=0; j<len; j++)
						strings[lines].string[j+overflow_len]=dummy[old_sz+j];

					// clear overflow
					overflow_len=0;

					//CMath::display_vector(strings[lines].string, len);
					old_sz=i+1;
					lines++;
					SG_PROGRESS(lines, 0, num_str, 1, "LOADING:\t")
				}
			}

			for (size_t i=old_sz; i<sz; i++)
				overflow[i-old_sz]=dummy[i];

			overflow_len=sz-old_sz;
		}
		SG_INFO("file successfully read\n")
		SG_INFO("max_string_length=%d\n", max_string_len)
		SG_INFO("num_strings=%d\n", num_str)
	}

	SG_FREE(dummy);
	SG_FREE(overflow);
}

void CAsciiFile::get_string_list(SGString<int32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_string_list(SGString<uint32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_string_list(SGString<int16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_string_list(SGString<uint16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_string_list(SGString<int64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_string_list(SGString<uint64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_string_list(SGString<float32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_string_list(SGString<float64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_string_list(SGString<floatmax_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}


/** set functions - to pass data from shogun to the target interface */

#define SET_VECTOR(fname, mfname, sg_type)	\
void CAsciiFile::fname(const sg_type* vec, int32_t len)	\
{															\
	mfname(vec, len, 1);									\
}
SET_VECTOR(set_vector, set_matrix, int8_t)
SET_VECTOR(set_vector, set_matrix, uint8_t)
SET_VECTOR(set_vector, set_matrix, char)
SET_VECTOR(set_vector, set_matrix, int32_t)
SET_VECTOR(set_vector, set_matrix, uint32_t)
SET_VECTOR(set_vector, set_matrix, float32_t)
SET_VECTOR(set_vector, set_matrix, float64_t)
SET_VECTOR(set_vector, set_matrix, floatmax_t)
SET_VECTOR(set_vector, set_matrix, int16_t)
SET_VECTOR(set_vector, set_matrix, uint16_t)
SET_VECTOR(set_vector, set_matrix, int64_t)
SET_VECTOR(set_vector, set_matrix, uint64_t)
#undef SET_VECTOR

#define SET_MATRIX(fname, sg_type, fprt_type, type_str) \
void CAsciiFile::fname(const sg_type* matrix, int32_t num_feat, int32_t num_vec)	\
{																					\
	if (!(file && matrix))															\
		SG_ERROR("File or matrix invalid.\n")										\
																					\
	for (int32_t i=0; i<num_vec; i++)												\
	{																				\
		for (int32_t j=0; j<num_feat; j++)											\
		{																			\
			sg_type v=matrix[num_feat*i+j];											\
			if (j==num_feat-1)														\
				fprintf(file, type_str "\n", (fprt_type) v);						\
			else																	\
				fprintf(file, type_str " ", (fprt_type) v);							\
		}																			\
	}																				\
}
SET_MATRIX(set_matrix, char, char, "%c")
SET_MATRIX(set_matrix, uint8_t, uint8_t, "%u")
SET_MATRIX(set_matrix, int8_t, int8_t, "%d")
SET_MATRIX(set_matrix, int32_t, int32_t, "%i")
SET_MATRIX(set_matrix, uint32_t, uint32_t, "%u")
SET_MATRIX(set_matrix, int64_t, long long int, "%lli")
SET_MATRIX(set_matrix, uint64_t, long long unsigned int, "%llu")
SET_MATRIX(set_matrix, int16_t, int16_t, "%i")
SET_MATRIX(set_matrix, uint16_t, uint16_t, "%u")
SET_MATRIX(set_matrix, float32_t, float32_t, "%.16g")
SET_MATRIX(set_matrix, float64_t, float64_t, "%.16lg")
SET_MATRIX(set_matrix, floatmax_t, floatmax_t, "%.16Lg")
#undef SET_MATRIX

#define SET_NDARRAY(fname, sg_type, fprt_type, type_str) \
void CAsciiFile::fname(const sg_type* array, int32_t * dims, int32_t num_dims)	\
{										\
	if (!(file && array))							\
		SG_ERROR("File or data invalid.\n")				\
										\
        size_t total = 1;   							\
        for(int i = 0;i < num_dims;i++)        					\
            total *= dims[i];                                   		\
        int32_t block_size = dims[num_dims-1];                                  \
                                                                		\
        fprintf(file,"%d ",num_dims); 						\
        for(int i = 0;i < num_dims;i++) 					\
            fprintf(file,"%d ",dims[i]);    					\
        fprintf(file,"\n"); 							\
                                                                                \
        for (size_t i=0; i < total; i++)					\
	{									\
		sg_type v= array[i];						\
		if ( ((i+1) % block_size) == 0)					\
			fprintf(file, type_str "\n", (fprt_type) v);		\
		else								\
			fprintf(file, type_str " ", (fprt_type) v);		\
	}									\
}

SET_NDARRAY(set_ndarray, char, char, "%c")
SET_NDARRAY(set_ndarray, uint8_t, uint8_t, "%u")
SET_NDARRAY(set_ndarray, int8_t, int8_t, "%d")
SET_NDARRAY(set_ndarray, int32_t, int32_t, "%i")
SET_NDARRAY(set_ndarray, uint32_t, uint32_t, "%u")
SET_NDARRAY(set_ndarray, int64_t, long long int, "%lli")
SET_NDARRAY(set_ndarray, uint64_t, long long unsigned int, "%llu")
SET_NDARRAY(set_ndarray, int16_t, int16_t, "%i")
SET_NDARRAY(set_ndarray, uint16_t, uint16_t, "%u")
SET_NDARRAY(set_ndarray, float32_t, float32_t, "%f")
SET_NDARRAY(set_ndarray, float64_t, float64_t, "%f")
SET_NDARRAY(set_ndarray, floatmax_t, floatmax_t, "%Lf")
#undef SET_NDARRAY

#define SET_SPARSEMATRIX(fname, sg_type, fprt_type, type_str) \
void CAsciiFile::fname(const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec)	\
{																							\
	if (!(file && matrix))																	\
		SG_ERROR("File or matrix invalid.\n")												\
																							\
	for (int32_t i=0; i<num_vec; i++)														\
	{																						\
		SGSparseVectorEntry<sg_type>* vec = matrix[i].features;									\
		int32_t len=matrix[i].num_feat_entries;												\
																							\
		for (int32_t j=0; j<len; j++)														\
		{																					\
			if (j<len-1)																	\
			{																				\
				fprintf(file, "%d:" type_str " ",											\
						(int32_t) vec[j].feat_index+1, (fprt_type) vec[j].entry);			\
			}																				\
			else																			\
			{																				\
				fprintf(file, "%d:" type_str "\n",											\
						(int32_t) vec[j].feat_index+1, (fprt_type) vec[j].entry);			\
			}																				\
		}																					\
	}																						\
}
SET_SPARSEMATRIX(set_sparse_matrix, bool, uint8_t, "%u")
SET_SPARSEMATRIX(set_sparse_matrix, char, char, "%c")
SET_SPARSEMATRIX(set_sparse_matrix, uint8_t, uint8_t, "%u")
SET_SPARSEMATRIX(set_sparse_matrix, int8_t, int8_t, "%d")
SET_SPARSEMATRIX(set_sparse_matrix, int32_t, int32_t, "%i")
SET_SPARSEMATRIX(set_sparse_matrix, uint32_t, uint32_t, "%u")
SET_SPARSEMATRIX(set_sparse_matrix, int64_t, long long int, "%lli")
SET_SPARSEMATRIX(set_sparse_matrix, uint64_t, long long unsigned int, "%llu")
SET_SPARSEMATRIX(set_sparse_matrix, int16_t, int16_t, "%i")
SET_SPARSEMATRIX(set_sparse_matrix, uint16_t, uint16_t, "%u")
SET_SPARSEMATRIX(set_sparse_matrix, float32_t, float32_t, "%f")
SET_SPARSEMATRIX(set_sparse_matrix, float64_t, float64_t, "%f")
SET_SPARSEMATRIX(set_sparse_matrix, floatmax_t, floatmax_t, "%Lf")
#undef SET_SPARSEMATRIX

void CAsciiFile::set_string_list(const SGString<uint8_t>* strings, int32_t num_str)
{
	if (!(file && strings))
		SG_ERROR("File or strings invalid.\n")

	for (int32_t i=0; i<num_str; i++)
	{
		int32_t len = strings[i].slen;
		fwrite(strings[i].string, sizeof(uint8_t), len, file);
		fprintf(file, "\n");
	}
}

void CAsciiFile::set_string_list(const SGString<int8_t>* strings, int32_t num_str)
{
	if (!(file && strings))
		SG_ERROR("File or strings invalid.\n")

	for (int32_t i=0; i<num_str; i++)
	{
		int32_t len = strings[i].slen;
		fwrite(strings[i].string, sizeof(int8_t), len, file);
		fprintf(file, "\n");
	}
}

void CAsciiFile::set_string_list(const SGString<char>* strings, int32_t num_str)
{
	if (!(file && strings))
		SG_ERROR("File or strings invalid.\n")

	for (int32_t i=0; i<num_str; i++)
	{
		int32_t len = strings[i].slen;
		fwrite(strings[i].string, sizeof(char), len, file);
		fprintf(file, "\n");
	}
}

void CAsciiFile::set_string_list(const SGString<int32_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_string_list(const SGString<uint32_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_string_list(const SGString<int16_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_string_list(const SGString<uint16_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_string_list(const SGString<int64_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_string_list(const SGString<uint64_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_string_list(const SGString<float32_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_string_list(const SGString<float64_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_string_list(const SGString<floatmax_t>* strings, int32_t num_str)
{
}

template <class T> void CAsciiFile::append_item(
	DynArray<T>* items, char* ptr_data, char* ptr_item)
{
	size_t len=(ptr_data-ptr_item)/sizeof(char);
	char* item=SG_MALLOC(char, len+1);
	memset(item, 0, sizeof(char)*(len+1));
	item=strncpy(item, ptr_item, len);

	SG_DEBUG("current %c, len %d, item %s\n", *ptr_data, len, item)
	items->append_element(item);
}

void CAsciiFile::tokenize(char delim, substring s, v_array<substring>& ret)
{
	ret.erase();
	char *last = s.start;
	for (; s.start != s.end; s.start++)
	{
		if (*s.start == delim)
		{
			if (s.start != last)
			{
				substring temp = {last,s.start};
				ret.push(temp);
			}
			last = s.start+1;
		}
	}
	if (s.start != last)
	{
		substring final = {last, s.start};
		ret.push(final);
	}
}
