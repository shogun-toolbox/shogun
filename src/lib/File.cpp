/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ctype.h>

#include "lib/File.h"
#include "lib/SimpleFile.h"

#include "features/StringFeatures.h"
#include "features/SparseFeatures.h"


CFile::CFile(FILE* f)
: CSGObject()
{
	file=f;
	filename=NULL;
	expected_type=F_UNKNOWN;
}

CFile::CFile(char* fname, char rw, EFeatureType typ, char file_fourcc[4])
: CSGObject()
{
	status=false;
	task=rw;
	expected_type=typ;
	filename=strdup(fname);
	char mode[2];
	mode[0]=rw;
	mode[1]='\0';


	if (rw=='r' || rw == 'w')
	{
		if (filename)
		{
			if ((file=fopen((const char*) filename, (const char*) mode)))
				status=true;
		}
	}
	else
		SG_ERROR("unknown mode '%c'\n", mode[0]);

	if (file_fourcc)
	{
		if (rw=='r')
			status=read_header();
		else if (rw=='w')
			status=write_header();

		if (!status)
			fclose(file);

		file=NULL;
	}
}

CFile::~CFile()
{
	free(filename);
	if (file)
	  fclose(file);
	filename=NULL;
	file=NULL;
}

INT* CFile::load_int_data(INT* target, LONG& num)
{
	ASSERT(expected_type==F_INT);
	CSimpleFile<INT> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_int_data(INT* src, LONG num)
{
	ASSERT(expected_type==F_INT);
	CSimpleFile<INT> f(filename, file);
	status=f.save(src, num);
	return status;
}

DREAL* CFile::load_real_data(DREAL* target, LONG& num)
{
	ASSERT(expected_type==F_DREAL);
	CSimpleFile<DREAL> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

SHORTREAL* CFile::load_shortreal_data(SHORTREAL* target, LONG& num)
{
	ASSERT(expected_type==F_SHORTREAL);
	CSimpleFile<SHORTREAL> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_real_data(DREAL* src, LONG num)
{
	ASSERT(expected_type==F_DREAL);
	CSimpleFile<DREAL> f(filename, file);
	status=f.save(src, num);
	return status;
}

bool CFile::save_shortreal_data(SHORTREAL* src, LONG num)
{
	ASSERT(expected_type==F_SHORTREAL);
	CSimpleFile<SHORTREAL> f(filename, file);
	status=f.save(src, num);
	return status;
}

char* CFile::load_char_data(char* target, LONG& num)
{
	ASSERT(expected_type==F_CHAR);
	CSimpleFile<char> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_char_data(char* src, LONG num)
{
	ASSERT(expected_type==F_CHAR);
	CSimpleFile<char> f(filename, file);
	status=f.save(src, num);
	return status;
}

BYTE* CFile::load_byte_data(BYTE* target, LONG& num)
{
	ASSERT(expected_type==F_BYTE);
	CSimpleFile<BYTE> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_byte_data(BYTE* src, LONG num)
{
	ASSERT(expected_type==F_BYTE);
	CSimpleFile<BYTE> f(filename, file);
	status=f.save(src, num);
	return status;
}

WORD* CFile::load_word_data(WORD* target, LONG& num)
{
	ASSERT(expected_type==F_WORD);
	CSimpleFile<WORD> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_word_data(WORD* src, LONG num)
{
	ASSERT(expected_type==F_WORD);
	CSimpleFile<WORD> f(filename, file);
	status=f.save(src, num);
	return status;
}

SHORT* CFile::load_short_data(SHORT* target, LONG& num)
{
	ASSERT(expected_type==F_SHORT);
	CSimpleFile<SHORT> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_short_data(SHORT* src, LONG num)
{
	ASSERT(expected_type==F_SHORT);
	CSimpleFile<SHORT> f(filename, file);
	status=f.save(src, num);
	return status;
}

INT CFile::parse_first_header(EFeatureType &type)
{
	return -1;
}

INT CFile::parse_next_header(EFeatureType &type)
{
	return -1;
}


bool CFile::read_header()
{
	ASSERT(file);
	UINT intlen=0;
	UINT endian=0;
	UINT file_fourcc=0;
	UINT doublelen=0;

	if ( (fread(&intlen, sizeof(BYTE), 1, file)==1) &&
			(fread(&doublelen, sizeof(BYTE), 1, file)==1) &&
			(fread(&endian, (UINT) intlen, 1, file)== 1) &&
			(fread(&file_fourcc, (UINT) intlen, 1, file)==1))
		return true;
	else
		return false;
}

bool CFile::write_header()
{
    BYTE intlen=sizeof(UINT);
    BYTE doublelen=sizeof(double);
    UINT endian=0x12345678;

	if ((fwrite(&intlen, sizeof(BYTE), 1, file)==1) &&
			(fwrite(&doublelen, sizeof(BYTE), 1, file)==1) &&
			(fwrite(&endian, sizeof(UINT), 1, file)==1) &&
			(fwrite(&fourcc, 4*sizeof(char), 1, file)==1))
		return true;
	else
		return false;
}

template <class T> void CFile::append_item(CDynamicArray<T>* items, char* ptr_data, char* ptr_item)
{
	size_t len=(ptr_data-ptr_item)/sizeof(char);
	char* item=new char[len+1];
	memset(item, 0, sizeof(char)*(len+1));
	item=strncpy(item, ptr_item, len);

	SG_DEBUG("current %c, len %d, item %s\n", *ptr_data, len, item);
	items->append_element(item);
}

bool CFile::read_real_valued_dense(DREAL*& matrix, INT& num_feat, INT& num_vec)
{
	ASSERT(expected_type==F_DREAL);

	struct stat stats;
	if (stat(filename, &stats)!=0)
		SG_ERROR("Could not get file statistics.\n");

	char* data=new char[stats.st_size+1];
	memset(data, 0, sizeof(char)*(stats.st_size+1));
	size_t nread=fread(data, sizeof(char), stats.st_size, file);
	if (nread<=0)
		SG_ERROR("Could not read data from %s.\n");

	SG_DEBUG("data read from file:\n%s\n", data);

	// determine num_feat and num_vec, populate dynamic array
	INT nf=0;
	num_feat=0;
	num_vec=0;
	char* ptr_item=NULL;
	char* ptr_data=data;
	CDynamicArray<char*>* items=new CDynamicArray<char*>();

	while (*ptr_data)
	{
		if (*ptr_data=='\n')
		{
			if (ptr_item)
				nf++;

			if (num_feat!=0 && nf!=num_feat)
				SG_ERROR("Number of features mismatches (%d != %d) in vector %d in file %s.\n", num_feat, nf, num_vec, filename);

			append_item(items, ptr_data, ptr_item);
			num_feat=nf;
			num_vec++;
			nf=0;
			ptr_item=NULL;
		}
		else if (!isblank(*ptr_data) && !ptr_item)
		{
			ptr_item=ptr_data;
		}
		else if (isblank(*ptr_data) && ptr_item)
		{
			append_item(items, ptr_data, ptr_item);
			ptr_item=NULL;
			nf++;
		}

		ptr_data++;
	}

	SG_DEBUG("num feat: %d, num_vec %d\n", num_feat, num_vec);
	delete[] data;

	// now copy data into matrix
	matrix=new DREAL[num_vec*num_feat];
	for (INT i=0; i<num_vec; i++)
	{
		for (INT j=0; j<num_feat; j++)
		{
			char* item=items->get_element(i*num_feat+j);
			matrix[i*num_feat+j]=atof(item);
			delete[] item;
		}
	}
	delete items;

	//CMath::display_matrix(matrix, num_feat, num_vec);
	return true;
}

bool CFile::write_real_valued_dense(const DREAL* matrix, INT num_feat, INT num_vec)
{
	if (!(file && matrix))
		SG_ERROR("File or matrix invalid.\n");

	for (INT i=0; i<num_feat; i++)
	{
		for (INT j=0; j<num_vec; j++)
		{
			DREAL v=matrix[num_feat*j+i];
			if (j==num_vec-1)
				fprintf(file, "%f\n", v);
			else
				fprintf(file, "%f ", v);
		}
	}

	return true;
}

bool CFile::read_real_valued_sparse(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)
{
	size_t blocksize=1024*1024;
	size_t required_blocksize=blocksize;
	BYTE* dummy=new BYTE[blocksize];

	if (file)
	{
		num_vec=0;
		num_feat=0;

		SG_INFO("counting line numbers in file %s\n", filename);
		size_t sz=blocksize;
		size_t block_offs=0;
		size_t old_block_offs=0;
		fseek(file, 0, SEEK_END);
		size_t fsize=ftell(file);
		rewind(file);

		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(BYTE), blocksize, file);
			bool contains_cr=false;
			for (size_t i=0; i<sz; i++)
			{
				block_offs++;
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					num_vec++;
					contains_cr=true;
					required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs+1);
					old_block_offs=block_offs;
				}
			}
			SG_PROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t");
		}

		SG_INFO("found %d feature vectors\n", num_vec);
		delete[] dummy;
		blocksize=required_blocksize;
		dummy = new BYTE[blocksize+1]; //allow setting of '\0' at EOL
		matrix=new TSparse<DREAL>[num_vec];

		rewind(file);
		sz=blocksize;
		INT lines=0;
		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(BYTE), blocksize, file);

			size_t old_sz=0;
			for (size_t i=0; i<sz; i++)
			{
				if (i==sz-1 && dummy[i]!='\n' && sz==blocksize)
				{
					size_t len=i-old_sz+1;
					BYTE* data=&dummy[old_sz];

					for (size_t j=0; j<len; j++)
						dummy[j]=data[j];

					sz=fread(dummy+len, sizeof(BYTE), blocksize-len, file);
					i=0;
					old_sz=0;
					sz+=len;
				}

				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{

					size_t len=i-old_sz;
					BYTE* data=&dummy[old_sz];

					INT dims=0;
					for (size_t j=0; j<len; j++)
					{
						if (data[j]==':')
							dims++;
					}

					if (dims<=0)
					{
						SG_ERROR("Error in line %d - number of"
								" dimensions is %d line is %d characters"
								" long\n line_content:'%.*s'\n", lines,
								dims, len, len, (const char*) data);
					}

					TSparseEntry<DREAL>* feat=new TSparseEntry<DREAL>[dims];

					//skip label part
					size_t j=0;
					for (; j<len; j++)
					{
						if (data[j]==':')
						{
							j=-1; //file without label
							break;
						}

						if (data[j]==' ')
						{
							data[j]='\0';

							//skip label part
							break;
						}
					}

					INT d=0;
					j++;
					BYTE* start=&data[j];
					for (; j<len; j++)
					{
						if (data[j]==':')
						{
							data[j]='\0';

							feat[d].feat_index=(INT) atoi((const char*) start)-1;
							num_feat=CMath::max(num_feat, feat[d].feat_index+1);

							j++;
							start=&data[j];
							for (; j<len; j++)
							{
								if (data[j]==' ' || data[j]=='\n')
								{
									data[j]='\0';
									feat[d].entry=(DREAL) atof((const char*) start);
									d++;
									break;
								}
							}

							if (j==len)
							{
								data[j]='\0';
								feat[dims-1].entry=(DREAL) atof((const char*) start);
							}

							j++;
							start=&data[j];
						}
					}

					matrix[lines].vec_index=lines;
					matrix[lines].num_feat_entries=dims;
					matrix[lines].features=feat;

					old_sz=i+1;
					lines++;
					SG_PROGRESS(lines, 0, num_vec, 1, "LOADING:\t");
				}
			}
		}

		SG_INFO("file successfully read\n");
	}

	delete[] dummy;
	return true;
}

bool CFile::write_real_valued_sparse(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec)
{
	if (!(file && matrix))
		SG_ERROR("File or matrix invalid.\n");

	for (INT i=0; i<num_vec; i++)
	{
		TSparseEntry<DREAL>* vec = matrix[i].features;
		INT len=matrix[i].num_feat_entries;

		for (INT j=0; j<len; j++)
		{
			if (j<len-1)
				fprintf(file, "%d:%f ", (INT) vec[j].feat_index+1, (double) vec[j].entry);
			else
				fprintf(file, "%d:%f\n", (INT) vec[j].feat_index+1, (double) vec[j].entry);
		}
	}

	return true;
}


bool CFile::read_char_valued_strings(T_STRING<char>*& strings, INT& num_str, INT& max_string_len)
{
	bool result=false;

	size_t blocksize=1024*1024;
	size_t required_blocksize=0;
	char* dummy=new char[blocksize];
	char* overflow=NULL;
	INT overflow_len=0;

	if (file)
	{
		num_str=0;
		max_string_len=0;

		SG_INFO("counting line numbers in file %s\n", filename);
		size_t sz=blocksize;
		size_t block_offs=0;
		size_t old_block_offs=0;
		fseek(file, 0, SEEK_END);
		size_t fsize=ftell(file);
		rewind(file);

		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(char), blocksize, file);
			bool contains_cr=false;
			for (size_t i=0; i<sz; i++)
			{
				block_offs++;
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					num_str++;
					contains_cr=true;
					required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs);
					old_block_offs=block_offs;
				}
			}
			SG_PROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t");
		}

		SG_INFO("found %d strings\n", num_str);
		SG_DEBUG("block_size=%d\n", required_blocksize);
		delete[] dummy;
		blocksize=required_blocksize;
		dummy=new char[blocksize];
		overflow=new char[blocksize];
		strings=new T_STRING<char>[num_str];

		rewind(file);
		sz=blocksize;
		INT lines=0;
		size_t old_sz=0;
		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(char), blocksize, file);

			old_sz=0;
			for (size_t i=0; i<sz; i++)
			{
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					INT len=i-old_sz;
					max_string_len=CMath::max(max_string_len, len+overflow_len);

					strings[lines].length=len+overflow_len;
					strings[lines].string=new char[len+overflow_len];

					for (INT j=0; j<overflow_len; j++)
						strings[lines].string[j]=overflow[j];
					for (INT j=0; j<len; j++)
						strings[lines].string[j+overflow_len]=dummy[old_sz+j];

					// clear overflow
					overflow_len=0;

					//CMath::display_vector(strings[lines].string, len);
					old_sz=i+1;
					lines++;
					SG_PROGRESS(lines, 0, num_str, 1, "LOADING:\t");
				}
			}

			for (size_t i=old_sz; i<sz; i++)
				overflow[i-old_sz]=dummy[i];

			overflow_len=sz-old_sz;
		}
		result=true;
		SG_INFO("file successfully read\n");
		SG_INFO("max_string_length=%d\n", max_string_len);
		SG_INFO("num_strings=%d\n", num_str);
	}

	delete[] dummy;
	delete[] overflow;

	return result;
}

bool CFile::write_char_valued_strings(const T_STRING<char>* strings, INT num_str)
{
	if (!(file && strings))
		SG_ERROR("File or strings invalid.\n");

	for (INT i=0; i<num_str; i++)
	{
		INT len = strings[i].length;
		fwrite(strings[i].string, sizeof(char), len, file);
		fprintf(file, "\n");
	}

	return true;
}


