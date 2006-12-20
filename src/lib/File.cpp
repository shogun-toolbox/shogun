/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <string.h>

#include "lib/File.h"
#include "lib/SimpleFile.h"

CFile::CFile(CHAR* fname, CHAR rw, EFeatureType typ, CHAR file_fourcc[4])
{
	status=false;
	task=rw;
	expected_type=typ;
	filename=strdup(fname);
	CHAR mode[2];
	mode[0]=rw;
	mode[1]='\0';


	if (rw=='r' || rw == 'w')
	{
		if (filename)
		{
			if ((file=fopen((const CHAR*) filename, (const CHAR*) mode)))
				status=true;
		}
	}

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

bool CFile::save_real_data(DREAL* src, LONG num)
{
	ASSERT(expected_type==F_DREAL);
	CSimpleFile<DREAL> f(filename, file);
	status=f.save(src, num);
	return status;
}

CHAR* CFile::load_char_data(CHAR* target, LONG& num)
{
	ASSERT(expected_type==F_CHAR);
	CSimpleFile<CHAR> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_char_data(CHAR* src, LONG num)
{
	ASSERT(expected_type==F_CHAR);
	CSimpleFile<CHAR> f(filename, file);
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
    ASSERT(file!=NULL);
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
