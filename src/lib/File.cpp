#include <string.h>

#include "lib/File.h"
#include "lib/SimpleFile.h"

CFile::CFile(CHAR* fname, CHAR rw, EFeatureType typ, CHAR fourcc[4])
{
	status=false;
	task=rw;
	expected_type=typ;
	this->fname=strdup(fname);
	CHAR mode[2];
	mode[0]=rw;
	mode[1]='\0';


	if (rw=='r' || rw == 'w')
	{
		if ( fname)
		{
			if ((file=fopen((const CHAR*) fname, (const CHAR*) mode)))
				status=true;
		}
	}

	if (fourcc)
	{
		if (rw=='r')
			status=read_header();
		else if (rw=='w')
			status=read_header();

		if (!status)
			fclose(file);

		file=NULL;
	}
}

CFile::~CFile()
{
	free(fname);
	if (file)
	  fclose(file);
	fname=NULL;
	file=NULL;
}

INT* CFile::load_int_data(INT* target, long& num)
{
	ASSERT(expected_type==F_INT);
	CSimpleFile<INT> f(fname, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_int_data(INT* src, LONG num)
{
	ASSERT(expected_type==F_INT);
	CSimpleFile<INT> f(fname, file);
	status=f.save(src, num);
	return status;
}

REAL* CFile::load_real_data(REAL* target, long& num)
{
	ASSERT(expected_type==F_REAL);
	CSimpleFile<REAL> f(fname, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_real_data(REAL* src, LONG num)
{
	ASSERT(expected_type==F_REAL);
	CSimpleFile<REAL> f(fname, file);
	status=f.save(src, num);
	return status;
}

CHAR* CFile::load_char_data(CHAR* target, long& num)
{
	ASSERT(expected_type==F_CHAR);
	CSimpleFile<CHAR> f(fname, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_char_data(CHAR* src, LONG num)
{
	ASSERT(expected_type==F_CHAR);
	CSimpleFile<CHAR> f(fname, file);
	status=f.save(src, num);
	return status;
}

BYTE* CFile::load_byte_data(BYTE* target, long& num)
{
	ASSERT(expected_type==F_BYTE);
	CSimpleFile<BYTE> f(fname, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_byte_data(BYTE* src, LONG num)
{
	ASSERT(expected_type==F_BYTE);
	CSimpleFile<BYTE> f(fname, file);
	status=f.save(src, num);
	return status;
}

WORD* CFile::load_word_data(WORD* target, long& num)
{
	ASSERT(expected_type==F_WORD);
	CSimpleFile<WORD> f(fname, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_word_data(WORD* src, LONG num)
{
	ASSERT(expected_type==F_WORD);
	CSimpleFile<WORD> f(fname, file);
	status=f.save(src, num);
	return status;
}

SHORT* CFile::load_short_data(SHORT* target, long& num)
{
	ASSERT(expected_type==F_SHORT);
	CSimpleFile<SHORT> f(fname, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_short_data(SHORT* src, LONG num)
{
	ASSERT(expected_type==F_SHORT);
	CSimpleFile<SHORT> f(fname, file);
	status=f.save(src, num);
	return status;
}

INT parse_first_header(EFeatureType &type)
{
	return -1;
}

INT parse_next_header(EFeatureType &type)
{
	return -1;
}


bool CFile::read_header()
{
    ASSERT(file!=NULL);
    UINT intlen=0;
    UINT endian=0;
    UINT fourcc=0;
    UINT doublelen=0;

	if ( (fread(&intlen, sizeof(BYTE), 1, file)==1) &&
			(fread(&doublelen, sizeof(BYTE), 1, file)==1) &&
			(fread(&endian, (UINT) intlen, 1, file)== 1) &&
			(fread(&fourcc, (UINT) intlen, 1, file)==1))
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
