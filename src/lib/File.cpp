#include <string.h>
#include <assert.h>

#include "lib/File.h"
#include "lib/SimpleFile.h"

CFile::CFile(char* fname, char rw, EType typ, bool autodetection)
{
	status=false;
	task=rw;
	filetype=typ;
	this->fname=strdup(fname);
	char mode[2];
	mode[0]=rw;
	mode[1]='\0';


	if (rw=='r' || rw == 'w')
	{
		if ( fname)
		{
			if ((file=fopen((const char*) fname, (const char*) mode)))
			{
				status=true;
			}
		}
	}

	if (status && rw=='r')
		status=read_header();
}

REAL* CFile::load_real_data(int num, REAL* target)
{
	assert(filetype==F_REAL);
	CSimpleFile<REAL> f(fname, file);
	target=f.load(num, target);
	return target;
}

SHORT* CFile::load_short_data(int num, SHORT* target)
{
	return target;
}

CHAR* CFile::load_char_data(int num, CHAR* target)
{
	return target;
}

bool CFile::read_header()
{
	return false;
}

bool CFile::write_header()
{
	return false;
}
