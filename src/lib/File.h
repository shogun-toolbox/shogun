#ifndef __FILE_H__
#define __FILE_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "lib/common.h"

class CFile
{
public:
	CFile(char* fname, char rw, EType type, bool autodetection=false);
	~CFile();

	INT* load_int_data(int num, INT* target);
	REAL* load_real_data(int num, REAL* target);
	CHAR* load_char_data(int num, CHAR* target);
	BYTE* load_byte_data(int num, BYTE* target);
	WORD* load_word_data(int num, WORD* target);
	SHORT* load_short_data(int num, SHORT* target);

	bool save_int_data(int num, INT* src);
	bool save_real_data(int num, REAL* src);
	bool save_char_data(int num, CHAR* src);
	bool save_byte_data(int num, BYTE* src);
	bool save_word_data(int num, WORD* src);
	bool save_short_data(int num, SHORT* src);

	inline bool is_ok()
	{
		return status;
	}

protected: 
	bool read_header();
	bool write_header();

protected: 
	FILE* file;
	bool status;
	char task;
	char* fname;
	EType filetype;
};
#endif
