#ifndef __FILE_H__
#define __FILE_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "lib/common.h"

/// A file consists of a header
/// then an alternation of a type header and data
///
/// or just raw data (simplefile=true)
///
/// the more complex stuff is currently not implemented
///
class CFile
{
public:
	///Open a file of name fname with mode rw (r or w)
	/// - type specifies the datatype used in the file (F_INT,...)
	/// - fourcc : in the case fourcc is 0, type will be ignored
	/// and the file is treated as if it has a header/[typeheader,data]+
	/// else a the files header will be checked to contain the specified
	/// fourcc (e.g. 'RFEA')
	CFile(char* fname, char rw, EType type, char fourcc[4]=NULL);
	~CFile();

	int parse_first_header(EType &type);
	int parse_next_header(EType &type);

	// set target to NULL to get it automagically allocated
	// set num to 0 if whole file is to be read
	INT*   load_int_data(INT* target, int& num);
	REAL*  load_real_data(REAL* target, int& num);
	CHAR*  load_char_data(CHAR* target, int& num);
	BYTE*  load_byte_data(BYTE* target, int& num);
	WORD*  load_word_data(WORD* target, int& num);
	SHORT* load_short_data(SHORT* target, int& num);

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
	EType expected_type;
	int num_header;
	char fourcc[4];
};
#endif
