#include "lib/io.h"

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>

FILE* CIO::target=stdout;

CIO::CIO()
{
}

void CIO::message(const char *fmt, ... )
{
	check_target();
    va_list list;
    va_start(list,fmt);
    vfprintf(target,fmt,list);
    va_end(list);
    fflush(target);
}

void CIO::message(FILE* target, const char *fmt, ... )
{
	check_target();
    va_list list;
    va_start(list,fmt);
    vfprintf(target,fmt,list);
    va_end(list);
    fflush(target);
}

void CIO::buffered_message(const char *fmt, ... )
{
	check_target();
    va_list list;
    va_start(list,fmt);
    vfprintf(target,fmt,list);
    va_end(list);
}

void CIO::buffered_message(FILE* target, const char *fmt, ... )
{
	check_target();
    va_list list;
    va_start(list,fmt);
    vfprintf(target,fmt,list);
    va_end(list);
}

char* CIO::skip_spaces(char* str)
{
	int i=0;

	if (str)
	{
		for (i=0; isspace(str[i]); i++);

		return &str[i];
	}
	else 
		return str;
}

void CIO::set_target(FILE* t)
{
	target=t;
}

void CIO::check_target()
{
	if (!target)
		target=stdout;
}
