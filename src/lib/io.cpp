#include "lib/io.h"

#include <stdio.h>
#include <stdarg.h>

FILE* CIO::target=stdout;

CIO::CIO()
{
}

void CIO::message(const char *fmt, ... )
{
    va_list list;
    va_start(list,fmt);
    vfprintf(target,fmt,list);
    va_end(list);
    fflush(target);
}

void CIO::message(FILE* target, const char *fmt, ... )
{
    va_list list;
    va_start(list,fmt);
    vfprintf(target,fmt,list);
    va_end(list);
    fflush(target);
}

void CIO::buffered_message(const char *fmt, ... )
{
    va_list list;
    va_start(list,fmt);
    vfprintf(target,fmt,list);
    va_end(list);
}

void CIO::buffered_message(FILE* target, const char *fmt, ... )
{
    va_list list;
    va_start(list,fmt);
    vfprintf(target,fmt,list);
    va_end(list);
}
