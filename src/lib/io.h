#ifndef __CIO_H__
#define __CIO_H__

#include <stdio.h>
#include <stdarg.h>
class CIO
{
    public:
	CIO();

	static void set_target(FILE* target);
	static void message(const char *fmt, ... );
	static void message(FILE* target, const char *fmt, ... );

	static void buffered_message(const char *fmt, ... );
	static void buffered_message(FILE* target, const char *fmt, ... );
    protected:
	static FILE* target;
};
#endif
