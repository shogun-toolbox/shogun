#ifndef __CIO_H__
#define __CIO_H__

#include "lib/common.h"

#include <stdio.h>
#include <stdarg.h>

class CIO
{
public:
	CIO();

	static void set_target(FILE* target);
	static void message(EMessageType prio, const CHAR *fmt, ... );

	inline static void not_implemented() 
	{
		message(M_ERROR, "Sorry, not yet implemented\n");
	}

	static void buffered_message(EMessageType prio, const CHAR *fmt, ... );

	static CHAR* skip_spaces(CHAR* str);

protected:
	static void check_target();
	static void print_message_prio(EMessageType prio, FILE* target);

protected:
	static FILE* target;
};
#endif
