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
	static void message(const CHAR *fmt, ... );
	static void message(FILE* target, const CHAR *fmt, ... );
	static void not_implemented() 
	  {
	    message(stderr, "Sorry, not yet implemented\n");
	  };

	static void buffered_message(const CHAR *fmt, ... );
	static void buffered_message(FILE* target, const CHAR *fmt, ... );

	static CHAR* skip_spaces(CHAR* str);
protected:
	void static check_target();

protected:
	static FILE* target;
};
#endif
