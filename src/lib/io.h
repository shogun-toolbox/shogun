#ifndef __CIO_H__
#define __CIO_H__

#include <time.h>
#include "lib/common.h"

#include <stdio.h>
#include <stdarg.h>

#define ASSERT(x) { if (!(x)) CIO::message(M_ERROR, "assertion %s failed in file %s line %d",#x, __FILE__, __LINE__);}

#define NUM_LOG_LEVELS 9

class CIO
{
public:
	CIO();

	static void set_target(FILE* target);
	static void set_loglevel(EMessageType level);
	static void message(EMessageType prio, const CHAR *fmt, ... );
	static void progress(DREAL current_val, DREAL min_val=0.0, DREAL max_val=1.0, INT decimals=1, const char* prefix="PROGRESS:\t");
	static void absolute_progress(DREAL current_val, DREAL val, DREAL min_val=0.0, DREAL max_val=1.0, INT decimals=1, const char* prefix="PROGRESS:\t");

	inline static void not_implemented() 
	{
		message(M_ERROR, "Sorry, not yet implemented\n");
	}

	static void buffered_message(EMessageType prio, const CHAR *fmt, ... );
	static CHAR* skip_spaces(CHAR* str);

protected:
	static void check_target();

	//return index into levels array or -1 if message not to be printed
	static int get_prio_string(EMessageType prio);

protected:
	static FILE* target;
	static LONG last_progress_time, progress_start_time ;
	static DREAL last_progress ;

	static EMessageType loglevel;
	const static char* message_strings[NUM_LOG_LEVELS];
	const static EMessageType levels[NUM_LOG_LEVELS];
};
#endif
