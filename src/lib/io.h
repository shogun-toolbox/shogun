/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __CIO_H__
#define __CIO_H__

#include <time.h>
#include "lib/common.h"

#include <stdio.h>
#include <stdarg.h>

#include "exceptions/ShogunException.h"

#define NUM_LOG_LEVELS 9


class CIO
{
public:
	CIO();

	static void set_target(FILE* target);
	static void set_loglevel(EMessageType level);
	static void message(EMessageType prio, const char *fmt, ... );
	static void progress(DREAL current_val, DREAL min_val=0.0, DREAL max_val=1.0, INT decimals=1, const char* prefix="PROGRESS:\t");
	static void absolute_progress(DREAL current_val, DREAL val, DREAL min_val=0.0, DREAL max_val=1.0, INT decimals=1, const char* prefix="PROGRESS:\t");

	inline static void not_implemented() 
	{
		message(M_ERROR, "Sorry, not yet implemented\n");
	}

	static void buffered_message(EMessageType prio, const CHAR *fmt, ... );
	static CHAR* skip_spaces(CHAR* str);

	static EMessageType loglevel;
	static const EMessageType levels[NUM_LOG_LEVELS];

protected:
	static void check_target();

	//return index into levels array or -1 if message not to be printed
	static int get_prio_string(EMessageType prio);

protected:
	static FILE* target;
	static LONG last_progress_time, progress_start_time ;
	static DREAL last_progress ;

	//const static char* message_strings[NUM_LOG_LEVELS];
	//const static EMessageType levels[NUM_LOG_LEVELS];
	static const char* message_strings[NUM_LOG_LEVELS];
};

#define ASSERT(x) { if (!(x)) CIO::message(M_ERROR, "assertion %s failed in file %s line %d\n",#x, __FILE__, __LINE__);}

#ifdef HAVE_PYTHON
  #define sg_err_fun &throwException
#else
  #define sg_err_fun &cio
#endif

void sg_error(void (*funcPtr)(char*), char *fmt, ... );
void throwException(char *val);
void cio(char *val);

#endif
