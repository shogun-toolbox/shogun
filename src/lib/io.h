/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __CIO_H__
#define __CIO_H__

#include <time.h>
#include "lib/common.h"
#include "lib/ShogunException.h"

#include <stdio.h>
#include <stdarg.h>


#define ASSERT(x) { if (!(x)) CIO::message(M_ERROR, "assertion %s failed in file %s line %d\n",#x, __FILE__, __LINE__);}

// printf like funktions (with additional severity level)
// for object derived from CSGObject
#define SG_DEBUG(x...) io.message(M_DEBUG,x)
#define SG_INFO(x...) io.message(M_INFO,x)
#define SG_WARNING(x...) io.message(M_WARN,x)
#define SG_ERROR(x...) io.message(M_ERROR,x)
#define SG_PRINT(x...) io.message(M_MESSAGEONLY,x)
#define SG_PRINT(x...) io.message(M_MESSAGEONLY,x)

#define SG_PROGRESS(x...) io.progress(x)
#define SG_ABS_PROGRESS(x...) io.absolute_progress(x)

// printf like funktions (with additional severity level)
// static versions
#define SG_SDEBUG(x...) CIO::message(M_DEBUG,x)
#define SG_SINFO(x...) CIO::message(M_INFO,x)
#define SG_SWARNING(x...) CIO::message(M_WARN,x)
#define SG_SERROR(x...) CIO::message(M_ERROR,x)
#define SG_SPRINT(x...) CIO::message(M_MESSAGEONLY,x)

#define SG_SPROGRESS(x...) CIO::progress(x)
#define SG_SABS_PROGRESS(x...) CIO::absolute_progress(x)

// printf like funktions (with additional severity level)
// when global gui object is available
#define SG_GDEBUG(x...) gui->io.message(M_DEBUG,x)
#define SG_GINFO(x...) gui->io.message(M_INFO,x)
#define SG_GWARNING(x...) gui->io.message(M_WARN,x)
#define SG_GERROR(x...) gui->io.message(M_ERROR,x)
#define SG_GPRINT(x...) gui->io.message(M_MESSAGEONLY,x)

#define SG_SPROGRESS(x...) CIO::progress(x)
#define SG_SABS_PROGRESS(x...) CIO::absolute_progress(x)

#define NUM_LOG_LEVELS 9
#define FBUFSIZE 4096

extern CHAR file_buffer[FBUFSIZE];
extern CHAR directory_name[FBUFSIZE];

class CIO
{
public:
	CIO();

	static void set_target(FILE* target);
	static void set_loglevel(EMessageType level);
	static EMessageType get_loglevel();
	static void message(EMessageType prio, const char *fmt, ... );
	static void progress(DREAL current_val, DREAL min_val=0.0, DREAL max_val=1.0, INT decimals=1, const char* prefix="PROGRESS:\t");
	static void absolute_progress(DREAL current_val, DREAL val, DREAL min_val=0.0, DREAL max_val=1.0, INT decimals=1, const char* prefix="PROGRESS:\t");

	inline static void not_implemented() 
	{
		message(M_ERROR, "Sorry, not yet implemented\n");
	}

	static void buffered_message(EMessageType prio, const CHAR *fmt, ... );
	static CHAR* skip_spaces(CHAR* str);

	///set directory-name
	static inline void set_dirname(const CHAR* dirname)
	{
		strncpy(directory_name, dirname, FBUFSIZE);
	}

	///concatenate directory and filename
	/// ( non thread safe )
	static CHAR* concat_filename(const CHAR* filename);

	///concatenate directory and filename
	/// ( non thread safe )
	static INT filter(const struct dirent* d);

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

	static EMessageType loglevel;
	static const EMessageType levels[NUM_LOG_LEVELS];
};

#endif // __CIO_H__
