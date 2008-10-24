/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"
#include "lib/matlab.h"
#include "lib/octave.h"
#include "lib/r.h"
#include "lib/python.h"

#include "lib/io.h"
#include "lib/ShogunException.h"
#include "lib/Signal.h"
#include "lib/common.h"
#include "lib/Time.h"
#include "lib/Mathematics.h"

#include <sys/types.h>
#include <sys/stat.h>

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include <dirent.h>
#include <stdlib.h>
#include <unistd.h>

const EMessageType CIO::levels[NUM_LOG_LEVELS]={M_DEBUG, M_INFO, M_NOTICE, M_WARN, M_ERROR, M_CRITICAL, M_ALERT, M_EMERGENCY, M_MESSAGEONLY};
const char* CIO::message_strings[NUM_LOG_LEVELS]={"[DEBUG] ", "[INFO] ", "[NOTICE] ", "\033[1;34m[WARN]\033[0m ", "\033[1;31m[ERROR]\033[0m ", "[CRITICAL] ", "[ALERT] ", "[EMERGENCY] ", ""};

/// file name buffer
char file_buffer[FBUFSIZE];

/// directory name buffer
char directory_name[FBUFSIZE];

CIO::CIO() : target(stdout), last_progress_time(0), progress_start_time(0),
	last_progress(1), show_progress(false), loglevel(M_WARN)
{
}

CIO::CIO(const CIO& orig) : target(orig.get_target()), last_progress_time(0),
	progress_start_time(0), last_progress(1), show_progress(orig.get_show_progress()),
	loglevel(orig.get_loglevel())
{
}

void CIO::message(EMessageType prio, const char *fmt, ... ) const
{
	const char* msg_intro=get_msg_intro(prio);
	if (!msg_intro)
		return;

	char str[4096];
	va_list list;
	va_start(list,fmt);
	vsnprintf(str, sizeof(str), fmt, list);
	va_end(list);

	switch (prio)
	{
		case M_DEBUG:
		case M_INFO:
		case M_NOTICE:
		case M_MESSAGEONLY:
#if defined(WIN32) && defined(HAVE_MATLAB)
			fprintf(target, "%s", msg_intro);
			mexPrintf("%s", str);
#elif defined(HAVE_R)
			if (target==stdout)
			{
				Rprintf((char*) "%s", msg_intro);
				Rprintf((char*) "%s", str);
			}
			else
			{
				fprintf(target, "%s", msg_intro);
				fprintf(target, "%s", str);
			}
#else
			fprintf(target, "%s", msg_intro);
			fprintf(target, "%s", str);
#endif
			break;

		case M_WARN:
#if defined(HAVE_MATLAB)
			mexWarnMsgTxt(str);
#elif defined(HAVE_OCTAVE)
			::warning(str);
#elif defined(HAVE_PYTHON) // no check for swig necessary
			PyErr_Warn(NULL, str);
#elif defined(HAVE_R)
			if (target==stdout)
			{
				Rprintf((char*) "%s", msg_intro);
				Rprintf((char*) "%s", str);
			}
			else
			{
				fprintf(target, "%s", msg_intro);
				fprintf(target, "%s", str);
			}
#else
			fprintf(target, "%s", msg_intro);
			fprintf(target, "%s", str);
#endif
			break;

		case M_ERROR:
		case M_CRITICAL:
		case M_ALERT:
		case M_EMERGENCY:
#if defined(WIN32) && defined(HAVE_MATLAB)
			mexPrintf("%s", str);
#elif defined(HAVE_PYTHON)
			// nop - str will be printed when exception is displayed in python
#elif defined(HAVE_R)
			if (target==stdout)
			{
				Rprintf((char*) "%s", msg_intro);
				Rprintf((char*) "%s", str);
			}
			else
			{
				fprintf(target, "%s", msg_intro);
				fprintf(target, "%s", str);
			}
#else
			fprintf(target, "%s", str);
#endif
			throw ShogunException(str);
			break;
		default:
			break;
	}

	fflush(target);
}

void CIO::buffered_message(EMessageType prio, const char *fmt, ... ) const
{
	const char* msg_intro=get_msg_intro(prio);
	if (!msg_intro)
		return;

	fprintf(target, "%s", msg_intro);
	va_list list;
	va_start(list,fmt);
	vfprintf(target,fmt,list);
	va_end(list);
}

void CIO::progress(DREAL current_val, DREAL min_val, DREAL max_val, INT decimals, const char* prefix)
{
	if (!show_progress)
		return;

	LONG runtime = CTime::get_runtime() ;

	char str[1000];
	DREAL v=-1, estimate=0, total_estimate=0 ;

	if (max_val-min_val>0.0)
		v=100*(current_val-min_val+1)/(max_val-min_val+1);

	if (decimals < 1)
		decimals = 1;

	if (last_progress>v)
	{
		last_progress_time = runtime ;
		progress_start_time = runtime;
		last_progress = v ;
	}
	else
	{
		if (v>100) v=100.0 ;
		if (v<=0) v=1e-5 ;
		last_progress = v-1e-6 ; ;

		if ((v!=100.0) && (runtime - last_progress_time<10))
			return ;

		last_progress_time = runtime ;
		estimate = (1-v/100)*(last_progress_time-progress_start_time)/(v/100) ;
		total_estimate = (last_progress_time-progress_start_time)/(v/100) ;
	}

	if (estimate/100>120)
	{
		snprintf(str, sizeof(str), "%%s %%%d.%df%%%%    %%1.1f minutes remaining    %%1.1f minutes total    \r",decimals+3, decimals);
		message(M_MESSAGEONLY, str, prefix, v, (float)estimate/100/60, (float)total_estimate/100/60);
	}
	else
	{
		snprintf(str, sizeof(str), "%%s %%%d.%df%%%%    %%1.1f seconds remaining    %%1.1f seconds total    \r",decimals+3, decimals);
		message(M_MESSAGEONLY, str, prefix, v, (float)estimate/100, (float)total_estimate/100);
	}

    fflush(target);
}

void CIO::absolute_progress(DREAL current_val, DREAL val, DREAL min_val, DREAL max_val, INT decimals, const char* prefix)
{
	if (!show_progress)
		return;

	LONG runtime = CTime::get_runtime() ;

	char str[1000];
	DREAL v=-1, estimate=0, total_estimate=0 ;

	if (max_val-min_val>0)
		v=100*(val-min_val+1)/(max_val-min_val+1);

	if (decimals < 1)
		decimals = 1;

	if (last_progress>v)
	{
		last_progress_time = runtime ;
		progress_start_time = runtime;
		last_progress = v ;
	}
	else
	{
		if (v>100) v=100.0 ;
		if (v<=0) v=1e-6 ;
		last_progress = v-1e-5 ; ;

		if ((v!=100.0) && (runtime - last_progress_time<100))
			return ;

		last_progress_time = runtime ;
		estimate = (1-v/100)*(last_progress_time-progress_start_time)/(v/100) ;
		total_estimate = (last_progress_time-progress_start_time)/(v/100) ;
	}

	if (estimate/100>120)
	{
		snprintf(str, sizeof(str), "%%s %%%d.%df    %%1.1f minutes remaining    %%1.1f minutes total    \r",decimals+3, decimals);
		message(M_MESSAGEONLY, str, prefix, current_val, (float)estimate/100/60, (float)total_estimate/100/60);
	}
	else
	{
		snprintf(str, sizeof(str), "%%s %%%d.%df    %%1.1f seconds remaining    %%1.1f seconds total    \r",decimals+3, decimals);
		message(M_MESSAGEONLY, str, prefix, current_val, (float)estimate/100, (float)total_estimate/100);
	}

    fflush(target);
}

void CIO::done()
{
	if (!show_progress)
		return;

	message(M_INFO, "done.\n");
}

char* CIO::skip_spaces(char* str)
{
	INT i=0;

	if (str)
	{
		for (i=0; isspace(str[i]); i++);

		return &str[i];
	}
	else
		return str;
}

char* CIO::skip_blanks(char* str)
{
	INT i=0;

	if (str)
	{
		for (i=0; isblank(str[i]); i++);

		return &str[i];
	}
	else
		return str;
}

EMessageType CIO::get_loglevel() const
{
	return loglevel;
}

void CIO::set_loglevel(EMessageType level)
{
	loglevel=level;
}

bool CIO::get_show_progress() const
{
	return show_progress;
}

void CIO::set_target(FILE* t)
{
	target=t;
}

const char* CIO::get_msg_intro(EMessageType prio) const
{
	for (INT i=NUM_LOG_LEVELS-1; i>=0; i--)
	{
		// ignore msg if prio's level is under loglevel,
		// but not if prio's level higher than M_WARN
		if (levels[i]<loglevel && prio<=M_WARN)
			return NULL;

		if (levels[i]==prio)
			return message_strings[i];
	}

	return NULL;
}

char* CIO::concat_filename(const char* filename)
{
	if (snprintf(file_buffer, FBUFSIZE, "%s/%s", directory_name, filename) > FBUFSIZE)
		SG_SERROR("filename too long");
	return file_buffer;
}

int CIO::filter(CONST_DIRENT_T* d)
{
	if (d)
	{
		char* fname=concat_filename(d->d_name);

		if (!access(fname, R_OK))
		{
			struct stat s;
			if (!stat(fname, &s) && S_ISREG(s.st_mode))
				return 1;
		}
	}

	return 0;
}
