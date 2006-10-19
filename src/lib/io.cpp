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

#include "lib/config.h"

#ifdef HAVE_OCTAVE
#include <octave/config.h>
#include <octave/error.h>
#include <octave/lo-error.h>
#endif

#ifdef HAVE_MATLAB
#include <mex.h>
#endif

#ifdef HAVE_PYTHON
#include <Python.h>
#endif

#ifdef HAVE_R
#include <R.h>
#include <Rinternals.h>
#include <R_ext/RS.h>
#endif

#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/common.h"
#include "lib/Time.h"
#include "lib/Mathematics.h"

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>

FILE* CIO::target=stdout;
LONG CIO::last_progress_time=0 ;
LONG CIO::progress_start_time=0 ;
DREAL CIO::last_progress=1 ;
EMessageType CIO::loglevel = M_WARN;
const EMessageType CIO::levels[NUM_LOG_LEVELS]={M_DEBUG, M_INFO, M_NOTICE, M_WARN, M_ERROR, M_CRITICAL, M_ALERT, M_EMERGENCY,M_MESSAGEONLY};
const char* CIO::message_strings[NUM_LOG_LEVELS]={"[DEBUG] ", "[INFO] ", "[NOTICE] ", "\033[1;34m[WARN]\033[0m ", "\033[1;31m[ERROR]\033[0m ", "[CRITICAL] ", "[ALERT] ", "[EMERGENCY] ", ""};

CIO::CIO()
{
}

void CIO::message(EMessageType prio, const CHAR *fmt, ... )
{
#if defined(HAVE_MATLAB) || defined(HAVE_PYTHON) || defined(HAVE_OCTAVE) || defined(HAVE_R)
	char str[4096];
    va_list list;
    va_start(list,fmt);
	vsnprintf(str, sizeof(str), fmt, list);
    va_end(list);

	check_target();
	int p=get_prio_string(prio);
	if (p>=0)
	{
#ifdef HAVE_MATLAB
		switch (prio)
		{
			case M_DEBUG:
			case M_INFO:
			case M_NOTICE:
			case M_MESSAGEONLY:
				fprintf(target, message_strings[p]);
				fprintf(target, "%s", str);
				break;

			case M_WARN:
				mexWarnMsgTxt(str);
				break;

			case M_ERROR:
			case M_CRITICAL:
			case M_ALERT:
			case M_EMERGENCY:
				CSignal::unset_handler();
				mexErrMsgTxt(str);
				break;
			default:
				break;
		}
#elif defined(HAVE_OCTAVE)
		switch (prio)
		{
			case M_DEBUG:
			case M_INFO:
			case M_NOTICE:
			case M_MESSAGEONLY:
				fprintf(target, message_strings[p]);
				fprintf(target, "%s", str);
				break;

			case M_WARN:
				::warning(str);
				break;

			case M_ERROR:
			case M_CRITICAL:
			case M_ALERT:
			case M_EMERGENCY:
				CSignal::unset_handler();
				error("%s", str);
				break;
			default:
				break;
		}
#elif defined(HAVE_PYTHON)
		switch (prio)
		{
			case M_DEBUG:
			case M_INFO:
			case M_NOTICE:
			case M_MESSAGEONLY:
				fprintf(target, message_strings[p]);
				fprintf(target, "%s", str);
				break;

			case M_WARN:
				PyErr_Warn(NULL, str);
				break;

			case M_ERROR:
			case M_CRITICAL:
			case M_ALERT:
			case M_EMERGENCY:
				PyErr_SetString(PyExc_RuntimeError,str);
				fprintf(target, message_strings[p]);
				fprintf(target, "%s", str);
				break;
			default:
				break;
		}
#elif defined(HAVE_R)
		switch (prio)
		{
			case M_DEBUG:
			case M_INFO:
			case M_NOTICE:
			case M_MESSAGEONLY:
			case M_WARN:
				{
					int p=get_prio_string(prio);

					if (p>=0)
					{
						Rprintf("%s",message_strings[p]);
						va_list list;
						va_start(list,fmt);
						Rvprintf(fmt,list);
						va_end(list);
					}
				}
				break;
			case M_ERROR:
			case M_CRITICAL:
			case M_ALERT:
			case M_EMERGENCY:
				error("%s", str);
				break;
			default:
				break;
		}

#endif
		fflush(target);
	}
#else
	check_target();
	int p=get_prio_string(prio);
	if (p>=0)
	{
		fprintf(target, message_strings[p]);
		va_list list;
		va_start(list,fmt);
		vfprintf(target,fmt,list);
		va_end(list);
		fflush(target);
	}
#endif
}

void CIO::buffered_message(EMessageType prio, const CHAR *fmt, ... )
{
	check_target();
	int p=get_prio_string(prio);
	if (p>=0)
	{
		fprintf(target, message_strings[p]);
		va_list list;
		va_start(list,fmt);
		vfprintf(target,fmt,list);
		va_end(list);
	}
}

void CIO::progress(DREAL current_val, DREAL min_val, DREAL max_val, INT decimals, const char* prefix)
{
	LONG runtime = CTime::get_runtime() ;
	
	char str[1000];
	DREAL v=-1, estimate=0, total_estimate=0 ;
	check_target();
	
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
	LONG runtime = CTime::get_runtime() ;
	
	char str[1000];
	DREAL v=-1, estimate=0, total_estimate=0 ;
	check_target();

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

CHAR* CIO::skip_spaces(CHAR* str)
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

void CIO::set_loglevel(EMessageType level)
{
	loglevel=level;
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

int CIO::get_prio_string(EMessageType prio)
{
	int i=0;
	int idx=-1;

	while (i<NUM_LOG_LEVELS)
	{
		if (levels[i]==loglevel)
		{
			while (i<NUM_LOG_LEVELS)
			{
				if (levels[i]==prio)
				{
					idx=i;
					break;
				}
				i++;
			}
			break;
		}
		i++;
	}

	return idx;
}
