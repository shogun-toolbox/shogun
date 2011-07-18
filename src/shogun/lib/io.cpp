/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2009 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#include "lib/io.h"
#include "lib/ShogunException.h"
#include "lib/Signal.h"
#include "lib/common.h"
#include "lib/Time.h"
#include "lib/Mathematics.h"

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>

#include <stdlib.h>

using namespace shogun;

const EMessageType IO::levels[NUM_LOG_LEVELS]={MSG_GCDEBUG, MSG_DEBUG, MSG_INFO, MSG_NOTICE,
	MSG_WARN, MSG_ERROR, MSG_CRITICAL, MSG_ALERT, MSG_EMERGENCY, MSG_MESSAGEONLY};

const char* IO::message_strings[NUM_LOG_LEVELS]={"[GCDEBUG] \0", "[DEBUG] \0", "[INFO] \0",
	"[NOTICE] \0", "[WARN] \0", "[ERROR] \0",
	"[CRITICAL] \0", "[ALERT] \0", "[EMERGENCY] \0", "\0"};

const char* IO::message_strings_highlighted[NUM_LOG_LEVELS]={"[GCDEBUG] \0", "[DEBUG] \0", "[INFO] \0",
	"[NOTICE] \0", "\033[1;34m[WARN]\033[0m \0", "\033[1;31m[ERROR]\033[0m \0",
	"[CRITICAL] \0", "[ALERT] \0", "[EMERGENCY] \0", "\0"};

/// file name buffer
char IO::file_buffer[FBUFSIZE];

/// directory name buffer
char IO::directory_name[FBUFSIZE];

IO::IO()
: target(stdout), last_progress_time(0), progress_start_time(0),
	last_progress(1), show_progress(false), show_file_and_line(false),
	syntax_highlight(true), loglevel(MSG_WARN), refcount(0)
{
}

IO::IO(const IO& orig)
: target(orig.get_target()), last_progress_time(0),
	progress_start_time(0), last_progress(1),
	show_progress(orig.get_show_progress()),
	show_file_and_line(orig.get_show_file_and_line()),
	syntax_highlight(orig.get_syntax_highlight()),
	loglevel(orig.get_loglevel()), refcount(0)
{
}

void IO::message(EMessageType prio, const char* file,
		int32_t line, const char *fmt, ... ) const
{
	const char* msg_intro=get_msg_intro(prio);

	if (msg_intro)
	{
		char str[4096];
		snprintf(str, sizeof(str), "%s", msg_intro);
		int len=strlen(msg_intro);
		char* s=str+len;

		if (show_file_and_line && line>=0)
		{
			snprintf(s, sizeof(str)-len, "In file %s line %d: ", file, line);
			len=strlen(str);
			s=str+len;
		}

		va_list list;
		va_start(list,fmt);
		vsnprintf(s, sizeof(str)-len, fmt, list);
		va_end(list);

		switch (prio)
		{
			case MSG_GCDEBUG:
			case MSG_DEBUG:
			case MSG_INFO:
			case MSG_NOTICE:
			case MSG_MESSAGEONLY:
				if (sg_print_message)
					sg_print_message(target, str);
				break;

			case MSG_WARN:
				if (sg_print_warning)
					sg_print_warning(target, str);
				break;

			case MSG_ERROR:
			case MSG_CRITICAL:
			case MSG_ALERT:
			case MSG_EMERGENCY:
				if (sg_print_error)
					sg_print_error(target, str);
				throw ShogunException(str);
				break;
			default:
				break;
		}

		fflush(target);
	}
}

void IO::buffered_message(EMessageType prio, const char *fmt, ... ) const
{
	const char* msg_intro=get_msg_intro(prio);

	if (msg_intro)
	{
		fprintf(target, "%s", msg_intro);
		va_list list;
		va_start(list,fmt);
		vfprintf(target,fmt,list);
		va_end(list);
	}
}

void IO::progress(
	float64_t current_val, float64_t min_val, float64_t max_val,
	int32_t decimals, const char* prefix)
{
	if (!show_progress)
		return;

	float64_t runtime = CTime::get_curtime();

	char str[1000];
	float64_t v=-1, estimate=0, total_estimate=0 ;

	if (max_val-min_val>0.0)
		v=100*(current_val-min_val+1)/(max_val-min_val+1);

	if (decimals < 1)
		decimals = 1;

	if (last_progress>v)
	{
		last_progress_time = runtime;
		progress_start_time = runtime;
		last_progress = v;
	}
	else
	{
		v=CMath::clamp(v,1e-5,100.0);
		last_progress = v-1e-6;

		if ((v!=100.0) && (runtime - last_progress_time<0.5))
			return;

		last_progress_time = runtime;
		estimate = (1-v/100)*(last_progress_time-progress_start_time)/(v/100);
		total_estimate = (last_progress_time-progress_start_time)/(v/100);
	}

	if (estimate>120)
	{
		snprintf(str, sizeof(str), "%%s %%%d.%df%%%%    %%1.1f minutes remaining    %%1.1f minutes total    \r",decimals+3, decimals);
		message(MSG_MESSAGEONLY, "", -1, str, prefix, v, estimate/60, total_estimate/60);
	}
	else
	{
		snprintf(str, sizeof(str), "%%s %%%d.%df%%%%    %%1.1f seconds remaining    %%1.1f seconds total    \r",decimals+3, decimals);
		message(MSG_MESSAGEONLY, "", -1, str, prefix, v, estimate, total_estimate);
	}

    fflush(target);
}

void IO::absolute_progress(
	float64_t current_val, float64_t val, float64_t min_val, float64_t max_val,
	int32_t decimals, const char* prefix)
{
	if (!show_progress)
		return;

	float64_t runtime = CTime::get_curtime();

	char str[1000];
	float64_t v=-1, estimate=0, total_estimate=0 ;

	if (max_val-min_val>0)
		v=100*(val-min_val+1)/(max_val-min_val+1);

	if (decimals < 1)
		decimals = 1;

	if (last_progress>v)
	{
		last_progress_time = runtime;
		progress_start_time = runtime;
		last_progress = v;
	}
	else
	{
		v=CMath::clamp(v,1e-5,100.0);
		last_progress = v-1e-6;

		if ((v!=100.0) && (runtime - last_progress_time<100))
			return;

		last_progress_time = runtime;
		estimate = (1-v/100)*(last_progress_time-progress_start_time)/(v/100);
		total_estimate = (last_progress_time-progress_start_time)/(v/100);
	}

	if (estimate>120)
	{
		snprintf(str, sizeof(str), "%%s %%%d.%df    %%1.1f minutes remaining    %%1.1f minutes total    \r",decimals+3, decimals);
		message(MSG_MESSAGEONLY, "", -1, str, prefix, current_val, estimate/60, total_estimate/60);
	}
	else
	{
		snprintf(str, sizeof(str), "%%s %%%d.%df    %%1.1f seconds remaining    %%1.1f seconds total    \r",decimals+3, decimals);
		message(MSG_MESSAGEONLY, "", -1, str, prefix, current_val, estimate, total_estimate);
	}

    fflush(target);
}

void IO::done()
{
	if (!show_progress)
		return;

	message(MSG_INFO, "", -1, "done.\n");
}

char* IO::skip_spaces(char* str)
{
	int32_t i=0;

	if (str)
	{
		for (i=0; isspace(str[i]); i++);

		return &str[i];
	}
	else
		return str;
}

char* IO::skip_blanks(char* str)
{
	int32_t i=0;

	if (str)
	{
		for (i=0; isblank(str[i]); i++);

		return &str[i];
	}
	else
		return str;
}

EMessageType IO::get_loglevel() const
{
	return loglevel;
}

void IO::set_loglevel(EMessageType level)
{
	loglevel=level;
}

void IO::set_target(FILE* t)
{
	target=t;
}

const char* IO::get_msg_intro(EMessageType prio) const
{
	for (int32_t i=NUM_LOG_LEVELS-1; i>=0; i--)
	{
		// ignore msg if prio's level is under loglevel,
		// but not if prio's level higher than MSG_WARN
		if (levels[i]<loglevel && prio<=MSG_WARN)
			return NULL;

		if (levels[i]==prio)
		{
			if (syntax_highlight)
				return message_strings_highlighted[i];
			else
				return message_strings[i];
		}
	}

	return NULL;
}
