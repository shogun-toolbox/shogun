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

#include <shogun/io/SGIO.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>
#include <shogun/lib/memory.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/RefCount.h>

#include <stdarg.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>

using namespace shogun;

const EMessageType SGIO::levels[NUM_LOG_LEVELS]={MSG_GCDEBUG, MSG_DEBUG, MSG_INFO, MSG_NOTICE,
	MSG_WARN, MSG_ERROR, MSG_CRITICAL, MSG_ALERT, MSG_EMERGENCY, MSG_MESSAGEONLY};

const char* SGIO::message_strings[NUM_LOG_LEVELS]={"[GCDEBUG] \0", "[DEBUG] \0", "[INFO] \0",
	"[NOTICE] \0", "[WARN] \0", "[ERROR] \0",
	"[CRITICAL] \0", "[ALERT] \0", "[EMERGENCY] \0", "\0"};

const char* SGIO::message_strings_highlighted[NUM_LOG_LEVELS]={"[GCDEBUG] \0", "[DEBUG] \0", "[INFO] \0",
	"[NOTICE] \0", "\033[1;34m[WARN]\033[0m \0", "\033[1;31m[ERROR]\033[0m \0",
	"[CRITICAL] \0", "[ALERT] \0", "[EMERGENCY] \0", "\0"};

/// file name buffer
char SGIO::file_buffer[FBUFSIZE];

/// directory name buffer
char SGIO::directory_name[FBUFSIZE];

SGIO::SGIO()
: target(stdout), last_progress_time(0), progress_start_time(0),
	last_progress(1), show_progress(false), location_info(MSG_NONE),
	syntax_highlight(true), loglevel(MSG_WARN)
{
	m_refcount = new RefCount();
}

SGIO::SGIO(const SGIO& orig)
: target(orig.get_target()), last_progress_time(0),
	progress_start_time(0), last_progress(1),
	show_progress(orig.get_show_progress()),
	location_info(orig.get_location_info()),
	syntax_highlight(orig.get_syntax_highlight()),
	loglevel(orig.get_loglevel())
{
	m_refcount = new RefCount();
}

void SGIO::message(EMessageType prio, const char* function, const char* file,
		int32_t line, const char *fmt, ... ) const
{
	const char* msg_intro=get_msg_intro(prio);

	if (msg_intro)
	{
		char str[4096];
		snprintf(str, sizeof(str), "%s", msg_intro);
		int len=strlen(msg_intro);
		char* s=str+len;

		/* file and line are shown for warnings and worse */
		if (location_info==MSG_LINE_AND_FILE || prio==MSG_WARN || prio==MSG_ERROR)
		{
			snprintf(s, sizeof(str)-len, "In file %s line %d: ", file, line);
			len=strlen(str);
			s=str+len;
		}
		else if (location_info==MSG_FUNCTION)
		{
			snprintf(s, sizeof(str)-len, "%s: ", function);
			len=strlen(str);
			s=str+len;
		}
		else if (location_info==MSG_NONE)
		{
			;
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

void SGIO::buffered_message(EMessageType prio, const char *fmt, ... ) const
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

void SGIO::progress(
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
		message(MSG_MESSAGEONLY, "", "", -1, str, prefix, v, estimate/60, total_estimate/60);
	}
	else
	{
		snprintf(str, sizeof(str), "%%s %%%d.%df%%%%    %%1.1f seconds remaining    %%1.1f seconds total    \r",decimals+3, decimals);
		message(MSG_MESSAGEONLY, "", "", -1, str, prefix, v, estimate, total_estimate);
	}

    fflush(target);
}

void SGIO::absolute_progress(
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
		message(MSG_MESSAGEONLY, "", "", -1, str, prefix, current_val, estimate/60, total_estimate/60);
	}
	else
	{
		snprintf(str, sizeof(str), "%%s %%%d.%df    %%1.1f seconds remaining    %%1.1f seconds total    \r",decimals+3, decimals);
		message(MSG_MESSAGEONLY, "", "", -1, str, prefix, current_val, estimate, total_estimate);
	}

    fflush(target);
}

void SGIO::done()
{
	if (!show_progress)
		return;

	message(MSG_INFO, "", "", -1, "done.\n");
}

char* SGIO::skip_spaces(char* str)
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

char* SGIO::skip_blanks(char* str)
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

EMessageType SGIO::get_loglevel() const
{
	return loglevel;
}

void SGIO::set_loglevel(EMessageType level)
{
	loglevel=level;
}

void SGIO::set_target(FILE* t)
{
	target=t;
}

const char* SGIO::get_msg_intro(EMessageType prio) const
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

char* SGIO::c_string_of_substring(substring s)
{
	uint32_t len = s.end - s.start+1;
	char* ret = SG_CALLOC(char, len);
	memcpy(ret,s.start,len-1);
	return ret;
}

void SGIO::print_substring(substring s)
{
	char* c_string = c_string_of_substring(s);
	SG_SPRINT("%s\n", c_string)
	SG_FREE(c_string);
}

float32_t SGIO::float_of_substring(substring s)
{
	char* endptr = s.end;
	float32_t f = strtof(s.start,&endptr);
	if (endptr == s.start && s.start != s.end)
		SG_SERROR("error: %s is not a float!\n", c_string_of_substring(s))

	return f;
}

float64_t SGIO::double_of_substring(substring s)
{
	char* endptr = s.end;
	float64_t f = strtod(s.start,&endptr);
	if (endptr == s.start && s.start != s.end)
		SG_SERROR("Error!:%s is not a double!\n", c_string_of_substring(s))

	return f;
}

int32_t SGIO::int_of_substring(substring s)
{
	char* c_string = c_string_of_substring(s);
	int32_t int_val = atoi(c_string);
	SG_FREE(c_string);

	return int_val;
}

uint32_t SGIO::ulong_of_substring(substring s)
{
	return strtoul(s.start,NULL,10);
}

uint32_t SGIO::ss_length(substring s)
{
	return (s.end - s.start);
}

char* SGIO::concat_filename(const char* filename)
{
	if (snprintf(file_buffer, FBUFSIZE, "%s/%s", directory_name, filename) > FBUFSIZE)
		SG_SERROR("filename too long")

	SG_SDEBUG("filename=\"%s\"\n", file_buffer)
	return file_buffer;
}

int SGIO::filter(CONST_DIRENT_T* d)
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

SGIO::~SGIO()
{
	delete m_refcount;
}

int32_t SGIO::ref()
{
	return m_refcount->ref();
}

int32_t SGIO::ref_count() const
{
	return m_refcount->ref_count();
}

int32_t SGIO::unref()
{
	int32_t rc = m_refcount->unref();
	if (rc==0)
	{
		delete this;
		return 0;
	}

	return rc;
}
