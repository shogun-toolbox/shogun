/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Giovanni De Toni, Viktor Gal, Heiko Strathmann,
 *          Thoralf Klein, Evangelos Anagnostopoulos, Weijie Lin, Bjoern Esser,
 *          Saurabh Goyal
 */

#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>
#include <shogun/lib/memory.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/RefCount.h>

#include <stdarg.h>
#include <ctype.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <stdlib.h>

#ifdef _WIN32
#define R_OK 4
#endif

using namespace shogun;

const EMessageType SGIO::levels[NUM_LOG_LEVELS]={MSG_GCDEBUG, MSG_DEBUG, MSG_INFO, MSG_NOTICE,
	MSG_WARN, MSG_ERROR, MSG_CRITICAL, MSG_ALERT, MSG_EMERGENCY, MSG_MESSAGEONLY};

const char* SGIO::message_strings[NUM_LOG_LEVELS]={"[GCDEBUG] \0", "[DEBUG] \0", "[INFO] \0",
	"[NOTICE] \0", "[WARN] \0", "[ERROR] \0",
	"[CRITICAL] \0", "[ALERT] \0", "[EMERGENCY] \0", "\0"};

const char* SGIO::message_strings_highlighted[NUM_LOG_LEVELS]={"[GCDEBUG] \0", "[DEBUG] \0", "[INFO] \0",
	"[NOTICE] \0", "\033[1;34m[WARN]\033[0m \0", "\033[1;31m[ERROR]\033[0m \0",
	"[CRITICAL] \0", "[ALERT] \0", "[EMERGENCY] \0", "\0"};

SGIO::SGIO()
    : target(stdout), show_progress(false), location_info(MSG_NONE),
      syntax_highlight(true), loglevel(MSG_WARN)
{
	m_refcount = new RefCount();
}

SGIO::SGIO(const SGIO& orig)
    : target(orig.get_target()), show_progress(orig.get_show_progress()),
      location_info(orig.get_location_info()),
      syntax_highlight(orig.get_syntax_highlight()),
      loglevel(orig.get_loglevel())
{
	m_refcount = new RefCount();
}

std::string SGIO::format(
    EMessageType prio, const char* function, const char* file, int32_t line,
    const char* fmt, ...) const
{
	const char* msg_intro=get_msg_intro(prio);

	char str[4096];
	snprintf(str, sizeof(str), "%s", msg_intro);
	int len = strlen(msg_intro);
	char* s = str + len;

	/* file and line are shown for warnings and worse */
	if (location_info == MSG_LINE_AND_FILE || prio == MSG_WARN ||
	    prio == MSG_ERROR)
	{
		snprintf(s, sizeof(str) - len, "In file %s line %d: ", file, line);
		len = strlen(str);
		s = str + len;
	}
	else if (location_info == MSG_FUNCTION)
	{
		snprintf(s, sizeof(str) - len, "%s: ", function);
		len = strlen(str);
		s = str + len;
	}
	else if (location_info == MSG_NONE)
	{
		;
	}

	va_list list;
	va_start(list, fmt);
	vsnprintf(s, sizeof(str) - len, fmt, list);
	va_end(list);

	return std::string(str);
}

void SGIO::print(EMessageType prio, const std::string& msg) const
{
	switch (prio)
	{
	case MSG_GCDEBUG:
	case MSG_DEBUG:
	case MSG_INFO:
	case MSG_NOTICE:
	case MSG_MESSAGEONLY:
		if (sg_print_message)
			sg_print_message(target, msg.c_str());
		break;

	case MSG_WARN:
		if (sg_print_warning)
			sg_print_warning(target, msg.c_str());
		break;

	case MSG_ERROR:
	case MSG_CRITICAL:
	case MSG_ALERT:
	case MSG_EMERGENCY:
		if (sg_print_error)
			sg_print_error(target, msg.c_str());
		break;
	default:
		break;
	}

	fflush(target);
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
		if (levels[i]==prio)
		{
			if (syntax_highlight)
				return message_strings_highlighted[i];
			else
				return message_strings[i];
		}
	}

	return nullptr; // unreachable
}

char* SGIO::c_string_of_substring(substring s)
{
	uint32_t len = s.end - s.start+1;
	char* ret = SG_CALLOC(char, len);
	sg_memcpy(ret,s.start,len-1);
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
