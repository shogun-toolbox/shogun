/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Giovanni De Toni, Viktor Gal, Heiko Strathmann,
 *          Thoralf Klein, Evangelos Anagnostopoulos, Weijie Lin, Bjoern Esser,
 *          Saurabh Goyal
 */

#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/lib/memory.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/RefCount.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <sstream>
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

SGIO::SGIO()
    : show_progress(false), location_info(MSG_NONE),
      syntax_highlight(true)
{
	m_refcount = new RefCount();
	logger = spdlog::stdout_color_mt("console");
	update_pattern();
}

SGIO::SGIO(const SGIO& orig)
    : show_progress(orig.get_show_progress()),
      location_info(orig.get_location_info()),
      syntax_highlight(orig.get_syntax_highlight()),
	  logger(orig.logger)
{
	m_refcount = new RefCount();
	update_pattern();
}

void SGIO::done()
{
	if (!show_progress)
		return;

	message(MSG_INFO, "done.\n");
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
	return static_cast<EMessageType>(logger->level());
}

void SGIO::set_loglevel(EMessageType level)
{
	logger->set_level(static_cast<spdlog::level::level_enum>(level));
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

void SGIO::update_pattern()
{
	std::stringstream pattern_builder;
	pattern_builder << "[%D %T ";
	switch(location_info)
	{
	case MSG_LINE_AND_FILE:
		pattern_builder << "%@ ";
		break;
	case MSG_FUNCTION:
		pattern_builder << "%! ";
		break;
	default:
		break;
	}
	if (syntax_highlight)
		pattern_builder << "%^%l%$] ";
	else
		pattern_builder << "%l] ";

	logger->set_pattern(pattern_builder.str());
}
