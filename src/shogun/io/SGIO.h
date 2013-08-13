/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2013 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010-2013 Soeren Sonnenburg
 */

#ifndef __SGIO_H__
#define __SGIO_H__

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <locale.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <shogun/lib/common.h>
#include <shogun/base/init.h>

namespace shogun
{
	class RefCount;
	class SGIO;
	/** shogun IO */
	extern SGIO* sg_io;
}


namespace shogun
{
/** The io libs output [DEBUG] etc in front of every message 'higher' messages
 * filter output depending on the loglevel, i.e. CRITICAL messages will print
 * all MSG_CRITICAL TO MSG_EMERGENCY messages.
 */
enum EMessageType
{
	MSG_GCDEBUG=0,
	MSG_DEBUG=1,
	MSG_INFO=2,
	MSG_NOTICE=3,
	MSG_WARN=4,
	MSG_ERROR=5,
	MSG_CRITICAL=6,
	MSG_ALERT=7,
	MSG_EMERGENCY=8,
	MSG_MESSAGEONLY=9
};

enum EMessageLocation
{
	MSG_NONE=0,
	MSG_FUNCTION=1,
	MSG_LINE_AND_FILE=2
};

#define NUM_LOG_LEVELS 10
#define FBUFSIZE 4096

#ifdef DARWIN
#include <Availability.h>
#ifndef __MAC_10_8
#define __MAC_10_8 1080
#endif
#if __MAC_OS_X_VERSION_MIN_REQUIRED < __MAC_10_8
#define CONST_DIRENT_T struct dirent
#else
#define CONST_DIRENT_T const struct dirent
#endif // Lion or earlier
#else //DARWIN
#define CONST_DIRENT_T const struct dirent
#endif //DARWIN

#define SG_SET_LOCALE_C setlocale(LC_ALL, "C")
#define SG_RESET_LOCALE setlocale(LC_ALL, "")

#if !defined(SG_UNLIKELY)
#if __GNUC__ >= 3
#define SG_UNLIKELY(expr) __builtin_expect(expr, 0)
#else
#define SG_UNLIKELY(expr) expr
#endif
#endif

// printf like funktions (with additional severity level)
// for object derived from CSGObject
#define SG_GCDEBUG(...) {											\
	if (SG_UNLIKELY(io->loglevel_above(MSG_GCDEBUG)))				\
		io->message(MSG_GCDEBUG, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);	\
}

#define SG_DEBUG(...) {												\
	if (SG_UNLIKELY(io->loglevel_above(MSG_DEBUG)))					\
		io->message(MSG_DEBUG, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);	\
}

#define SG_OBJ_DEBUG(o,...) {										\
	if (SG_UNLIKELY(o->io->loglevel_above(MSG_DEBUG)))				\
		o->io->message(MSG_DEBUG, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);	\
}


#define SG_INFO(...) {												\
	if (SG_UNLIKELY(io->loglevel_above(MSG_INFO)))					\
		io->message(MSG_INFO, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);		\
}

#define SG_CLASS_INFO(c, ...) {										\
	if (SG_UNLIKELY(c::io->loglevel_above(MSG_INFO)))				\
		c::io->message(MSG_INFO, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);	\
}

#define SG_WARNING(...) { io->message(MSG_WARN, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); }
#define SG_ERROR(...) { io->message(MSG_ERROR, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); }
#define SG_OBJ_ERROR(o, ...) { o->io->message(MSG_ERROR, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); }
#define SG_CLASS_ERROR(c, ...) { c::io->message(MSG_ERROR, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); }
#define SG_UNSTABLE(func, ...) { io->message(MSG_WARN, __PRETTY_FUNCTION__, __FILE__, __LINE__, \
__FILE__ ":" func ": Unstable method!  Please report if it seems to " \
"work or not to the Shogun mailing list.  Thanking you in " \
"anticipation.  " __VA_ARGS__); }

#define SG_PRINT(...) { io->message(MSG_MESSAGEONLY, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); }
#define SG_OBJ_PRINT(o, ...) { o->io->message(MSG_MESSAGEONLY, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); }
#define SG_NOTIMPLEMENTED { io->not_implemented(__PRETTY_FUNCTION__, __FILE__, __LINE__); }
#define SG_DEPRECATED { io->deprecated(__PRETTY_FUNCTION__, __FILE__, __LINE__); }

#define SG_PROGRESS(...) {						\
	if (SG_UNLIKELY(io->get_show_progress()))	\
		io->progress(__VA_ARGS__);				\
}

#define SG_OBJ_PROGRESS(o, ...) {				\
	if (SG_UNLIKELY(o->io->get_show_progress()))\
		o->io->progress(__VA_ARGS__);			\
}

#define SG_ABS_PROGRESS(...) {					\
	if (SG_UNLIKELY(io->get_show_progress()))	\
		io->absolute_progress(__VA_ARGS__);		\
}

#define SG_DONE() {								\
	if (SG_UNLIKELY(io->get_show_progress()))	\
		io->done();								\
}

// printf like function using the global sg_io object
#define SG_SGCDEBUG(...) {											\
	if (SG_UNLIKELY(sg_io->loglevel_above(MSG_GCDEBUG)))			\
		sg_io->message(MSG_GCDEBUG,__PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);\
}

#define SG_SDEBUG(...) {											\
	if (SG_UNLIKELY(sg_io->loglevel_above(MSG_DEBUG)))			\
		sg_io->message(MSG_DEBUG,__PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);	\
}

#define SG_SINFO(...) {												\
	if (SG_UNLIKELY(sg_io->loglevel_above(MSG_INFO)))				\
		sg_io->message(MSG_INFO,__PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__);	\
}

#define SG_SWARNING(...) { sg_io->message(MSG_WARN,__PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); }
#define SG_SERROR(...) { sg_io->message(MSG_ERROR,__PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); }
#define SG_SPRINT(...) { sg_io->message(MSG_MESSAGEONLY,__PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); }


#define SG_SPROGRESS(...) {							\
	if (SG_UNLIKELY(sg_io->get_show_progress()))	\
		sg_io->progress(__VA_ARGS__);				\
}

#define SG_SABS_PROGRESS(...) {						\
	if (SG_UNLIKELY(sg_io->get_show_progress()))	\
		sg_io->absolute_progress(__VA_ARGS__);		\
}

#define SG_SDONE() {								\
	if (SG_UNLIKELY(sg_io->get_show_progress()))	\
		sg_io->done();								\
}

#define SG_SNOTIMPLEMENTED { sg_io->not_implemented(__PRETTY_FUNCTION__, __FILE__, __LINE__); }
#define SG_SDEPRECATED { sg_io->deprecated(__PRETTY_FUNCTION__, __FILE__, __LINE__); }

#define ASSERT(x) { 																	\
	if (SG_UNLIKELY(!(x)))																\
		SG_SERROR("assertion %s failed in %s file %s line %d\n",#x, __PRETTY_FUNCTION__, __FILE__, __LINE__)	\
}

#define REQUIRE(x, ...) {		\
	if (SG_UNLIKELY(!(x)))		\
		SG_SERROR(__VA_ARGS__)	\
}

/**
 * @brief struct Substring, specified by
 * start position and end position.
 *
 * Used to mark strings in a buffer, where they
 * need not be delimited by NUL characters.
 */
struct substring
{
	/** start */
	char *start;
	/** end */
	char *end;
};

/** @brief Class SGIO, used to do input output operations throughout shogun.
 *
 * Any debug or error or progress message is passed through the functions of
 * this class to be in the end written to the screen. Note that messages don't
 * have to be written to stdout or stderr, but can be redirected to a file.
 */
class SGIO
{
	public:
		/** default constructor */
		SGIO();
		/** copy constructor */
		SGIO(const SGIO& orig);

		/** destructor */
		virtual ~SGIO();

		/** set loglevel
		 *
		 * @param level level of log messages
		 */
		void set_loglevel(EMessageType level);

		/** get loglevel
		 *
		 * @return level of log messages
		 */
		EMessageType get_loglevel() const;

		/** loglevel above
		 *
		 * @return whether loglevel is above specified level and thus the
		 * message should be printed
		 */
		inline bool loglevel_above(EMessageType type) const
		{
			return loglevel <= type;
		}

		/** get show_progress
		 *
		 * @return if progress bar is shown
		 */
		inline bool get_show_progress() const
		{
			return show_progress;
		}

		/** show location where printing occurs
		 */
		inline EMessageLocation get_location_info() const
		{
			return location_info;
		}

		/** get syntax highlight
		 *
		 * @return if syntax highlighting is enabled
		 */
		inline bool get_syntax_highlight() const
		{
			return syntax_highlight;
		}

		/** print a message
		 *
		 * optionally prefixed with file name and line number
		 * from (use -1 in line to disable this)
		 *
		 * @param prio message priority
		 * @param file file name from where the message is called
		 * @param line line number from where the message is called
		 * @param fmt format string
		 */
		void message(EMessageType prio, const char* function, const char* file,
				int32_t line, const char *fmt, ... ) const;

		/** print progress bar
		 *
		 * @param current_val current value
		 * @param min_val minimum value
		 * @param max_val maximum value
		 * @param decimals decimals
		 * @param prefix message prefix
		 */
		void progress(
			float64_t current_val,
			float64_t min_val=0.0, float64_t max_val=1.0, int32_t decimals=1,
			const char* prefix="PROGRESS:\t");

		/** print absolute progress bar
		 *
		 * @param current_val current value
		 * @param val value
		 * @param min_val minimum value
		 * @param max_val maximum value
		 * @param decimals decimals
		 * @param prefix message prefix
		 */
		void absolute_progress(
			float64_t current_val, float64_t val,
			float64_t min_val=0.0, float64_t max_val=1.0, int32_t decimals=1,
			const char* prefix="PROGRESS:\t");

		/** print 'done' with priority INFO,
		 * but only if progress bar is enabled
		 *
		 */
		void done();

		/** print error message 'not implemented' */
		inline void not_implemented(const char* function, const char* file, int32_t line) const
		{
			message(MSG_ERROR, function, file, line, "Sorry, not yet implemented .\n");
		}

		/** print warning message 'function deprecated' */
		inline void deprecated(const char* function, const char* file, int32_t line) const
		{
			message(MSG_WARN, function, file, line,
					"This function is deprecated and will be removed soon.\n");
		}

		/** print a buffered message
		 *
		 * @param prio message priority
		 * @param fmt format string
		 */
		void buffered_message(EMessageType prio, const char *fmt, ... ) const;

		/** skip leading spaces
		 *
		 * @param str string in which to look for spaces
		 * @return string after after skipping leading spaces
		 */
		static char* skip_spaces(char* str);

		/** skip leading spaces + tabs
		 *
		 * @param str string in which to look for blanks
		 * @return string after after skipping leading blanks
		 */
		static char* skip_blanks(char* str);

		/** get target
		 *
		 * @return file descriptor for target
		 */
		inline FILE* get_target() const
		{
			return target;
		}

		/** set target
		 *
		 * @param target file descriptor for target
		 */
		void set_target(FILE* target);

		/** set target to stderr */
		inline void set_target_to_stderr() { set_target(stderr); }

		/** set target to stdout */
		inline void set_target_to_stdout() { set_target(stdout); }

		/** enable progress bar */
		inline void enable_progress()
		{
			show_progress=true;

			// static functions like CSVM::classify_example_helper call SG_PROGRESS
			if (sg_io!=this)
				sg_io->enable_progress();
		}

		/** disable progress bar */
		inline void disable_progress()
		{
			show_progress=false;

			// static functions like CSVM::classify_example_helper call SG_PROGRESS
			if (sg_io!=this)
				sg_io->disable_progress();
		}

		/** enable displaying of file and line when printing messages etc
		 *
		 * @param location location info (none, function, ...)
		 *
		 */
		inline void set_location_info(EMessageLocation location)
		{
			location_info = location;

			if (sg_io!=this)
				sg_io->set_location_info(location);
		}

		/** enable syntax highlighting */
		inline void enable_syntax_highlighting()
		{
			syntax_highlight=true;

			if (sg_io!=this)
				sg_io->enable_syntax_highlighting();
		}

		/** disable syntax highlighting */
		inline void disable_syntax_highlighting()
		{
			syntax_highlight=false;

			if (sg_io!=this)
				sg_io->disable_syntax_highlighting();
		}

		/** set directory name
		 *
		 * @param dirname new directory name
		 */
		static inline void set_dirname(const char* dirname)
		{
			strncpy(directory_name, dirname, FBUFSIZE);
		}

		/** concatenate directory and filename
		 * ( non thread safe )
		 *
		 * @param filename new filename
		 * @return concatenated directory and filename
		 */
        static char* concat_filename(const char* filename);

		/** filter
		 *
		 * @param d directory entry
		 * @return 1 if d is a readable file
		 */
		static int filter(CONST_DIRENT_T* d);

		/**
		 * Return a C string from the substring
		 * @param s substring
		 * @return new C string representation
		 */
		static char* c_string_of_substring(substring s);

		/**
		 * Print the substring
		 * @param s substring
		 */
		static void print_substring(substring s);

		/**
		 * Get value of substring as float
		 * (if possible)
		 * @param s substring
		 * @return float32_t value of substring
		 */
		static float32_t float_of_substring(substring s);

		/**
		 * Return value of substring as double
		 * @param s substring
		 * @return substring as double
		 */
		static float64_t double_of_substring(substring s);

		/**
		 * Integer value of substring
		 * @param s substring
		 * @return int value of substring
		 */
		static int32_t int_of_substring(substring s);

		/**
		 * Unsigned long value of substring
		 * @param s substring
		 * @return unsigned long value of substring
		 */
		static uint32_t ulong_of_substring(substring s);

		/**
		 * Length of substring
		 * @param s substring
		 * @return length of substring
		 */
		static uint32_t ss_length(substring s);

		/** increase reference counter
		 *
		 * @return reference count
		 */
		int32_t ref();

		/** display reference counter
		 *
		 * @return reference count
		 */
		int32_t ref_count() const;

		/** decrement reference counter and deallocate object if refcount is zero
		 * before or after decrementing it
		 *
		 * @return reference count
		 */
		int32_t unref();

		/** @return object name */
		inline const char* get_name() { return "SGIO"; }

	protected:
		/** get message intro
		 *
		 * @param prio message priority
		 * @return message intro or NULL if message is not to be
		 *         printed
		 */
		const char* get_msg_intro(EMessageType prio) const;

	protected:
		/** target file */
		FILE* target;
		/** last progress time */
		float64_t last_progress_time;
		/** progress start time */
		float64_t progress_start_time;
		/** last progress */
		float64_t last_progress;
		/** if progress bar shall be shown */
		bool show_progress;
		/** if each print function should append filename and linenumber of
		 * where the print occurs etc */
		EMessageLocation location_info;
		/** whether syntax highlighting is enabled */
		bool syntax_highlight;

		/** log level */
		EMessageType loglevel;
		/** available log levels */
		static const EMessageType levels[NUM_LOG_LEVELS];
		/** message strings syntax highlighted*/
		static const char* message_strings_highlighted[NUM_LOG_LEVELS];
		/** message strings */
		static const char* message_strings[NUM_LOG_LEVELS];

        /** file buffer */
        static char file_buffer[FBUFSIZE];
        /** directory name (for filter function) */
        static char directory_name[FBUFSIZE];

	private:
		RefCount* m_refcount;
};
}
#endif // __SGIO_H__
