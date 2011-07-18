/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __IO_H__
#define __IO_H__

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
	class IO;
	extern IO* sg_io;
}


namespace shogun
{
/** The io libs output [DEBUG] etc in front of every message 'higher' messages
 * filter output depending on the loglevel, i.e. CRITICAL messages will print
 * all MSG_CRITICAL TO MSG_EMERGENCY messages.
 */
enum EMessageType
{
	MSG_GCDEBUG,
	MSG_DEBUG,
	MSG_INFO,
	MSG_NOTICE,
	MSG_WARN,
	MSG_ERROR,
	MSG_CRITICAL,
	MSG_ALERT,
	MSG_EMERGENCY,
	MSG_MESSAGEONLY
};


#define NUM_LOG_LEVELS 10
#define FBUFSIZE 4096

#ifdef DARWIN
#define CONST_DIRENT_T struct dirent
#else //DARWIN
#define CONST_DIRENT_T const struct dirent
#endif //DARWIN

#define SG_SET_LOCALE_C setlocale(LC_ALL, "C")
#define SG_RESET_LOCALE setlocale(LC_ALL, "")

// printf like funktions (with additional severity level)
// for object derived from CSGObject
#define SG_GCDEBUG(...) io->message(MSG_GCDEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define SG_DEBUG(...) io->message(MSG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define SG_INFO(...) io->message(MSG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define SG_WARNING(...) io->message(MSG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define SG_ERROR(...) io->message(MSG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define SG_UNSTABLE(func, ...) io->message(MSG_WARN, __FILE__, __LINE__, \
__FILE__ ":" func ": Unstable method!  Please report if it seems to " \
"work or not to the Shogun mailing list.  Thanking you in " \
"anticipation.  " __VA_ARGS__)

#define SG_PRINT(...) io->message(MSG_MESSAGEONLY, __FILE__, __LINE__, __VA_ARGS__)
#define SG_NOTIMPLEMENTED io->not_implemented(__FILE__, __LINE__)
#define SG_DEPRECATED io->deprecated(__FILE__, __LINE__)

#define SG_PROGRESS(...) io->progress(__VA_ARGS__)
#define SG_ABS_PROGRESS(...) io->absolute_progress(__VA_ARGS__)
#define SG_DONE() io->done()

// printf like function using the global sg_io object
#define SG_SGCDEBUG(...) sg_io->message(MSG_GCDEBUG,__FILE__, __LINE__, __VA_ARGS__)
#define SG_SDEBUG(...) sg_io->message(MSG_DEBUG,__FILE__, __LINE__, __VA_ARGS__)
#define SG_SINFO(...) sg_io->message(MSG_INFO,__FILE__, __LINE__, __VA_ARGS__)
#define SG_SWARNING(...) sg_io->message(MSG_WARN,__FILE__, __LINE__, __VA_ARGS__)
#define SG_SERROR(...) sg_io->message(MSG_ERROR,__FILE__, __LINE__, __VA_ARGS__)
#define SG_SPRINT(...) sg_io->message(MSG_MESSAGEONLY,__FILE__, __LINE__, __VA_ARGS__)
#define SG_SPROGRESS(...) sg_io->progress(__VA_ARGS__)
#define SG_SABS_PROGRESS(...) sg_io->absolute_progress(__VA_ARGS__)
#define SG_SDONE() sg_io->done()
#define SG_SNOTIMPLEMENTED sg_io->not_implemented(__FILE__, __LINE__)
#define SG_SDEPRECATED sg_io->deprecated(__FILE__, __LINE__)

#define ASSERT(x) { if (!(x)) SG_SERROR("assertion %s failed in file %s line %d\n",#x, __FILE__, __LINE__);}


/** @brief Class IO, used to do input output operations throughout shogun.
 *
 * Any debug or error or progress message is passed through the functions of
 * this class to be in the end written to the screen. Note that messages don't
 * have to be written to stdout or stderr, but can be redirected to a file.
 */
class IO
{
	public:
		/** default constructor */
		IO();
		/** copy constructor */
		IO(const IO& orig);

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

		/** get show_progress
		 *
		 * @return if progress bar is shown
		 */
		inline bool get_show_progress() const
		{
			return show_progress;
		}

		/** get show file and line
		 *
		 * @return if file and line should prefix messages
		 */
		inline bool get_show_file_and_line() const
		{
			return show_file_and_line;
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
		void message(EMessageType prio, const char* file,
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
		inline void not_implemented(const char* file, int32_t line) const
		{
			message(MSG_ERROR, file, line, "Sorry, not yet implemented .\n");
		}

		/** print warning message 'function deprecated' */
		inline void deprecated(const char* file, int32_t line) const
		{
			message(MSG_WARN, file, line,
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

		/** enable displaying of file and line when printing messages*/
		inline void enable_file_and_line()
		{
			show_file_and_line=true;

			if (sg_io!=this)
				sg_io->enable_file_and_line();
		}

		/** disable displaying of file and line when printing messages*/
		inline void disable_file_and_line()
		{
			show_file_and_line=false;

			if (sg_io!=this)
				sg_io->disable_file_and_line();
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
        static inline char* concat_filename(const char* filename)
        {
            if (snprintf(file_buffer, FBUFSIZE, "%s/%s", directory_name, filename) > FBUFSIZE)
                SG_SERROR("filename too long");
            SG_SDEBUG("filename=\"%s\"\n", file_buffer);
            return file_buffer;
        }

		/** filter
		 *
		 * @param d directory entry
		 * @return 1 if d is a readable file
		 */
		static inline int filter(CONST_DIRENT_T* d)
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

		/** increase reference counter
		 *
		 * @return reference count
		 */
		inline int32_t ref()
		{
			++refcount;
			return refcount;
		}

		/** display reference counter
		 *
		 * @return reference count
		 */
		inline int32_t ref_count() const
		{
			return refcount;
		}

		/** decrement reference counter and deallocate object if refcount is zero
		 * before or after decrementing it
		 *
		 * @return reference count
		 */
		inline int32_t unref()
		{
			if (refcount==0 || --refcount==0)
			{
				delete this;
				return 0;
			}
			else
				return refcount;
		}

		/** @return object name */
		inline const char* get_name() { return "IO"; }

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
		 * where the print occurs */
		bool show_file_and_line;
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
		int32_t refcount;
};
}
#endif // __IO_H__
