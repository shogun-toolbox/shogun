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

#ifndef __CIO_H__
#define __CIO_H__

#include <time.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "lib/common.h"


/// The io libs output [DEBUG] etc in front of every message
/// 'higher' messages filter output depending on the loglevel, i.e. CRITICAL messages
/// will print all M_CRITICAL TO M_EMERGENCY messages to
enum EMessageType
{
	M_DEBUG,
	M_INFO,
	M_NOTICE,
	M_WARN,
	M_ERROR,
	M_CRITICAL,
	M_ALERT,
	M_EMERGENCY,
	M_MESSAGEONLY
};


#define NUM_LOG_LEVELS 9
#define FBUFSIZE 4096

#ifdef DARWIN
#define CONST_DIRENT_T struct dirent
#else //DARWIN
#define CONST_DIRENT_T const struct dirent
#endif //DARWIN

extern char file_buffer[FBUFSIZE];
extern char directory_name[FBUFSIZE];

class CIO;

// printf like funktions (with additional severity level)
// for object derived from CSGObject
#define SG_DEBUG(...) CSGObject::io.message(M_DEBUG, __VA_ARGS__)
#define SG_INFO(...) CSGObject::io.message(M_INFO, __VA_ARGS__)
#define SG_WARNING(...) CSGObject::io.message(M_WARN, __VA_ARGS__)
#define SG_ERROR(...) CSGObject::io.message(M_ERROR, __VA_ARGS__)
#define SG_PRINT(...) CSGObject::io.message(M_MESSAGEONLY, __VA_ARGS__)
#define SG_NOTIMPLEMENTED CSGObject::io.not_implemented()

#define SG_PROGRESS(...) CSGObject::io.progress(__VA_ARGS__)
#define SG_ABS_PROGRESS(...) CSGObject::io.absolute_progress(__VA_ARGS__)
#define SG_DONE() CSGObject::io.done()

#ifndef HAVE_SWIG
extern CIO* sg_io;
// printf like function using the global sg_io object
#define SG_SDEBUG(...) sg_io->message(M_DEBUG,__VA_ARGS__)
#define SG_SINFO(...) sg_io->message(M_INFO,__VA_ARGS__)
#define SG_SWARNING(...) sg_io->message(M_WARN,__VA_ARGS__)
#define SG_SERROR(...) sg_io->message(M_ERROR,__VA_ARGS__)
#define SG_SPRINT(...) sg_io->message(M_MESSAGEONLY,__VA_ARGS__)
#define SG_SPROGRESS(...) sg_io->progress(__VA_ARGS__)
#define SG_SABS_PROGRESS(...) sg_io->absolute_progress(__VA_ARGS__)
#define SG_SDONE() sg_io->done()
#define SG_SNOTIMPLEMENTED sg_io->not_implemented()
#else
extern CIO sg_io;
// printf like function using the global sg_io object
#define SG_SDEBUG(...) sg_io.message(M_DEBUG, __VA_ARGS__)
#define SG_SINFO(...) sg_io.message(M_INFO, __VA_ARGS__)
#define SG_SWARNING(...) sg_io.message(M_WARN, __VA_ARGS__)
#define SG_SERROR(...) sg_io.message(M_ERROR, __VA_ARGS__)
#define SG_SPRINT(...) sg_io.message(M_MESSAGEONLY, __VA_ARGS__)
#define SG_SPROGRESS(...) sg_io.progress(__VA_ARGS__)
#define SG_SABS_PROGRESS(...) sg_io.absolute_progress(__VA_ARGS__)
#define SG_SDONE() sg_io.done()
#define SG_SNOTIMPLEMENTED sg_io.not_implemented()
#endif

#define ASSERT(x) { if (!(x)) SG_SERROR("assertion %s failed in file %s line %d\n",#x, __FILE__, __LINE__);}


/** Class IO, used to do input output operations throughout shogun, i.e. any
 * debug or error or progress message is passed through the functions of this
 * class to be in the end written to the screen. Note that messages don't have
 * to be written to stdout or stderr, but can be redirected to a file.
 */
class CIO
{
	public:
		/** default constructor */
		CIO();
		/** copy constructor */
		CIO(const CIO& orig);

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
		bool get_show_progress() const;

		/** print a message
		 *
		 * @param prio message priority
		 * @param fmt format string
		 */
		void message(EMessageType prio, const char *fmt, ... ) const;

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
		inline void not_implemented() const
		{
			message(M_ERROR, "Sorry, not yet implemented\n");
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
#ifndef HAVE_SWIG
			if (sg_io!=this)
				sg_io->enable_progress();
#else
			if (&sg_io!=this)
				sg_io.enable_progress();
#endif
		}

		/** disable progress bar */
		inline void disable_progress()
		{
			show_progress=false;

			// static functions like CSVM::classify_example_helper call SG_PROGRESS
#ifndef HAVE_SWIG
			if (sg_io!=this)
				sg_io->disable_progress();
#else
			if (&sg_io!=this)
				sg_io.disable_progress();
#endif
		}

		/** set directory name
		 *
		 * @param dirname new directory name
		 */
		inline void set_dirname(const char* dirname)
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
		 * @return if filtering was successful
		 */
		static int filter(CONST_DIRENT_T* d);

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
		int64_t last_progress_time;
		/** progress start time */
		int64_t progress_start_time;
		/** last progress */
		float64_t last_progress;
		/** if progress bar shall be shown */
		bool show_progress;

		/** log level */
		EMessageType loglevel;
		/** available log levels */
		static const EMessageType levels[NUM_LOG_LEVELS];
		/** message strings */
		static const char* message_strings[NUM_LOG_LEVELS];
};

#endif // __CIO_H__
