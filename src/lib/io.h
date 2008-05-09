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
#include "lib/common.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#define NUM_LOG_LEVELS 9
#define FBUFSIZE 4096

#ifdef DARWIN
#define CONST_DIRENT_T struct dirent
#else //DARWIN
#define CONST_DIRENT_T const struct dirent
#endif //DARWIN

extern CHAR file_buffer[FBUFSIZE];
extern CHAR directory_name[FBUFSIZE];

class CIO;

// printf like funktions (with additional severity level)
// for object derived from CSGObject
#define SG_DEBUG(x...) CSGObject::io.message(M_DEBUG,x)
#define SG_INFO(x...) CSGObject::io.message(M_INFO,x)
#define SG_WARNING(x...) CSGObject::io.message(M_WARN,x)
#define SG_ERROR(x...) CSGObject::io.message(M_ERROR,x)
#define SG_PRINT(x...) CSGObject::io.message(M_MESSAGEONLY,x)
#define SG_NOTIMPLEMENTED CSGObject::io.not_implemented()

#define SG_PROGRESS(x...) CSGObject::io.progress(x)
#define SG_ABS_PROGRESS(x...) CSGObject::io.absolute_progress(x)

#ifndef HAVE_SWIG
extern CIO* sg_io;
// printf like function using the global sg_io object
#define SG_SDEBUG(x...) sg_io->message(M_DEBUG,x)
#define SG_SINFO(x...) sg_io->message(M_INFO,x)
#define SG_SWARNING(x...) sg_io->message(M_WARN,x)
#define SG_SERROR(x...) sg_io->message(M_ERROR,x)
#define SG_SPRINT(x...) sg_io->message(M_MESSAGEONLY,x)
#define SG_SPROGRESS(x...) sg_io->progress(x)
#define SG_SABS_PROGRESS(x...) sg_io->absolute_progress(x)
#define SG_SNOTIMPLEMENTED sg_io->not_implemented()
#else
extern CIO sg_io;
// printf like function using the global sg_io object
#define SG_SDEBUG(x...) sg_io.message(M_DEBUG,x)
#define SG_SINFO(x...) sg_io.message(M_INFO,x)
#define SG_SWARNING(x...) sg_io.message(M_WARN,x)
#define SG_SERROR(x...) sg_io.message(M_ERROR,x)
#define SG_SPRINT(x...) sg_io.message(M_MESSAGEONLY,x)
#define SG_SPROGRESS(x...) sg_io.progress(x)
#define SG_SABS_PROGRESS(x...) sg_io.absolute_progress(x)
#define SG_SNOTIMPLEMENTED sg_io.not_implemented()
#endif

#define ASSERT(x) { if (!(x)) SG_SERROR("assertion %s failed in file %s line %d\n",#x, __FILE__, __LINE__);}

/** class IO */
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
		void progress(DREAL current_val, DREAL min_val=0.0, DREAL max_val=1.0, INT decimals=1, const char* prefix="PROGRESS:\t");

		/** print absolute progress bar
		 *
		 * @param current_val current value
		 * @param val value
		 * @param min_val minimum value
		 * @param max_val maximum value
		 * @param decimals decimals
		 * @param prefix message prefix
		 */
		void absolute_progress(DREAL current_val, DREAL val, DREAL min_val=0.0, DREAL max_val=1.0, INT decimals=1, const char* prefix="PROGRESS:\t");

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
		void buffered_message(EMessageType prio, const CHAR *fmt, ... ) const;

		/** skip leading spaces
		 *
		 * @param str string in which to look for spaces
		 * @return string after after skipping leading spaces
		 */
		static CHAR* skip_spaces(CHAR* str);

		/** skip leading spaces + tabs
		 *
		 * @param str string in which to look for blanks
		 * @return string after after skipping leading blanks
		 */
		static CHAR* skip_blanks(CHAR* str);

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
		inline void set_dirname(const CHAR* dirname)
		{
			strncpy(directory_name, dirname, FBUFSIZE);
		}

		/** concatenate directory and filename
		 * ( non thread safe )
		 *
		 * @param filename new filename
		 * @return concatenated directory and filename
		 */
		static CHAR* concat_filename(const CHAR* filename);

		/** filter
		 *
		 * @param d directory entry
		 * @return if filtering was successful
		 */
		static int filter(CONST_DIRENT_T* d);

	protected:
		/** get index into levels array
		 *
		 * @param prio message priority
		 * @return index into levels array or -1 if message could
		 *         not to be printed
		 */
		int get_prio_string(EMessageType prio) const;

	protected:
		/** target file */
		FILE* target;
		/** last progress time */
		LONG last_progress_time;
		/** progress start time */
		LONG progress_start_time;
		/** last progress */
		DREAL last_progress;
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
