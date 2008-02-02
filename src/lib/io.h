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
#endif

#define ASSERT(x) { if (!(x)) SG_SERROR("assertion %s failed in file %s line %d\n",#x, __FILE__, __LINE__);}

class CIO
{
public:
	CIO();
	CIO(const CIO& orig);

	void set_loglevel(EMessageType level);
	EMessageType get_loglevel() const;
	bool get_show_progress() const;
	void message(EMessageType prio, const char *fmt, ... ) const;
	void progress(DREAL current_val, DREAL min_val=0.0, DREAL max_val=1.0, INT decimals=1, const char* prefix="PROGRESS:\t");
	void absolute_progress(DREAL current_val, DREAL val, DREAL min_val=0.0, DREAL max_val=1.0, INT decimals=1, const char* prefix="PROGRESS:\t");

	inline void not_implemented() const
	{
		message(M_ERROR, "Sorry, not yet implemented\n");
	}

	void buffered_message(EMessageType prio, const CHAR *fmt, ... ) const;
	static CHAR* skip_spaces(CHAR* str);

	inline FILE* get_target() const
	{
		return target;
	}

	void set_target(FILE* target);

	inline void set_target_to_stderr()
	{
		set_target(stderr);
	}

	inline void set_target_to_stdout()
	{
		set_target(stdout);
	}

	inline void enable_progress()
	{
		show_progress=true;

// static functions like CSVM::classify_example_helper call SG_PROGRESS
#ifndef HAVE_SWIG
		if (sg_io!=this)
			sg_io->enable_progress();
#else
		if (&sg_io!=this)
			sg_io.disable_progress();
#endif
	}

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

	///set directory-name
	inline void set_dirname(const CHAR* dirname)
	{
		strncpy(directory_name, dirname, FBUFSIZE);
	}

	///concatenate directory and filename
	/// ( non thread safe )
	static CHAR* concat_filename(const CHAR* filename);

	///concatenate directory and filename
	/// ( non thread safe )
	static int filter(CONST_DIRENT_T* d); 

protected:
	//return index into levels array or -1 if message not to be printed
	int get_prio_string(EMessageType prio) const;

protected:
	FILE* target;
	LONG last_progress_time, progress_start_time;
	DREAL last_progress;
	bool show_progress;

	EMessageType loglevel;
	static const EMessageType levels[NUM_LOG_LEVELS];
	static const char* message_strings[NUM_LOG_LEVELS];
};

#endif // __CIO_H__
