/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/versionstring.h"
#include "lib/io.h"
#include "lib/config.h"

#ifndef VERSION_H__
#define VERSION_H__
class CVersion
{
public:
	CVersion();
	~CVersion();

	static inline void print_version()
	{
		SG_SPRINT("shogun (%s/%s/%s%d)\n\n", TARGET, MACHINE, VERSION_RELEASE, version_revision);
		SG_SPRINT("Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society\n");
		SG_SPRINT("Written   (W) 2000-2007 Soeren Sonnenburg, Gunnar Raetsch et.al.\n\n");
#ifdef GPL
		SG_SPRINT("This is free software; see the source for copying conditions.  There is NO\n");
		SG_SPRINT("warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\n");
#endif
		SG_SPRINT( "( configure options: \"%s\" compile flags: \"%s\" link flags: \"%s\" )\n", CONFIGURE_OPTIONS, COMPFLAGS_CPP, LINKFLAGS);
	}

	static inline const CHAR* get_version_extra()
	{
		return version_extra;
	}

	static inline const CHAR* get_version_release()
	{
		return version_release;
	}

	static inline INT get_version_revision()
	{
		return version_revision;
	}

	static inline INT get_version_year()
	{
		return version_year;
	}

	static inline INT get_version_month()
	{
		return version_month;
	}

	static inline INT get_version_day()
	{
		return version_day;
	}

	static inline INT get_version_hour()
	{
		return version_hour;
	}

	static inline INT get_version_minute()
	{
		return version_year;
	}

	static inline LONG get_version_in_minutes()
	{
		return ((((version_year)*12 + version_month)*30 + version_day)* 24 + version_hour)*60 + version_minute;
	}

	static const CHAR* version_release;
	static const CHAR* version_extra;

	static const INT version_revision;
	static const INT version_year;
	static const INT version_month;
	static const INT version_day;
	static const INT version_hour;
	static const INT version_minute;
};
#endif
