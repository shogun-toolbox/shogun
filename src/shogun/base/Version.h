/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#include <shogun/base/init.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/versionstring.h>
#include <shogun/lib/config.h>

#ifndef VERSION_H__
#define VERSION_H__

namespace shogun
{
class IO;

/** @brief Class Version provides version information.
 *
 * It provides information of the version of shogun that is currently used, for
 * example the svn revision, time and date of compile and compilation and
 * the linkflags used.
 */
class Version
{
public:
	/** constructor */
	Version();
	/** destructor */
	virtual ~Version();

	/** print version */
	static inline void print_version()
	{
		SG_SPRINT("libshogun (%s/%s%d)\n\n", MACHINE, VERSION_RELEASE, version_revision);
		SG_SPRINT("Copyright (C) 1999-2009 Fraunhofer Institute FIRST\n");
		SG_SPRINT("Copyright (C) 1999-2011 Max Planck Society\n");
		SG_SPRINT("Copyright (C) 2009-2011 Berlin Institute of Technology\n");
		SG_SPRINT("Written   (W) 1999-2011 Soeren Sonnenburg, Gunnar Raetsch et al.\n\n");
#ifndef USE_SVMLIGHT
		SG_SPRINT("This is free software; see the source for copying conditions.  There is NO\n");
		SG_SPRINT("warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\n");
#endif
		SG_SPRINT( "( configure options: \"%s\" compile flags: \"%s\" link flags: \"%s\" )\n", CONFIGURE_OPTIONS, COMPFLAGS_CPP, LINKFLAGS);
	}

	/** get version extra */
	static inline const char* get_version_extra()
	{
		return version_extra;
	}

	/** get version release */
	static inline const char* get_version_release()
	{
		return version_release;
	}

	/** get version revision */
	static inline int32_t get_version_revision()
	{
		return version_revision;
	}

	/** get version year */
	static inline int32_t get_version_year()
	{
		return version_year;
	}

	/** get version month */
	static inline int32_t get_version_month()
	{
		return version_month;
	}

	/** get version day */
	static inline int32_t get_version_day()
	{
		return version_day;
	}

	/** get version hour */
	static inline int32_t get_version_hour()
	{
		return version_hour;
	}

	/** get version minute */
	static inline int32_t get_version_minute()
	{
		return version_year;
	}

	/** get version parameter */
	static inline const int32_t get_version_parameter()
	{
		return version_parameter;
	}

	/** get version in minutes */
	static inline int64_t get_version_in_minutes()
	{
		return ((((version_year)*12 + version_month)*30 + version_day)* 24 + version_hour)*60 + version_minute;
	}

	/** ref object 
	 * @return ref count
	 */
	inline int32_t ref()
	{
		++refcount;
		return refcount;
	}

	/** ref count
	 * @return ref count 
	 */
	inline int32_t ref_count() const
	{
		return refcount;
	}

	/** unref object
	 * @return ref count
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

protected:
	/** version release */
	static const char version_release[128];
	/** version extra */
	static const char version_extra[128];

	/** version revision */
	static const int32_t version_revision;
	/** version year */
	static const int32_t version_year;
	/** version month */
	static const int32_t version_month;
	/** version day */
	static const int32_t version_day;
	/** version hour */
	static const int32_t version_hour;
	/** version minute */
	static const int32_t version_minute;
	/** version parameter */
	static const int32_t version_parameter;
private:
	int32_t refcount;
};
}
#endif
