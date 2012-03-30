/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/base/Version.h>
#include <shogun/lib/versionstring.h>

using namespace shogun;

namespace shogun
{
const int32_t Version::version_revision = VERSION_REVISION;
const int32_t Version::version_year = VERSION_YEAR;
const int32_t Version::version_month = VERSION_MONTH;
const int32_t Version::version_day = VERSION_DAY;
const int32_t Version::version_hour = VERSION_HOUR;
const int32_t Version::version_minute = VERSION_MINUTE;
const int32_t Version::version_parameter=VERSION_PARAMETER;
const char Version::version_extra[128] = VERSION_EXTRA;
const char Version::version_release[128] = VERSION_RELEASE;
}

Version::Version() : refcount(0)
{
}


Version::~Version()
{
}

/** print version */
void Version::print_version()
{
	SG_SPRINT("libshogun (%s/%s%d)\n\n", MACHINE, VERSION_RELEASE, version_revision);
	SG_SPRINT("Copyright (C) 1999-2009 Fraunhofer Institute FIRST\n");
	SG_SPRINT("Copyright (C) 1999-2011 Max Planck Society\n");
	SG_SPRINT("Copyright (C) 2009-2011 Berlin Institute of Technology\n");
	SG_SPRINT("Copyright (C) 2012 Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann\n");
	SG_SPRINT("Written   (W) 1999-2012 Soeren Sonnenburg, Gunnar Raetsch et al.\n\n");
#ifndef USE_SVMLIGHT
	SG_SPRINT("This is free software; see the source for copying conditions.  There is NO\n");
	SG_SPRINT("warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\n");
#endif
	SG_SPRINT( "( configure options: \"%s\" compile flags: \"%s\" link flags: \"%s\" )\n", CONFIGURE_OPTIONS, COMPFLAGS_CPP, LINKFLAGS);
}

/** get version extra */
const char* Version::get_version_extra()
{
	return version_extra;
}

/** get version release */
const char* Version::get_version_release()
{
	return version_release;
}

/** get version revision */
int32_t Version::get_version_revision()
{
	return version_revision;
}

/** get version year */
int32_t Version::get_version_year()
{
	return version_year;
}

/** get version month */
int32_t Version::get_version_month()
{
	return version_month;
}

/** get version day */
int32_t Version::get_version_day()
{
	return version_day;
}

/** get version hour */
int32_t Version::get_version_hour()
{
	return version_hour;
}

/** get version minute */
int32_t Version::get_version_minute()
{
	return version_year;
}

/** get version parameter */
int32_t Version::get_version_parameter()
{
	return version_parameter;
}

/** get version in minutes */
int64_t Version::get_version_in_minutes()
{
	return ((((version_year)*12 + version_month)*30 + version_day)* 24 + version_hour)*60 + version_minute;
}

/** ref object
 * @return ref count
 */
int32_t Version::ref()
{
	++refcount;
	return refcount;
}

/** ref count
 * @return ref count
 */
int32_t Version::ref_count() const
{
	return refcount;
}

/** unref object
 * @return ref count
 */
int32_t Version::unref()
{
	if (refcount==0 || --refcount==0)
	{
		delete this;
		return 0;
	}
	else
		return refcount;
}
