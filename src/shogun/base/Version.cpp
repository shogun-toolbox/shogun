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
#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/RefCount.h>

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

Version::Version()
{
	m_refcount = new RefCount();
}


Version::~Version()
{
	delete m_refcount;
}

void Version::print_version()
{
	SG_SPRINT("libshogun (%s/%s%d)\n\n", MACHINE, VERSION_RELEASE, version_revision)
	SG_SPRINT("Copyright (C) 1999-2009 Fraunhofer Institute FIRST\n")
	SG_SPRINT("Copyright (C) 1999-2011 Max Planck Society\n")
	SG_SPRINT("Copyright (C) 2009-2011 Berlin Institute of Technology\n")
	SG_SPRINT("Copyright (C) 2012-2014 Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann, Viktor Gal, Fernando Iglesias et al\n")
	SG_SPRINT("Written   (W) 1999-2012 Soeren Sonnenburg, Gunnar Raetsch et al.\n\n")
#ifndef USE_SVMLIGHT
	SG_SPRINT("This is free software; see the source for copying conditions.  There is NO\n")
	SG_SPRINT("warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\n")
#endif

#ifdef LINKFLAGS
	SG_SPRINT("( configure options: \"%s\" compile flags: \"%s\" link flags: \"%s\" )\n", 
		CONFIGURE_OPTIONS, COMPFLAGS_CPP, LINKFLAGS)
#else
	SG_SPRINT("( configure options: \"%s\" compile flags: \"%s\" link flags: None )\n", 
		CONFIGURE_OPTIONS, COMPFLAGS_CPP)
#endif
}

const char* Version::get_version_extra()
{
	return version_extra;
}

const char* Version::get_version_release()
{
	return version_release;
}

int32_t Version::get_version_revision()
{
	return version_revision;
}

int32_t Version::get_version_year()
{
	return version_year;
}

int32_t Version::get_version_month()
{
	return version_month;
}

int32_t Version::get_version_day()
{
	return version_day;
}

int32_t Version::get_version_hour()
{
	return version_hour;
}

int32_t Version::get_version_minute()
{
	return version_year;
}

int32_t Version::get_version_parameter()
{
	return version_parameter;
}

int64_t Version::get_version_in_minutes()
{
	return ((((version_year)*12 + version_month)*30 + version_day)* 24 + version_hour)*60 + version_minute;
}

int32_t Version::ref()
{
	return m_refcount->ref();
}

int32_t Version::ref_count() const
{
	return m_refcount->ref_count();
}

int32_t Version::unref()
{
	int32_t rc = m_refcount->unref();

	if (rc==0)
	{
		delete this;
		return 0;
	}

	return rc;
}
