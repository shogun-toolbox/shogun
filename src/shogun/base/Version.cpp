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
