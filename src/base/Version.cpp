/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "base/Version.h"
#include "lib/versionstring.h"


const int32_t CVersion::version_revision = VERSION_REVISION;
const int32_t CVersion::version_year = VERSION_YEAR;
const int32_t CVersion::version_month = VERSION_MONTH;
const int32_t CVersion::version_day = VERSION_DAY;
const int32_t CVersion::version_hour = VERSION_HOUR;
const int32_t CVersion::version_minute = VERSION_MINUTE;
const char CVersion::version_extra[128] = VERSION_EXTRA;
const char CVersion::version_release[128] = VERSION_RELEASE;

CVersion::CVersion() : refcount(0)
{
}


CVersion::~CVersion()
{
}
