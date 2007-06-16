/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "base/Version.h"
#include "lib/versionstring.h"

const CHAR* CVersion::version_extra = VERSION_EXTRA;
const CHAR* CVersion::version_release = VERSION_RELEASE;

const INT CVersion::version_revision = VERSION_REVISION;
const INT CVersion::version_year = VERSION_YEAR;
const INT CVersion::version_month = VERSION_MONTH;
const INT CVersion::version_day = VERSION_DAY;
const INT CVersion::version_hour = VERSION_HOUR;
const INT CVersion::version_minute = VERSION_MINUTE;

CVersion::CVersion()
{
}


CVersion::~CVersion()
{
}
