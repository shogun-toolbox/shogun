#include "lib/Version.h"
#include "lib/io.h"
#include "lib/versionstring.h"

CVersion::CVersion()
{
	version_extra = VERSION_EXTRA;
	version_release = VERSION_RELEASE;
	version_year = VERSION_YEAR;
	version_month = VERSION_MONTH;
	version_day = VERSION_DAY;
	version_hour = VERSION_HOUR;
	version_minute = VERSION_MINUTE;

	CIO::message(M_INFO, "genefinder_" VERSION_RELEASE " (%d) (w) 2000-2004 Soeren Sonnenburg, Gunnar Raetsch\n", get_version_in_minutes());
}


CVersion::~CVersion()
{
}

CVersion version;
