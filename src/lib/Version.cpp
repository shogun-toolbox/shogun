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

	CIO::message(M_MESSAGEONLY, "genefinder (%s/%s/%s-%d)\n\n (C) 2000-2005 Soeren Sonnenburg, Gunnar Raetsch\n\n", TARGET, MACHINE, VERSION_RELEASE, get_version_in_minutes());
	CIO::message(M_MESSAGEONLY, "( configure options: \"%s\" compile flags: \"%s\" link flags: \"%s\" )\n", CONFIGURE_OPTIONS, COMPFLAGS_CPP, LINKFLAGS);
}


CVersion::~CVersion()
{
}

CVersion version;
