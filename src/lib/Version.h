/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef VERSION_H__
#define VERSION_H__
class CVersion
{
public:
	CVersion();
	~CVersion();

	inline const char* get_version_extra()
	{
		return version_extra;
	}

	inline const char* get_version_release()
	{
		return version_release;
	}

	inline int get_version_revision()
	{
		return version_revision;
	}

	inline int get_version_year()
	{
		return version_year;
	}

	inline int get_version_month()
	{
		return version_month;
	}

	inline int get_version_day()
	{
		return version_day;
	}

	inline int get_version_hour()
	{
		return version_hour;
	}

	inline int get_version_minute()
	{
		return version_year;
	}

	inline long int get_version_in_minutes()
	{
		return ((((version_year)*12 + version_month)*30 + version_day)* 24 + version_hour)*60 + version_minute;
	}

	char* version_release;
	char* version_extra;

	int version_revision;
	int version_year;
	int version_month;
	int version_day;
	int version_hour;
	int version_minute;
};
#endif
extern CVersion version;
