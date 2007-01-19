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

#ifndef VERSION_H__
#define VERSION_H__
class CVersion
{
public:
	CVersion();
	~CVersion();

	inline const CHAR* get_version_extra()
	{
		return version_extra;
	}

	inline const CHAR* get_version_release()
	{
		return version_release;
	}

	inline INT get_version_revision()
	{
		return version_revision;
	}

	inline INT get_version_year()
	{
		return version_year;
	}

	inline INT get_version_month()
	{
		return version_month;
	}

	inline INT get_version_day()
	{
		return version_day;
	}

	inline INT get_version_hour()
	{
		return version_hour;
	}

	inline INT get_version_minute()
	{
		return version_year;
	}

	inline LONG get_version_in_minutes()
	{
		return ((((version_year)*12 + version_month)*30 + version_day)* 24 + version_hour)*60 + version_minute;
	}

	CHAR* version_release;
	CHAR* version_extra;

	INT version_revision;
	INT version_year;
	INT version_month;
	INT version_day;
	INT version_hour;
	INT version_minute;
};
#endif
