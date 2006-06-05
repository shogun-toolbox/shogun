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

	int version_year;
	int version_month;
	int version_day;
	int version_hour;
	int version_minute;
};
#endif
extern CVersion version;
