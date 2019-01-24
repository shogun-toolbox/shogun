/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Yuyu Zhang, Viktor Gal, 
 *          Thoralf Klein, Evan Shelhamer, Sergey Lisitsyn, 
 *          Evangelos Anagnostopoulos
 */

#include <shogun/lib/common.h>

#ifndef VERSION_H__
#define VERSION_H__

#include <shogun/lib/config.h>
#include <shogun/lib/versionstring.h>

namespace shogun
{
class RefCount;

/** @brief Class Version provides version information.
 *
 * It provides information of the version of shogun that is currently used, for
 * example the svn revision, time and date of compile and compilation and
 * the linkflags used.
 */
class Version
{
public:
	/** constructor */
	Version();
	/** destructor */
	virtual ~Version();

	/** print version */
	static void print_version();

	/** get main version */
    static constexpr const char* get_version_main() {
        return version_main;
    }

	/** get version extra */
	static constexpr const char* get_version_extra() {
	    return version_extra;
	}

	/** get version release */
	static constexpr const char* get_version_release() {
	    return version_release;
	}

	/** get version revision */
    static constexpr int64_t get_version_revision() {
        return version_revision;
    }

	/** get version year */
    static constexpr int32_t get_version_year() {
        return version_year;
    }

	/** get version month */
    static constexpr int32_t get_version_month() {
        return version_month;
    }

	/** get version day */
    static constexpr int32_t get_version_day() {
        return version_day;
    }

	/** get version hour */
    static constexpr int32_t get_version_hour() {
        return version_hour;
    }

	/** get version minute */
    static constexpr int32_t get_version_minute() {
        return version_minute;
    }

	/** get parameter serialization version */
    static constexpr int32_t get_version_parameter() {
        return version_parameter;
    }

	/** get version in minutes */
    static constexpr int64_t get_version_in_minutes() {
        return version_minute;
    }

	/** ref object
	 * @return ref count
	 */
	int32_t ref();

	/** ref count
	 * @return ref count
	 */
	int32_t ref_count() const;

	/** unref object
	 * @return ref count
	 */
	int32_t unref();

protected:
	/** version release */
	static constexpr char version_release[128] = VERSION_RELEASE;
	/** version extra */
	static constexpr char version_extra[128] = VERSION_EXTRA;

	/** version revision */
	static constexpr int64_t version_revision = VERSION_REVISION;
	/** version year */
	static constexpr int32_t version_year = VERSION_YEAR;
	/** version month */
	static constexpr int32_t version_month = VERSION_MONTH;
	/** version day */
	static constexpr int32_t version_day = VERSION_DAY;
	/** version hour */
	static constexpr int32_t version_hour = VERSION_HOUR;
	/** version minute */
	static constexpr int32_t version_minute = VERSION_MINUTE;
	/** version parameter */
	static constexpr int32_t version_parameter = VERSION_PARAMETER;
    /** version main */
	static constexpr char version_main[32] = MAINVERSION;

private:
	RefCount* m_refcount;
};
}
#endif
